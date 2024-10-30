import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(url):
    """Carrega e pré-processa os dados"""
    print("Carregando e pré-processando dados...")
    
    import requests
    import zipfile
    import io
    
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('online_retail_dataset.csv') as f:
            df = pd.read_csv(f)
    
    # Pré-processamento básico
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    
    # Ajuste mais suave para outliers (97.5º percentil em vez de 95º)
    df = df[df['Quantity'] <= df['Quantity'].quantile(0.975)]
    df = df[df['UnitPrice'] <= df['UnitPrice'].quantile(0.975)]
    
    # Normalizar descrições
    df['Description'] = df['Description'].str.strip().str.upper()
    
    # Adicionar informação temporal
    df['Month'] = df['InvoiceDate'].dt.month
    df['WeekDay'] = df['InvoiceDate'].dt.dayofweek
    
    return df

def create_valid_transactions(df):
    """Cria transações válidas com critérios mais flexíveis"""
    print("Criando transações válidas...")
    
    # Critério mais flexível para tamanho das transações
    transaction_sizes = df.groupby('InvoiceNo').size()
    valid_size = transaction_sizes[
        (transaction_sizes >= 2) & 
        (transaction_sizes <= transaction_sizes.quantile(0.975))
    ].index
    
    transactions = df[df['InvoiceNo'].isin(valid_size)].groupby('InvoiceNo')['Description'].agg(list)
    return transactions.tolist()

def generate_rules_by_period(df, min_support=0.02, min_confidence=0.4, max_lift=30):
    """Gera regras por período com parâmetros mais flexíveis"""
    print("Gerando regras por período...")
    
    rules_by_month = {}
    
    for month in df['Month'].unique():
        month_data = df[df['Month'] == month]
        month_transactions = create_valid_transactions(month_data)
        
        te = TransactionEncoder()
        te_ary = te.fit(month_transactions).transform(month_transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(df_encoded, 
                                  min_support=min_support, 
                                  use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, 
                                    metric="confidence",
                                    min_threshold=min_confidence)
            
            # Filtro mais flexível para lift
            rules = rules[rules['lift'] <= max_lift]
            
            # Adicionar mês para rastreamento
            rules['month'] = month
            
            rules_by_month[month] = rules
            print(f"Mês {month}: {len(rules)} regras encontradas")
    
    return rules_by_month

def validate_rules(rules_by_month, min_periods=1):
    """Valida regras com critérios mais flexíveis"""
    print("Validando regras...")
    
    all_rules = []
    
    for month, rules in rules_by_month.items():
        if len(rules) > 0:
            all_rules.append(rules)
    
    if len(all_rules) > 0:
        validated_rules = pd.concat(all_rules)
        
        # Agrupar regras similares
        validated_rules['rule_key'] = validated_rules.apply(
            lambda x: f"{sorted(list(x['antecedents']))}→{sorted(list(x['consequents']))}", 
            axis=1
        )
        
        # Calcular métricas médias por regra
        aggregated_rules = validated_rules.groupby('rule_key').agg({
            'antecedents': 'first',
            'consequents': 'first',
            'support': 'mean',
            'confidence': 'mean',
            'lift': 'mean',
            'month': 'count'  # Conta em quantos meses a regra aparece
        }).reset_index()
        
        # Ordenar por contagem de meses e depois por lift
        aggregated_rules = aggregated_rules.sort_values(['month', 'lift'], ascending=[False, False])
        
        return aggregated_rules
    
    return pd.DataFrame()

def analyze_results(rules, df):
    """Análise mais detalhada dos resultados"""
    print("\n=== Análise das Regras ===")
    
    # Top regras por diferentes métricas
    print("\nTop 10 Regras por Confiança:")
    print(rules.nlargest(10, 'confidence')[
        ['antecedents', 'consequents', 'confidence', 'lift', 'support', 'month']
    ])
    
    print("\nTop 10 Regras por Lift:")
    print(rules.nlargest(10, 'lift')[
        ['antecedents', 'consequents', 'confidence', 'lift', 'support', 'month']
    ])
    
    print("\nTop 10 Regras mais Frequentes (por número de meses):")
    print(rules.nlargest(10, 'month')[
        ['antecedents', 'consequents', 'confidence', 'lift', 'support', 'month']
    ])
    
    # Estatísticas das métricas
    print("\nEstatísticas das Métricas:")
    metrics_stats = rules[['support', 'confidence', 'lift', 'month']].describe()
    print(metrics_stats)
    
    # Análise de valor
    print("\nAnálise de Valor das Regras Mais Frequentes:")
    df_value = df.copy()
    df_value['TotalValue'] = df_value['Quantity'] * df_value['UnitPrice']
    value_by_product = df_value.groupby('Description')['TotalValue'].sum()
    
    for _, rule in rules.nlargest(5, 'month').iterrows():
        ant_products = list(rule['antecedents'])
        cons_products = list(rule['consequents'])
        
        try:
            ant_value = sum(value_by_product[ant_products])
            cons_value = sum(value_by_product[cons_products])
            
            print(f"\nRegra (presente em {int(rule['month'])} meses):")
            print(f"Antecedentes: {ant_products}")
            print(f"Consequentes: {cons_products}")
            print(f"Valor Total Antecedentes: £{ant_value:.2f}")
            print(f"Valor Total Consequentes: £{cons_value:.2f}")
            print(f"Confiança: {rule['confidence']:.2%}")
            print(f"Lift: {rule['lift']:.2f}")
        except:
            continue

def main():
    # Parâmetros ajustados para encontrar mais regras
    MIN_SUPPORT = 0.02     # Reduzido de 0.05 para 0.02
    MIN_CONFIDENCE = 0.4   # Reduzido de 0.7 para 0.4
    MAX_LIFT = 30         # Aumentado de 20 para 30
    MIN_PERIODS = 1       # Reduzido de 2 para 1
    
    url = "https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/resources/online_retail.zip"
    
    df = load_and_preprocess_data(url)
    rules_by_month = generate_rules_by_period(df, MIN_SUPPORT, MIN_CONFIDENCE, MAX_LIFT)
    validated_rules = validate_rules(rules_by_month, MIN_PERIODS)
    
    if len(validated_rules) > 0:
        analyze_results(validated_rules, df)
        validated_rules.to_csv('regras_validadas_ajustadas.csv', index=False)
        print("\nResultados salvos em 'regras_validadas_ajustadas.csv'")
    else:
        print("\nNenhuma regra válida encontrada mesmo com parâmetros ajustados.")
        print("Considere reduzir ainda mais os parâmetros ou revisar os dados.")

if __name__ == "__main__":
    main()
