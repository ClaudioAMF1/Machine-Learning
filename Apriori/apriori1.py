import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. Carregamento e Pré-processamento
def load_and_preprocess_data(url):
    """Carrega e pré-processa os dados"""
    print("Carregando e pré-processando dados...")
    
    # Carregar dados
    import requests
    import zipfile
    import io
    
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('online_retail_dataset.csv') as f:
            df = pd.read_csv(f)
    
    # Pré-processamento básico
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]  # Remove cancelamentos
    df = df[df['Quantity'] > 0]  # Remove quantidades negativas
    df = df[df['UnitPrice'] > 0]  # Remove preços zero ou negativos
    
    # Remover outliers
    df = df[df['Quantity'] <= df['Quantity'].quantile(0.95)]
    df = df[df['UnitPrice'] <= df['UnitPrice'].quantile(0.95)]
    
    # Normalizar descrições
    df['Description'] = df['Description'].str.strip().str.upper()
    
    # Adicionar informação temporal
    df['Month'] = df['InvoiceDate'].dt.month
    df['WeekDay'] = df['InvoiceDate'].dt.dayofweek
    
    return df

# 2. Criar transações válidas
def create_valid_transactions(df):
    """Cria transações válidas a partir do DataFrame"""
    print("Criando transações válidas...")
    
    # Calcular tamanho das transações
    transaction_sizes = df.groupby('InvoiceNo').size()
    
    # Filtrar transações válidas (entre 2 e 95º percentil)
    valid_size = transaction_sizes[
        (transaction_sizes >= 2) & 
        (transaction_sizes <= transaction_sizes.quantile(0.95))
    ].index
    
    # Criar lista de transações
    transactions = df[df['InvoiceNo'].isin(valid_size)].groupby('InvoiceNo')['Description'].agg(list)
    
    return transactions.tolist()

# 3. Gerar regras com validação temporal
def generate_rules_by_period(df, min_support=0.05, min_confidence=0.7, max_lift=20):
    """Gera regras por período"""
    print("Gerando regras por período...")
    
    rules_by_month = {}
    
    for month in df['Month'].unique():
        # Filtrar dados do mês
        month_data = df[df['Month'] == month]
        
        # Criar transações do mês
        month_transactions = create_valid_transactions(month_data)
        
        # Codificar transações
        te = TransactionEncoder()
        te_ary = te.fit(month_transactions).transform(month_transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Gerar itemsets frequentes
        frequent_itemsets = apriori(df_encoded, 
                                  min_support=min_support, 
                                  use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            # Gerar regras
            rules = association_rules(frequent_itemsets, 
                                    metric="confidence",
                                    min_threshold=min_confidence)
            
            # Filtrar por lift
            rules = rules[rules['lift'] <= max_lift]
            
            rules_by_month[month] = rules
            print(f"Mês {month}: {len(rules)} regras encontradas")
    
    return rules_by_month

# 4. Validar regras
def validate_rules(rules_by_month, min_periods=2):
    """Valida regras entre períodos"""
    print("Validando regras...")
    
    all_rules = []
    
    # Combinar regras de todos os períodos
    for month, rules in rules_by_month.items():
        if len(rules) > 0:
            rules['month'] = month
            all_rules.append(rules)
    
    if len(all_rules) > 0:
        combined_rules = pd.concat(all_rules)
        
        # Contar ocorrências de cada regra
        rule_counts = combined_rules.groupby(['antecedents', 'consequents']).size()
        
        # Filtrar regras que aparecem em múltiplos períodos
        consistent_rules = rule_counts[rule_counts >= min_periods].index
        
        # Filtrar regras consistentes
        validated_rules = combined_rules[
            combined_rules.set_index(['antecedents', 'consequents']).index.isin(consistent_rules)
        ]
        
        return validated_rules
    
    return pd.DataFrame()

# 5. Analisar e apresentar resultados
def analyze_results(rules, df):
    """Analisa e apresenta os resultados"""
    print("\n=== Análise das Regras ===")
    
    # Top regras por diferentes métricas
    print("\nTop 5 Regras por Confiança:")
    print(rules.nlargest(5, 'confidence')[
        ['antecedents', 'consequents', 'confidence', 'lift', 'support']
    ])
    
    print("\nTop 5 Regras por Lift (realista):")
    print(rules.nlargest(5, 'lift')[
        ['antecedents', 'consequents', 'confidence', 'lift', 'support']
    ])
    
    # Estatísticas das métricas
    print("\nEstatísticas das Métricas:")
    metrics_stats = rules[['support', 'confidence', 'lift']].describe()
    print(metrics_stats)
    
    # Análise de valor
    print("\nAnálise de Valor das Regras:")
    df_value = df.copy()
    df_value['TotalValue'] = df_value['Quantity'] * df_value['UnitPrice']
    
    value_by_product = df_value.groupby('Description')['TotalValue'].sum()
    
    for _, rule in rules.nlargest(5, 'lift').iterrows():
        ant_products = list(rule['antecedents'])
        cons_products = list(rule['consequents'])
        
        ant_value = sum(value_by_product[ant_products])
        cons_value = sum(value_by_product[cons_products])
        
        print(f"\nRegra: {ant_products} -> {cons_products}")
        print(f"Valor Total Antecedentes: £{ant_value:.2f}")
        print(f"Valor Total Consequentes: £{cons_value:.2f}")

# 6. Função principal
def main():
    # Parâmetros
    MIN_SUPPORT = 0.05    # 5% de suporte mínimo
    MIN_CONFIDENCE = 0.7  # 70% de confiança mínima
    MAX_LIFT = 20        # Limite máximo de lift
    MIN_PERIODS = 2      # Número mínimo de períodos para validação
    
    # URL do dataset
    url = "https://raw.githubusercontent.com/klaytoncastro/idp-machinelearning/main/resources/online_retail.zip"
    
    # 1. Carregar e pré-processar dados
    df = load_and_preprocess_data(url)
    
    # 2. Gerar regras por período
    rules_by_month = generate_rules_by_period(df, MIN_SUPPORT, MIN_CONFIDENCE, MAX_LIFT)
    
    # 3. Validar regras
    validated_rules = validate_rules(rules_by_month, MIN_PERIODS)
    
    # 4. Analisar resultados
    if len(validated_rules) > 0:
        analyze_results(validated_rules, df)
        
        # Salvar resultados
        validated_rules.to_csv('regras_validadas.csv', index=False)
        print("\nResultados salvos em 'regras_validadas.csv'")
    else:
        print("\nNenhuma regra válida encontrada com os parâmetros atuais.")
        print("Tente ajustar os parâmetros (diminuir suporte ou confiança)")

if __name__ == "__main__":
    main()
