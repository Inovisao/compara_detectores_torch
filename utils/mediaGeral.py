import pandas as pd

# Carregar o arquivo CSV
file_path = '../results/results.csv'
data = pd.read_csv(file_path)

# Excluir colunas não numéricas
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calcular a média e o desvio padrão das métricas para cada modelo ('ml')
# Presumindo que 'ml' é a coluna pela qual você deseja agrupar
mean_metrics = numeric_data.groupby(data['ml']).mean()
std_metrics = numeric_data.groupby(data['ml']).std()

# Resetar o índice para ter 'ml' como coluna
mean_metrics = mean_metrics.reset_index()
std_metrics = std_metrics.reset_index()

# Salvar os resultados em arquivos CSV separados
mean_output_file_path = '../results/mean_metrics.csv'
std_output_file_path = '../results/std_metrics.csv'

mean_metrics.to_csv(mean_output_file_path, index=False)
std_metrics.to_csv(std_output_file_path, index=False)

# Exibir os resultados
print("Média das métricas:")
print(mean_metrics)
print("\nDesvio padrão das métricas:")
print(std_metrics)
