import sys

# Configuracoes da rede neural
num_imputs = 3 # Numero de entradas da rede neural (distance, speed, objHeight, etc...)
num_outputs = 3 # Numero de saidas da rede neural (3: KEY_UP, KEY_DOWN, NO_KEY)
num_pesos = 3
min_valor_peso = -1
max_valor_peso = 1

# Configuracoes do algoritmo genetico
max_size = sys.maxsize # Numero maximos de individuos em uma populacao
max_iter = 3 # Numero de iteracoes do algoritmo
cross_ratio = 0.9 # Taxa de crossover
mut_ratio = 0.1 # Taxa de mutacao
elite_pct = 20 # Taxa de elitismo
num_individuals_per_pop = 3 # Numero inicial de individuos por populacao
tam_individual = 3 # Tamanho do individuo (numero de pesos a serem obtidos da rede neural)
num_rounds_evaluation = 3 # Numero de round a serem jogados para avaliar o desempenho do individuo