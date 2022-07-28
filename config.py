import sys

# Configuracoes da rede neural
num_imputs = 4 # Numero de entradas da rede neural (distance_from_obstacle, obstacle_length, obstacle_height, game_speed)
num_outputs = 3 # Numero de saidas da rede neural (3: K_UP, K_DOWN, K_NO)
min_valor_peso = -1 # Limite inferior para o valor de um peso da rede
max_valor_peso = 1 # Limite superior para o valor de um peso da rede

# Configuracoes do algoritmo genetico
max_iter = 100000 # Numero maximo de iteracoes do algoritmo
max_time = 72000 # Tempo maximo em segundos de execucao do algoritmo (1h = 3600s, 15h = 54000s, 24h = 86400s)
cross_ratio = 0.9 # Taxa de crossover
mut_ratio = 0.1 # Taxa de mutacao
gene_mut_ratio = 0.1 # Taxa de genes a serem mutados
elite_pct = 3 # Taxa de elitismo
num_individuals_per_pop = 100 # Numero de individuos por populacao
