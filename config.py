import sys

# Configuracoes da rede neural
num_imputs = 5 # Numero de entradas da rede neural (distance_from_obstacle, obstacle_length, obstacle_height, dino_height, game_speed)
num_outputs = 3 # Numero de saidas da rede neural (3: K_UP, K_DOWN, K_NO)
min_valor_peso = -1 # Limite inferior para o valor de um peso da rede
max_valor_peso = 1 # Limite superior para o valor de um peso da rede

# Configuracoes do algoritmo genetico
max_iter = 100000 # Numero maximo de iteracoes do algoritmo
max_time = 28800 # Tempo maximo em segundos de execucao do algoritmo (1h = 3600s, 24h = 86400s)
cross_ratio = 0.75 # Taxa de crossover
mut_ratio = 0.05 # Taxa de mutacao
elite_pct = 20 # Taxa de elitismo
num_individuals_per_pop = 100 # Numero de individuos por populacao