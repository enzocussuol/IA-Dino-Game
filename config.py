import sys

# Configuracoes da rede neural
num_imputs = 5 # Numero de entradas da rede neural (distance_from_obstacle, obstacle_length, obstacle_height, dino_height, game_speed)
num_outputs = 3 # Numero de saidas da rede neural (3: K_UP, K_DOWN, K_NO)
min_valor_peso = -1 # Limite inferior para o valor de um peso da rede
max_valor_peso = 1 # Limite superior para o valor de um peso da rede

# Configuracoes do algoritmo genetico
max_size = sys.maxsize # Numero maximos de individuos em uma populacao
max_iter = 10 # Numero maximo de iteracoes do algoritmo
cross_ratio = 0.9 # Taxa de crossover
mut_ratio = 0.1 # Taxa de mutacao
elite_pct = 20 # Taxa de elitismo
num_individuals_per_pop = 5 # Numero inicial de individuos por populacao
num_rounds_evaluation = 3 # Numero de round a serem jogados para avaliar o desempenho do individuo