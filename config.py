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

# OPTIMAL_VALUE SO FAR:
# [0.677133121970197, 0.7094266673657799, 0.21470962511078717, -0.4082097776547975, 
# -0.524526295563561, 0.8976795084064886, 0.5990224331347083, 0.1955927258335033, -0.4048968668259689, 
# 0.11983010782604309, 0.3893727882561362, -0.47883310532259915, 0.5903531680083913, 
# -0.09714096801146921, 0.12351035221735018, -0.6584796962419717, -0.7727282085317522, 
# -0.4501892963271028, 0.8614196477707066, 0.8541648304396299, -0.5884773631428817, -0.7222372133369119, 
# -0.414581410094808, -0.09275172805684795, -0.051280823467390046, 0.6405120252949597, 
# -0.3344616147600876, 0.700630747188012, -0.8947656648277598, -0.04987974001020734, 0.7830195091089578, 
# 0.005149921599586049, -0.7870677936668058, 0.5362956095450957, 0.8900313456626952, 0.2860734491991441, 
# -0.18384750099573433, 0.5535427461997562, 0.1490829310459838, -0.5149428450760056,
# 0.14692042565394692, -0.031931135987872894, -0.1929810860185912]