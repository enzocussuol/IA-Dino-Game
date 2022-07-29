import numpy as np
import config as cfg
import genetic as ga
import neuralNetwork as nn
from game import playGame

def main():
    pop = ga.generate_initial_pop()
    opt_state, opt_value = ga.genetic(cfg.max_iter, cfg.max_time, cfg.cross_ratio, cfg.mut_ratio, cfg.elite_pct, pop)

    print("============================")
    print("FIM DO TREINAMENTO")
    print("opt_state:")
    print(opt_state)

    print("opt_value: ", opt_value)

    ai_player = nn.NN(cfg.num_imputs, cfg.num_outputs)
    nn.vector_to_parameters(nn.Tensor(opt_state), ai_player.parameters())

    # Caso queira jogar com o melhor estado ate entao, sem realizar nenhum treino, comente as linhas acima dessa e descomente a atribuicao abaixo
    # opt_state = [0.677133121970197, 0.7094266673657799, 0.21470962511078717, -0.4082097776547975, -0.524526295563561, 
    #         0.8976795084064886, 0.5990224331347083, 0.1955927258335033, -0.4048968668259689, 0.11983010782604309, 
    #         0.3893727882561362, -0.47883310532259915, 0.5903531680083913, -0.09714096801146921, 0.12351035221735018, 
    #         -0.6584796962419717, -0.7727282085317522, -0.4501892963271028, 0.8614196477707066, 0.8541648304396299, 
    #         -0.5884773631428817, -0.7222372133369119, -0.414581410094808, -0.09275172805684795, -0.051280823467390046, 0.6405120252949597, 
    #         -0.3344616147600876, 0.700630747188012, -0.8947656648277598, -0.04987974001020734, 0.7830195091089578, 0.005149921599586049, 
    #         -0.7870677936668058, 0.5362956095450957, 0.8900313456626952, 0.2860734491991441, -0.18384750099573433, 0.5535427461997562, 
    #         0.1490829310459838, -0.5149428450760056, 0.14692042565394692, -0.031931135987872894, -0.1929810860185912]
    
    results_and_states = []
    for i in range(30):
        print("============================")
        print("RODADA DE VALIDACAO: ", i)
        results_and_states.append(playGame([(ai_player, opt_state)], treino = False))
    
    results = []
    for result_and_state in results_and_states:
        results.append(result_and_state[0][0])
    
    npRes = np.asarray(results)

    print("============================")
    print("RESULTADOS FINAIS: ")
    print(results, npRes.mean(), npRes.std())

main()