import numpy as np
import config as cfg
import genetic as ga
import neuralNetwork as nn
from game import playGame

def main():
    pop = ga.generate_initial_pop()
    opt_state, opt_value = ga.genetic(cfg.max_iter, cfg.max_time, cfg.cross_ratio, cfg.mut_ratio, cfg.elite_pct, pop)

    print("opt_state:")
    print(opt_state)

    print("Optimal value: ", opt_value)

    ai_player = nn.NN(cfg.num_imputs, cfg.num_outputs)
    nn.vector_to_parameters(nn.Tensor(opt_state), ai_player.parameters())
    
    results_and_states = []
    for i in range(30):
        results_and_states.append(playGame([(ai_player, opt_state)]))
    
    results = []
    for result_and_state in results_and_states:
        results.append(result_and_state[0][0])
    
    npRes = np.asarray(results)
    print(results, npRes.mean(), npRes.std())

main()