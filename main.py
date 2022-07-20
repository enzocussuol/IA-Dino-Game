import random
import sys
import config as cfg
import genetic as ga

def main():
    pop = []

    for i in range(0, cfg.num_individuals_per_pop):
        individual = []
        for j in range(0, cfg.num_pesos):
            individual.append(random.uniform(cfg.min_valor_peso, cfg.max_valor_peso))
        pop.append(individual)

    print("Initial population:")
    print(pop)

    opt_state, opt_value, iter, conv = ga.genetic(cfg.max_size, cfg.max_iter, cfg.cross_ratio, cfg.mut_ratio, cfg.elite_pct, pop)

    print("\nBest individual:")
    print(opt_state)

    print("\nBest value:")
    print(opt_value)
    
main()
