import random
import sys
import config as cfg
import genetic as ga

def main():
    pop = ga.generate_initial_pop()
    opt_state, opt_value, iter, conv = ga.genetic(cfg.max_size, cfg.max_iter, cfg.cross_ratio, cfg.mut_ratio, cfg.elite_pct, pop)
    
main()
