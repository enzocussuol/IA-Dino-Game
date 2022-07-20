import random
import math
import numpy as np
import config as cfg
import neuralNetwork as nn
from game import playGame

def first(x):
    return x[0]

def roulette_construction(val_pop):
    aux_states = []
    roulette = []
    total_value = 0

    for v, s in val_pop:
        total_value = total_value + v

    for v, s in val_pop:
        if total_value != 0:
            ratio = v/total_value
        else:
            ratio = 1
        aux_states.append((ratio, s))

    acc_value = 0
    for state in aux_states:
        acc_value = acc_value + state[0]
        s = (acc_value, state[1])
        roulette.append(s)
    return roulette

def roulette_run(roulette, rounds):
    if roulette == []:
        return []
    selected = []
    while len(selected) < rounds:
        r = random.uniform(0, 1)
        for state in roulette:
            if r <= state[0]:
                selected.append(state[1])
                break
    return selected

def evaluate_state(s):
    results = []
    for round in range(0, cfg.num_rounds_evaluation):
        results += [playGame(ai_player)]
    npResults = np.asarray(results)
    return npResults.mean()

def evaluate_population(pop):
    eval = []
    for s in pop:
        eval = eval + [(evaluate_state(s), s)]
    return eval

def convergent(pop):
    conv = False
    if pop != []:
        base = pop[0]
        i = 0
        while i < len(pop):
            if base != pop[i]:
                return False
            i += 1
        return True

def elitism(val_pop, pct):
    n = math.floor((pct/100)*len(val_pop))
    if n < 1:
        n = 1
    val_elite = sorted(val_pop, key = first, reverse = True)[:n]
    elite = [s for v, s in val_elite]
    return elite

def selection(val_pop, n):
    roulette = roulette_construction(val_pop)
    new_population = roulette_run(roulette, n)
    return new_population

def crossover(dad, mom):
    r = random.randint(0, len(dad) - 1)
    son = dad[:r] + mom[r:]
    daug = mom[:r] + dad[r:]
    return son, daug

def crossover_step(pop, pop_size, cross_ratio, max_size):
    new_pop = []

    for _ in range(round(len(pop)/2)):
        rand = random.uniform(0, 1)

        fst_ind = random.randint(0, len(pop) - 1)
        scd_ind = random.randint(0, len(pop) - 1)

        parent1 = pop[fst_ind]
        parent2 = pop[scd_ind]

        if pop_size + 2 <= max_size and rand <= cross_ratio:
            offspring1, offspring2 = crossover(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2

        new_pop = new_pop + [offspring1, offspring2]

    return new_pop

def mutation(s):
    state = s.copy()
    rand = random.randint(0, len(state) - 1)
    state[rand] = random.uniform(cfg.min_valor_peso, cfg.max_valor_peso)
    return state

def mutation_step(pop, mut_ratio):
    ind = 0
    for s in pop:
        rand = random.uniform(0, 1)

        if rand <= mut_ratio:
            mutated = mutation(s)
            pop[ind] = mutated
        ind += 1
    return pop

def genetic(max_size, max_iter, cross_ratio, mut_ratio, elite_pct, pop):
    pop_size = len(pop)

    opt_state = [0] * pop_size
    opt_value = 0
    conv = convergent(pop)
    iter = 0

    global ai_player
    ai_player = nn.NN(cfg.num_imputs, cfg.num_outputs)

    while not conv and iter < max_iter:
        val_pop = evaluate_population(pop)
        print("\nPopulation evaluation:")
        print(val_pop)
        new_pop = elitism(val_pop, elite_pct)
        print("\nPopulation after elitism:")
        print(new_pop)
        best = new_pop[0]
        val_best = evaluate_state(best)

        if val_best > opt_value:
            opt_state = best
            opt_value = val_best

        selected = selection(val_pop, pop_size - len(new_pop))
        print("\nSelected individuals:")
        print(selected)
        crossed = crossover_step(selected, pop_size, cross_ratio, max_size)
        print("\nSelected individuals after crossover:")
        print(crossed)
        mutated = mutation_step(crossed, mut_ratio)
        print("\nSelected individuals after mutation:")
        print(mutated)
        
        pop = new_pop + mutated
        print("\nNew population:")
        print(pop)

        conv = convergent(pop)
        iter += 1
    
    return opt_state, opt_value, iter, conv