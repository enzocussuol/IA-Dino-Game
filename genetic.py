import time
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

def evaluate_population(pop):
    results = []
    final_results = []
    for i in range(0, 3):
        players_and_states = []
        for s in pop:
            ai_player = nn.NN(cfg.num_imputs, cfg.num_outputs)
            nn.vector_to_parameters(nn.Tensor(s), ai_player.parameters())
            players_and_states.append((ai_player, s))
        results.append(playGame(players_and_states))

    for i in range(0, len(results[0])):
        state = results[0][i][1]
        points = []
        for j in range(0, 3):
            points.append(results[j][i][0])
        points = np.asarray(points)
        final_points = int(points.mean())
        final_results.append((final_points, state))

    return final_results

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

def elitism(val_pop, pop_size, pct):
    n = math.floor((pct/100)*pop_size)
    if n < 1:
        n = 1
    val_elite = val_pop[(pop_size - n):]

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

def crossover_step(pop, cross_ratio):
    new_pop = []

    for _ in range(round(len(pop)/2)):
        rand = random.uniform(0, 1)

        fst_ind = random.randint(0, len(pop) - 1)
        scd_ind = random.randint(0, len(pop) - 1)

        parent1 = pop[fst_ind]
        parent2 = pop[scd_ind]

        if rand <= cross_ratio:
            offspring1, offspring2 = crossover(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2

        new_pop = new_pop + [offspring1, offspring2]

    return new_pop

def mutation(s):
    state = s.copy()

    for i in range(0, int(len(state)*cfg.gene_mut_ratio)):
        rand = random.randint(0, len(state) - 1)
        sum_or_subtract = random.randint(0, 1)

        if sum_or_subtract == 0:
            new_value = state[rand] + random.uniform(cfg.min_valor_peso, cfg.max_valor_peso)
        else:
            new_value = state[rand] - random.uniform(cfg.min_valor_peso, cfg.max_valor_peso)

        if new_value > 1:
            new_value = -1 + (new_value - 1)
        elif new_value < -1:
            new_value = 1 + (new_value + 1)
        
        state[rand] = new_value
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

def generate_initial_pop():
    pop = []
    ai_player = nn.NN(cfg.num_imputs, cfg.num_outputs)

    for i in range(0, cfg.num_individuals_per_pop):
        individual = []
        for j in range(0, len(nn.parameters_to_vector(ai_player.parameters()))):
            individual.append(random.uniform(cfg.min_valor_peso, cfg.max_valor_peso))
        pop.append(individual)
        
    return pop

def genetic(max_iter, max_time, cross_ratio, mut_ratio, elite_pct, pop):
    pop_size = len(pop)

    opt_state = [0] * pop_size
    opt_value = 0
    conv = convergent(pop)
    iter = 0

    start = time.time()
    end = 0

    while not conv and iter < max_iter and end - start <= cfg.max_time:
        val_pop = evaluate_population(pop)
        val_pop = sorted(val_pop, key=first)

        new_pop = elitism(val_pop, pop_size, elite_pct)

        best = new_pop[len(new_pop) - 1]

        vals = []
        for v, s in val_pop:
            vals.append(v)
        val_best = max(vals)

        if val_best > opt_value:
            opt_state = best
            opt_value = val_best

        selected = selection(val_pop, pop_size - len(new_pop))
        crossed = crossover_step(selected, cross_ratio)
        mutated = mutation_step(crossed, mut_ratio)
        
        pop = new_pop + mutated

        conv = convergent(pop)
        iter += 1

        end = time.time()

        print("============================")
        print("ITER: ", iter)
        print("BEST VALUE FROM ITER: ", val_best)
        print("BEST VALUE SO FAR: ", opt_value)
    
    return opt_state, opt_value