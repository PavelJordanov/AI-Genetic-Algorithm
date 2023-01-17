import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import math

state_indexes = {}
state_counter = 0
state_neighbors = [None] * 100

def Get_State_Index(state):
    global state_counter
    if state in state_indexes:
        return state_indexes[state]
    else:
        state_indexes[state] = state_counter
        state_counter += 1
        return state_counter - 1

def Get_State_Indexes(states):
    indexes = []
    for state in states:
        indexes.append(Get_State_Index(state))
    return indexes

# Prepare the data from the adjacent neighbors list
def Prep_Data():
    with open("state_neighbors.txt", encoding = "utf8") as f:
        for line in f:
            states_line = line.strip().split(" ")
            states_len = len(states_line)
            
            state_index = Get_State_Index(states_line[0])
            state_neighbors[state_index] = Get_State_Indexes(states_line[1:]) 

# Generate the Initial Population
def Generate_Initial_Population(population_size, state_counter):
    population = []
    
    # Producing an array of chromosomes
    for i in range (0, population_size):
        chromosome = []
        for y in range (0, state_counter):
            random_color = random.randint(1, 4)
            chromosome.append(random_color)
        population.append(chromosome)
            
    return population
    
# Fitness function to help determine which members of the population will offer a better 
# chance of finding the solution
def Fitness(present_population, state_counter):
    fitness = []
    
    population_max_fitness_score = 0
    population_min_fitness_score = 0
    population_sum_fitness_score = 0
    population_avg_fitness_score = 0
    
    population_size = len(present_population)
    for a in range(0, population_size):
        chromosome = present_population[a]
        fitness_score = 0
        non_violating_edges = 0
        total_edges = 0
        for i in range(0, state_counter):
            selected_state_neighbors = state_neighbors[i]
            for selected_state_neighbor in selected_state_neighbors:
                if (chromosome[i] != chromosome[selected_state_neighbor]):
                    non_violating_edges += 1
                total_edges += 1    
        fitness_score = non_violating_edges / total_edges
        if a == 0:
            population_max_fitness_score = fitness_score
            population_min_fitness_score = fitness_score
        else:
            if population_max_fitness_score < fitness_score:
                population_max_fitness_score = fitness_score
            if population_min_fitness_score > fitness_score:
                population_min_fitness_score = fitness_score
        population_sum_fitness_score += fitness_score
        fitness.append(fitness_score) 
        population_avg_fitness_score = population_sum_fitness_score / population_size
    
    return fitness, population_max_fitness_score, population_min_fitness_score, population_avg_fitness_score
    
# Here we choose the parents to use for the next generation
def Choose_Parents(tournament_size, present_population, fitness):  
    parents = []
    k = tournament_size
    #random.shuffle(present_population)
    counter = 0
    
    tournament_max_fitness_score = 0
    tournament_max_fitness_position = 0
    
    for i in range(0, len(present_population)):
        counter += 1
            
        if (tournament_max_fitness_score < fitness[i]):
            tournament_max_fitness_score = fitness[i]
            tournament_max_fitness_position = i
        
        if (counter == k):
            parents.append(present_population[tournament_max_fitness_position])
            tournament_max_fitness_score    = 0
            tournament_max_fitness_position = 0
            counter = 0
        
    return parents

# Here I combine the two parents in different ways to produce a child
def Crossover(parent1, parent2):
    number_of_nodes = len(parent1)
    border_index = random.randint(0, number_of_nodes)
    child = []
    
    for i in range(0, number_of_nodes):
        if i < border_index:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
            
    return child

# This function is used to produce the next generation     
def Next_Generation(parents, present_population):
    number_of_parents = len(parents)
    number_of_nodes = len(parents[0])
    next_population = []
    
    while len(next_population) < len(present_population):
        first_parent_index = random.randint(0, number_of_parents-1)
        second_parent_index = random.randint(0, number_of_parents-1)
        if (first_parent_index == second_parent_index):
            continue
            
        next_population.append(Crossover(parents[first_parent_index], parents[second_parent_index]))
    
    return next_population
    
# This function is used to alter some genes in the new population
def Mutation(mutation_rate, next_population):
    n = len(next_population[0])
    mutated_genes = len(next_population) * n * mutation_rate
    
    for i in range(0, math.floor(mutated_genes)):
        chromosome_index = random.randint(0, len(next_population)-1)
        gene_index = random.randint(0, n-1)
        
        gene_value = random.randint(1, 4)
            
        next_population[chromosome_index][gene_index] = gene_value

# Function to test if we have found a solution        
def reached_end_goal(fitness):
    for fitness_score in fitness:
        if fitness_score == 1.0:
            return True
    return False

def main():
    global state_counter
    Prep_Data()
    number_generations = 50
    population_size = 10
    tournament_size = 2
    mutation_rate = 0.01

    new_chromosome = []
    
    # Alter the values in the for loops to test out different values for the variables
    # 10 was removed for population because a tournament size of 10 does not work with 
    # a population of size 10 as it will only produce one generation. 
    for number_of_generations in [50, 500, 5000]: 
        for population_size in [100, 1000]:
            for mutation_rate in [0.01, 0.02, 0.05, 0.1]:
                for tournament_size in [2, 5, 10]:
                    population_min_fitness_scores = []
                    population_max_fitness_scores = []
                    population_avg_fitness_scores = []
                    
                    present_population = Generate_Initial_Population(population_size, state_counter)
                    
                    
                    current_generation = 1
                    while current_generation <= number_of_generations:
                        fitness, population_max_fitness_score, population_min_fitness_score, population_avg_fitness_score = Fitness(present_population, state_counter)
                        
                        print(f'\nGeneration #{current_generation}:')
                        print(f'Best fitness: {population_max_fitness_score}')
                        print(f'Worst fitness: {population_min_fitness_score}')
                        print(f'Average fitness: {population_avg_fitness_score}')
                        
                        population_min_fitness_scores.append(population_min_fitness_score)
                        population_max_fitness_scores.append(population_max_fitness_score)
                        population_avg_fitness_scores.append(population_avg_fitness_score)
                        
                        
                        if reached_end_goal(fitness):
                            for i in range (0, len(present_population)):
                                if fitness[i] == 1:
                                    answer = i
                            print("Reached goal ...")
                            counter = 0
                            print("Answer:")
                            for keys in state_indexes.keys():
                                print(f"{keys}: {present_population[i][counter]}")
                                counter += 1
                            current_generation += 1
                            break
                    
                        parents = Choose_Parents(tournament_size, present_population, fitness)
                        
                        next_population = Next_Generation(parents, present_population)
                        
                        Mutation(mutation_rate, next_population)
                        
                        present_population = next_population
                        current_generation += 1
                    
                    generations = range(1, current_generation)
                    x=np.array(population_max_fitness_scores)
                    y=np.array(population_avg_fitness_scores)
                    z=np.array(population_min_fitness_scores)
                    t=np.array(generations)

                    title = f'Fitness over Generation ({number_of_generations}, {population_size}, {mutation_rate}, {tournament_size})' 

                    plt.plot(t, x, label = "Best", color='r')
                    plt.plot(t, z, label = "Worst", color='b')
                    plt.plot(t, y, label = "Average", color='g')
                    plt.xlabel('Generation') 
                    plt.ylabel('Fitness')
                    plt.legend()
                    plt.title(title)
                    plt.show()
                    
if __name__ == "__main__":
    main()