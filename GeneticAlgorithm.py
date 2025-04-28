import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Class and function definitions for Node and Chromosome
class Node:
    def __init__(self, id, x, y):
        self.x = float(x)
        self.y = float(y)
        self.id = int(id)

file_name = "node_coordinates.txt"
dataset = []

with open(file_name, "r") as f:
    for line in f:
        new_line = line.strip()
        new_line = new_line.split(" ")
        id, x, y = new_line[0], new_line[1], new_line[2]
        dataset.append(Node(id=id, x=x, y=y))

N = 20

def create_distance_matrix(node_list):
    matrix = [[0 for _ in range(N)] for _ in range(N)]

    for i in range(0, len(matrix)-1):
        for j in range(0, len(matrix[0])-1):
            matrix[node_list[i].id][node_list[j].id] = math.sqrt(
                pow((node_list[i].x - node_list[j].x), 2) + pow((node_list[i].y - node_list[j].y), 2)
            )
    return matrix

matrix = create_distance_matrix(dataset)

class Chromosome:
    def __init__(self, node_list):
        self.chromosome = node_list

        chr_representation = []
        for i in range(0, len(node_list)):
            chr_representation.append(self.chromosome[i].id)
        self.chr_representation = chr_representation

        distance = 0
        for j in range(1, len(self.chr_representation) - 1):
            distance += matrix[self.chr_representation[j]-1][self.chr_representation[j + 1]-1]
        self.cost = distance

        self.fitness_value = 1 / self.cost


# Genetic Algorithm Functions
def create_random_list(node_list):
    start_node = node_list[0]

    remaining_nodes = node_list[1:]
    remaining_nodes = random.sample(remaining_nodes, len(remaining_nodes))

    remaining_nodes.insert(0, start_node)
    remaining_nodes.append(start_node)
    return remaining_nodes

def initialization(nodes, population_size):
    population = []
    for i in range(0, population_size):
        chromosome_nodes = create_random_list(nodes)
        new_chromosome = Chromosome(chromosome_nodes)
        population.append(new_chromosome)
    return population

def selection(population):
    index_1, index_2, index_3, index_4 = random.sample(range(0, 99), 4)

    chromosome_1 = population[index_1]
    chromosome_2 = population[index_2]
    chromosome_3 = population[index_3]
    chromosome_4 = population[index_4]

    if chromosome_1.fitness_value > chromosome_2.fitness_value:
        selected = chromosome_1
    else:
        selected = chromosome_2

    if chromosome_3.fitness_value > selected.fitness_value:
        selected = chromosome_3

    if chromosome_4.fitness_value > selected.fitness_value:
        selected = chromosome_4

    return selected

def crossover(parent_1, parent_2):
    crossover_point = random.randint(2, 14)

    offspring_1 = parent_1.chromosome[1:crossover_point]
    offspring_2 = parent_2.chromosome[1:crossover_point]

    offspring_1_remaining = [item for item in parent_2.chromosome[1:-1] if item not in offspring_1]
    offspring_2_remaining = [item for item in parent_1.chromosome[1:-1] if item not in offspring_2]

    offspring_1 += offspring_1_remaining
    offspring_2 += offspring_2_remaining

    offspring_1.insert(0, parent_1.chromosome[0])
    offspring_1.append(parent_1.chromosome[0])

    offspring_2.insert(0, parent_2.chromosome[0])
    offspring_2.append(parent_2.chromosome[0])

    return offspring_1, offspring_2

def crossover_two(parent_1, parent_2):
    point_1, point_2 = random.sample(range(1, len(parent_1.chromosome)-1), 2)
    start_point = min(point_1, point_2)
    end_point = max(point_1, point_2)

    offspring_1 = parent_1.chromosome[start_point:end_point+1]
    offspring_2 = parent_2.chromosome[start_point:end_point+1]

    offspring_1_remaining = [item for item in parent_2.chromosome[1:-1] if item not in offspring_1]
    offspring_2_remaining = [item for item in parent_1.chromosome[1:-1] if item not in offspring_2]

    offspring_1 += offspring_1_remaining
    offspring_2 += offspring_2_remaining

    offspring_1.insert(0, parent_1.chromosome[0])
    offspring_1.append(parent_1.chromosome[0])

    offspring_2.insert(0, parent_2.chromosome[0])
    offspring_2.append(parent_2.chromosome[0])

    return offspring_1, offspring_2

def crossover_mix(parent_1, parent_2):
    point_1, point_2 = random.sample(range(1, len(parent_1.chromosome)-1), 2)
    start_point = min(point_1, point_2)
    end_point = max(point_1, point_2)

    offspring_1_start = parent_1.chromosome[:start_point]
    offspring_1_end = parent_1.chromosome[end_point:]
    offspring_1 = offspring_1_start + offspring_1_end
    offspring_2 = parent_2.chromosome[start_point:end_point+1]

    offspring_1_remaining = [item for item in parent_2.chromosome[1:-1] if item not in offspring_1]
    offspring_2_remaining = [item for item in parent_1.chromosome[1:-1] if item not in offspring_2]

    offspring_1 = offspring_1_start + offspring_1_remaining + offspring_1_end
    offspring_2 += offspring_2_remaining

    offspring_2.insert(0, parent_2.chromosome[0])
    offspring_2.append(parent_2.chromosome[0])

    return offspring_1, offspring_2

def mutation(chromosome):
    index_1, index_2 = random.sample(range(1, 19), 2)
    chromosome[index_1], chromosome[index_2] = chromosome[index_2], chromosome[index_1]
    return chromosome

def find_best(generation):
    best_chromosome = generation[0]
    for n in range(1, len(generation)):
        if generation[n].cost < best_chromosome.cost:
            best_chromosome = generation[n]
    return best_chromosome

def create_new_generation(previous_generation, mutation_rate):
    new_generation = [find_best(previous_generation)]

    for a in range(0, int(len(previous_generation)/2)):
        parent_1 = selection(previous_generation)
        parent_2 = selection(previous_generation)

        offspring_1, offspring_2 = crossover_mix(parent_1, parent_2)
        offspring_1 = Chromosome(offspring_1)
        offspring_2 = Chromosome(offspring_2)

        if random.random() < mutation_rate:
            mutated_chromosome = mutation(offspring_1.chromosome)
            offspring_1 = Chromosome(mutated_chromosome)

        new_generation.append(offspring_1)
        new_generation.append(offspring_2)

    return new_generation


# Main Execution
max_generations = 200
population_count = 100
mutation_probability = 0.2
node_data = dataset

def genetic_algorithm(generation_limit, population_size, mutation_rate, nodes):
    current_generation = initialization(nodes, population_size)
    cost_history = []

    for generation in range(0, generation_limit):
        current_generation = create_new_generation(current_generation, mutation_rate)
        print(str(generation) + ". generation --> " + "cost --> " + str(current_generation[0].cost))
        cost_history.append(find_best(current_generation).cost)

    return current_generation, cost_history

def draw_cost_generation(cost_values):
    generation_numbers = np.arange(1, len(cost_values)+1)

    plt.plot(generation_numbers, cost_values)

    plt.title("Route Cost through Generations")
    plt.xlabel("Generations")
    plt.ylabel("Cost")

    plt.show()

def draw_path(chromosome):
    x_coordinates = []
    y_coordinates = []

    for node in range(0, len(chromosome.chromosome)):
        x_coordinates.append(chromosome.chromosome[node].x)
        y_coordinates.append(chromosome.chromosome[node].y)

    fig, ax = plt.subplots()
    plt.scatter(x_coordinates, y_coordinates)

    ax.plot(x_coordinates, y_coordinates, '--', lw=2, color='red', ms=10)
    ax.set_xlim(0, 1650)
    ax.set_ylim(0, 1300)

    plt.show()

final_generation, cost_plot_data = genetic_algorithm(
    max_generations, population_count, mutation_probability, node_data
)

draw_cost_generation(cost_plot_data)
draw_path(find_best(final_generation))
