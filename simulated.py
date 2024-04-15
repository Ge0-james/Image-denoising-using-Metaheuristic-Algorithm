import numpy as np
import random
import math


def create_random_filter(filter_size):
    # Assuming this function creates a random filter of the given size
    return np.random.rand(filter_size, filter_size)


def fitness(original_image, noisy_image, filter):
    # Assuming this function calculates the fitness score of a filter
    # This is a placeholder for your actual fitness function
    return np.sum(np.abs(original_image - noisy_image))


def mutate(filter):
    flattened_filter = filter.flatten()
    l = len(flattened_filter)
    k = l // 2
    limit1 = random.randint(0, k)
    limit2 = random.randint(k, l)
    shuffled_elements = flattened_filter[limit1:limit2]
    np.random.shuffle(shuffled_elements)
    j = 0
    for i in range(limit1, limit2):
        flattened_filter[i] = shuffled_elements[j]
        j = j + 1
    mutated_filter = flattened_filter.reshape(filter.shape)
    return mutated_filter


def simulated_annealing(
    original_image,
    noisy_image,
    initial_filter,
    num_iterations,
    initial_temperature,
    cooling_rate,
):
    current_filter = initial_filter
    current_fitness = fitness(original_image, noisy_image, current_filter)
    best_filter = current_filter
    best_fitness = current_fitness

    for i in range(num_iterations):
        temperature = initial_temperature / (1 + cooling_rate * i)

        neighbor_filter = mutate(current_filter)
        neighbor_fitness = fitness(original_image, noisy_image, neighbor_filter)

        if neighbor_fitness > current_fitness or random.uniform(0, 1) < math.exp(
            (neighbor_fitness - current_fitness) / temperature
        ):
            current_filter = neighbor_filter
            current_fitness = neighbor_fitness

        if current_fitness > best_fitness:
            best_filter = current_filter
            best_fitness = current_fitness

    return best_filter
