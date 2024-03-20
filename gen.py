import numpy as np
from skimage import img_as_float
import cv2
import matplotlib.pyplot as plt
import random
import time
from skimage.metrics import structural_similarity as ssim_metric

filter_size = 3
population_size = 10
num_generations = 10
threshold_value = 34


def add_speckle_noise_test(image, sigma):
    np.random.seed(int(time.time()))

    # Generate a random Gaussian noise array with the same shape as the image.
    noise = np.random.normal(0, sigma, image.shape)

    # Multiply the noise array by the image to add speckle noise to the image.
    noisy_image = image + image * noise

    return noisy_image


def create_random_filter(size):
    matrix = np.random.rand(size, size)
    matrix /= matrix.sum()
    return matrix


def apply_filter(image, filter):
    return img_as_float(cv2.filter2D(image, -1, filter))


def apply_filter2(image, filter):
    return img_as_float(cv2.filter2D(image, -1, filter))


def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def fitness(original_image, noisy_image, filter):
    filtered_image = apply_filter(noisy_image, filter)
    psnr = calculate_psnr(original_image, filtered_image)
    return psnr


def calculate_ssim(img1, img2):
    return ssim_metric(img1, img2)


def genetic_algorithm(
    original_image, noisy_image, filter_size, population_size, num_generations
):
    population = [create_random_filter(filter_size) for _ in range(population_size)]
    for generation in range(num_generations):
        fitness_scores = [
            fitness(original_image, noisy_image, filter) for filter in population
        ]
        selected_indices = np.argsort(fitness_scores)[-1 : population_size // 2 : -1]

        parents = [population[i] for i in selected_indices]

        least_filter = parents[-1]

        min_fitness_value = fitness(original_image, noisy_image, least_filter)

        if fitness_scores[selected_indices[0]] >= 34:
            return parents[selected_indices[0]]

        offspring = parents
        while len(offspring) < population_size:
            p1 = random.choice(selected_indices)
            p2 = random.choice(selected_indices)
            parent1 = population[p1]
            parent2 = population[p2]
            child = crossover(parent1, parent2)
            child = mutate(child)
            fitness_child = fitness(original_image, noisy_image, child)
            if fitness_child >= min_fitness_value:
                offspring.append(child)
        population = offspring
    best_filter = population[np.argmax(fitness_scores)]
    return best_filter


def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, filter_size * filter_size)
    flattened_parent1 = parent1.reshape(-1)
    flattened_parent2 = parent2.reshape(-1)
    child1 = np.concatenate(
        (flattened_parent1[:crossover_point], flattened_parent2[crossover_point:])
    )
    child1 = child1.reshape(filter_size, filter_size)
    return child1


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
