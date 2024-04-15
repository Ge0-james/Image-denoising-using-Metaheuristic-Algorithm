import numpy as np
import cv2
from skimage import img_as_float
import random
import time

filter_size = 3
population_size = 100
num_generations = 100
threshold_value = 33


def add_speckle_noise_test(image, sigma):
    np.random.seed(int(time.time()))
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + image * noise
    return noisy_image


def create_random_filter(size):
    matrix = np.random.rand(size, size)
    matrix /= matrix.sum()
    return matrix


def apply_filter(image, filter):
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


def differential_evolution(
    original_image, noisy_image, filter_size, population_size, num_generations
):
    population = [create_random_filter(filter_size) for _ in range(population_size)]
    for generation in range(num_generations):
        print(generation)
        fitness_scores = [
            fitness(original_image, noisy_image, filter) for filter in population
        ]
        if fitness_scores[0] >= threshold_value:
            print("found")
            return population[0]
        for i in range(population_size):
            # Generate trial vectors
            trial1 = population[i] + np.random.rand(filter_size, filter_size) * (
                population[i] - population[np.random.randint(0, population_size)]
            )
            trial2 = population[i] + np.random.rand(filter_size, filter_size) * (
                population[i] - population[np.random.randint(0, population_size)]
            )
            # Evaluate fitness of trial vectors
            fitness_trial1 = fitness(original_image, noisy_image, trial1)
            fitness_trial2 = fitness(original_image, noisy_image, trial2)
            # Select the best trial vector
            if fitness_trial1 > fitness_scores[i]:
                population[i] = trial1
                fitness_scores[i] = fitness_trial1
            if fitness_trial2 > fitness_scores[i]:
                population[i] = trial2
                fitness_scores[i] = fitness_trial2
    best_filter = population[np.argmax(fitness_scores)]
    return best_filter
