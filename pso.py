import numpy as np
from skimage import img_as_float
import cv2


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


def particle_swarm_optimization(
    original_image, noisy_image, num_particles, num_iterations
):
    particles = [create_random_filter(3) for _ in range(num_particles)]

    personal_best_positions = particles.copy()
    personal_best_fitness = [
        fitness(original_image, noisy_image, particle) for particle in particles
    ]
    global_best_index = np.argmax(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_index]
    global_best_fitness = personal_best_fitness[global_best_index]

    # PSO main loop
    for _ in range(num_iterations):
        fitness_values = [
            fitness(original_image, noisy_image, particle) for particle in particles
        ]
        for i in range(num_particles):
            if fitness_values[i] > personal_best_fitness[i]:
                personal_best_positions[i] = particles[i]
                personal_best_fitness[i] = fitness_values[i]
        best_index = np.argmax(personal_best_fitness)
        if personal_best_fitness[best_index] > global_best_fitness:
            global_best_position = personal_best_positions[best_index]
            global_best_fitness = personal_best_fitness[best_index]

    return global_best_position
