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

def jellyfish_search_optimizer(original_image, noisy_image, num_jellyfish, num_iterations):
    jellyfish = [create_random_filter(3) for _ in range(num_jellyfish)]
    
    personal_best_positions = jellyfish.copy()
    personal_best_fitness = [fitness(original_image, noisy_image, jelly) for jelly in jellyfish]
    global_best_index = np.argmax(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_index]
    global_best_fitness = personal_best_fitness[global_best_index]

    # JS optimizer parameters
    ocean_current_factor = 0.2
    active_motion_factor = 0.8

    for _ in range(num_iterations):
        for i in range(num_jellyfish):
            # Move following ocean current
            ocean_current = (global_best_position - jellyfish[i]) * ocean_current_factor
            jellyfish[i] += ocean_current
            
            # Active or passive motion
            if np.random.rand() < 0.5:
                # Active motion
                random_jellyfish = jellyfish[np.random.randint(num_jellyfish)]
                active_motion = (random_jellyfish - jellyfish[i]) * active_motion_factor
                jellyfish[i] += active_motion
            else:
                # Passive motion (random movement)
                passive_motion = create_random_filter(3) - jellyfish[i]
                jellyfish[i] += passive_motion
            
            # Evaluate new position
            current_fitness = fitness(original_image, noisy_image, jellyfish[i])
            if current_fitness > personal_best_fitness[i]:
                personal_best_positions[i] = jellyfish[i]
                personal_best_fitness[i] = current_fitness
        
        # Update global best
        best_index = np.argmax(personal_best_fitness)
        if personal_best_fitness[best_index] > global_best_fitness:
            global_best_position = personal_best_positions[best_index]
            global_best_fitness = personal_best_fitness[best_index]

    return global_best_position
