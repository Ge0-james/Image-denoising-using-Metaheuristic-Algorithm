from gen import *
import matplotlib.pyplot as plt
import os
import otherfilters as o

path = "img"

files = os.listdir(path)
sig = 0.1  # sigma
f = open("psnr.txt", "a")
for file in files:
    w = []
    f.write("\n\n")
    original_image = img_as_float(cv2.imread(path + "/" + file, cv2.IMREAD_GRAYSCALE))
    noisy_image = add_speckle_noise_test(original_image, sig)
    best_filter = genetic_algorithm(
        original_image, noisy_image, filter_size, population_size, num_generations
    )
    print(best_filter)
    filtered_image = apply_filter(noisy_image, best_filter)
    psnr_filtered = calculate_psnr(original_image, filtered_image)
    psnr_noisy = calculate_psnr(original_image, noisy_image)
    print("PSNR of filtered image:", psnr_filtered)
    print("PSNR of noisy imamge:", psnr_noisy)
    filtered_image_uint8 = np.clip(filtered_image * 255, 0, 255).astype(np.uint8)
    noisy_image_uint8 = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite("output/filtered" + file, filtered_image_uint8)
    cv2.imwrite("output/noisy" + file, noisy_image_uint8)
    w.append(file + "\n")
    w.append("gen : " + str(psnr_filtered) + "\n")
    w.append("filter : " + str(best_filter) + "\n")

    # applying lee filter
    leeimage = o.lee_filter(noisy_image, filter_size)
    o_psnr = calculate_psnr(original_image, leeimage)
    w.append("lee : " + str(o_psnr) + "\n")

    # applying frost filter
    frostimage = o.frost_filter(noisy_image, filter_size, 2.0)
    o_psnr = calculate_psnr(original_image, frostimage)
    w.append("frost : " + str(o_psnr) + "\n")

    # bm3d
    bm3dimage = o.bm3d_filter(noisy_image, sigma_psd=0.1)
    o_psnr = calculate_psnr(original_image, bm3dimage)
    w.append("bm3d : " + str(o_psnr) + "\n")

    f.writelines(w)
f.close()
