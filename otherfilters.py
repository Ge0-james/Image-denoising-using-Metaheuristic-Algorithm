from scipy.ndimage import uniform_filter
import numpy as np
import bm3d


# lee filter
def lee_filter(img, kernel_size):
    img_mean = uniform_filter(img, (kernel_size, kernel_size))
    img_sqr_mean = uniform_filter(img**2, (kernel_size, kernel_size))
    img_variance = img_sqr_mean - img_mean**2
    overall_variance = np.var(img)
    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output


def frost_filter(img, kernel_size, damping_factor=2.0):
    h, w = img.shape
    padded_img = np.pad(img, kernel_size // 2, mode="reflect")
    filtered_img = np.zeros_like(img, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            window = padded_img[i : i + kernel_size, j : j + kernel_size]
            center_pixel = window[kernel_size // 2, kernel_size // 2]
            distances = (window - center_pixel) ** 2
            weights = np.exp(-damping_factor * distances)
            filtered_img[i, j] = np.sum(window * weights) / np.sum(weights)
    return filtered_img


# def bm3d_filter(image_noisy, sigma_psd=0.1):
#     image_noisy_float = np.float32(image_noisy)
#     denoised_image = bm3d.bm3d(image_noisy_float, sigma_psd=sigma_psd)
#     denoised_image = np.uint8(denoised_image)
#     return denoised_image


def bm3d_filter(image_noisy):
    denoised_image = bm3d.bm3d(
        image_noisy, sigma_psd=0.1, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
    )
    return denoised_image
