from PIL import ImageFilter
from PIL.Image import Image, Resampling
import numpy as np
from sewar.full_ref import msssim, ssim, mse
from skimage.feature import match_template


def mae(a, b):
    return np.mean(np.abs(a - b))


def normalize_size(src_img, res_img, min_dim=float('inf')):
    target_size = (min(src_img.width, res_img.width),
                   min(src_img.height, res_img.height))

    if min(target_size) < min_dim:
        scale = min_dim / min(target_size)
        target_size = (int(target_size[0] * scale),
                       int(target_size[1] * scale))

    if src_img.size != target_size:
        src_img = src_img.resize(target_size, Resampling.BICUBIC)

    if res_img.size != target_size:
        res_img = res_img.resize(target_size, Resampling.BICUBIC)

    return src_img, res_img


def blur_ssim(src_img, res_img, radii=(1, 3, 5)):
    src_img, res_img = normalize_size(src_img, res_img, min_dim=176)
    src_img = src_img.convert('L')
    res_img = res_img.convert('L')
    
    total_weighted_ssim = 0.0
    total_weight = 0.0
    history = {}

    for r in radii:
        blurred_src = src_img.filter(ImageFilter.GaussianBlur(radius=r))
        blurred_res = res_img.filter(ImageFilter.GaussianBlur(radius=r))

        src_arr = np.array(blurred_src)
        res_arr = np.array(blurred_res)

        current_ssim, _ = ssim(src_arr, res_arr)

        weight = 1.0 / r
        total_weighted_ssim += (current_ssim * weight)
        total_weight += weight

        history[r] = {
            'ssim': current_ssim,
            'ssim_to_blur_ratio': current_ssim / r,
            'weight_applied': weight
        }

    final_score = total_weighted_ssim / total_weight
    
    return final_score, history


def img_similarity(src_img: Image, res_img: Image, blur: bool = False) -> float:
    src_img, res_img = normalize_size(src_img, res_img, min_dim=176)

    if blur:
        src_img = src_img.filter(ImageFilter.GaussianBlur(radius=2))
        res_img = res_img.filter(ImageFilter.GaussianBlur(radius=2))

    src_arr = np.array(src_img.convert("L"))
    res_arr = np.array(res_img.convert("L"))
    return np.real(msssim(src_arr, res_arr))


def line_similarity(src_img: Image, res_img: Image) -> float:
    src_img, res_img = normalize_size(src_img, res_img, min_dim=176)
    src_arr = np.array(src_img.convert("L"))
    res_arr = np.array(res_img.convert("L"))
    return (0.3 * ssim(src_arr, res_arr)[0] +
            0.6 * np.mean(match_template(src_arr, res_arr)) +
            0.1 * (1 - mae(src_arr / 255, res_arr / 255)))
