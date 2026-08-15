from PIL.Image import Image, Resampling
import numpy as np
from sewar.full_ref import msssim, ssim, mse
from skimage.feature import match_template


def mae(a, b):
    return np.mean(np.abs(a - b))


def img_similarity(src_img: Image, res_img: Image):
    target_size = (min(src_img.width, res_img.width),
                   min(src_img.height, res_img.height))
    min_dim = min(target_size)
    if min_dim < 176:
        scale = 176 / min_dim
        target_size = (int(target_size[0] * scale),
                       int(target_size[1] * scale))

    if src_img.size != target_size:
        src_img = src_img.resize(target_size, Resampling.BICUBIC)

    if res_img.size != target_size:
        res_img = res_img.resize(target_size, Resampling.BICUBIC)

    src_img = np.array(src_img.convert("L"))
    res_img = np.array(res_img.convert("L"))
    return np.real(msssim(src_img, res_img))


def line_similarity(src_img: Image, res_img: Image) -> float:
    target_size = (
        min(src_img.width, res_img.width),
        min(src_img.height, res_img.height),
    )

    if src_img.size != target_size:
        src_img = src_img.resize(target_size, Resampling.BICUBIC)

    if res_img.size != target_size:
        res_img = res_img.resize(target_size, Resampling.BICUBIC)

    src_arr = np.array(src_img.convert("L"))
    res_arr = np.array(res_img.convert("L"))
    return (0.3 * ssim(src_arr, res_arr)[0] +
            0.6 * np.mean(match_template(src_arr, res_arr)) +
            0.1 * (1 - mae(src_arr / 255, res_arr / 255)))
