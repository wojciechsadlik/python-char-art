from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from enum import Enum


class DITHER_MODES(Enum):
    NONE = 'NONE'
    BAYER = 'BAYER'
    JJN = 'JJN'
    FS = 'FS'
    ATKINSON = 'ATKINSON'


def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0]])

    n_sq = n ** 2
    m = generate_bayer_matrix(n / 2)
    r0 = np.concatenate((n_sq * m, n_sq * m + 2), axis=1)
    r1 = np.concatenate((n_sq * m + 3, n_sq * m + 1), axis=1)
    m = np.concatenate((r0, r1), axis=0)
    return (1 / n_sq) * m


dither_bayer_m = generate_bayer_matrix(32) - 0.5

jjn_k = np.array([
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
]) / 48

fs_k = np.array([
    [0, 0, 7],
    [3, 5, 1]
]) / 16

atkinson_k = np.array([
    [0, 0, 1, 1],
    [1, 1, 1, 0],
    [0, 1, 0, 0]
]) / 8


def apply_threshold_map(c, m, r, x, y):
    n = len(m)
    return c + r * m[x % n][y % n]


def apply_error_diff(c_new, img_arr, x, y, kernel, kernel_off_x):
    c = img_arr[y][x]
    c_err = c - c_new
    c_err_kernel = c_err * kernel
    x_left = x - kernel_off_x
    if x_left < 0:
        c_err_kernel = c_err_kernel[:, -x_left:]
        x_left = 0
    x_right = x_left + c_err_kernel.shape[1]
    if x_right > img_arr.shape[1]:
        c_err_kernel = c_err_kernel[:,
                                    :c_err_kernel.shape[1] - (x_right - img_arr.shape[1])]
        x_right = img_arr.shape[1]
    top = y
    bottom = y + c_err_kernel.shape[0]
    if bottom > img_arr.shape[0]:
        c_err_kernel = c_err_kernel[:c_err_kernel.shape[0] -
                                    (bottom - img_arr.shape[0]), :]
        bottom = img_arr.shape[0]
    img_arr[top:bottom, x_left:x_right] += c_err_kernel


def apply_error_diff_window(
        c_new,
        c_width,
        c_height,
        img_arr,
        x,
        y,
        kernel,
        kernel_off_x):
    c = img_arr[y:y + c_height, x:x + c_width]
    c_err = c - c_new
    c_err = np.tile(
        c_err,
        (kernel.shape[0] //
         c_height,
         kernel.shape[1] //
         c_width))
    c_err_kernel = c_err * kernel
    x_left = x - kernel_off_x
    if x_left < 0:
        c_err_kernel = c_err_kernel[:, -x_left:]
        x_left = 0
    x_right = x_left + c_err_kernel.shape[1]
    if x_right > img_arr.shape[1]:
        c_err_kernel = c_err_kernel[:,
                                    :c_err_kernel.shape[1] - (x_right - img_arr.shape[1])]
        x_right = img_arr.shape[1]
    top = y
    bottom = y + c_err_kernel.shape[0]
    if bottom > img_arr.shape[0]:
        c_err_kernel = c_err_kernel[:c_err_kernel.shape[0] -
                                    (bottom - img_arr.shape[0]), :]
        bottom = img_arr.shape[0]
    img_arr[top:bottom, x_left:x_right] += c_err_kernel
    img_arr[top:bottom, x_left:x_right] = np.clip(img_arr[top:bottom, x_left:x_right], 0, 1)


def quantize_grayscale(img: Image.Image, img_colors: int,
                       dither=DITHER_MODES.NONE, return_palette_map=False,
                       palette: np.ndarray = None) -> Image.Image | np.ndarray:
    if (img.mode != "L"):
        raise Exception("img mode should be \"L\"")
    if (img_colors <= 0):
        raise Exception("img_colors should be > 0")
    if (palette is not None and img_colors != len(palette)):
        raise Exception("img_colors and length of palette should match")

    if (palette is None):
        palette = np.linspace(0, 1, img_colors)

    img_arr = np.array(img) / 255
    color_step = 1 / img_colors
    palette_map = np.zeros(img_arr.shape, dtype=np.int32)

    for y in range(0, img_arr.shape[0]):
        for x in range(0, img_arr.shape[1]):
            c = img_arr[y][x]
            c_new = c
            if (dither == DITHER_MODES.BAYER):
                c_new = apply_threshold_map(
                    c, dither_bayer_m, color_step, x, y)

            c_new_idx = int(c_new / color_step)
            c_new_idx = min(len(palette) - 1, max(0, c_new_idx))
            palette_map[y][x] = c_new_idx
            c_new = palette[c_new_idx]

            if (dither == DITHER_MODES.JJN):
                apply_error_diff(c_new, img_arr, x, y, jjn_k, 2)

            if (dither == DITHER_MODES.FS):
                apply_error_diff(c_new, img_arr, x, y, fs_k, 1)

            if (dither == DITHER_MODES.ATKINSON):
                apply_error_diff(c_new, img_arr, x, y, atkinson_k, 1)

            img_arr[y][x] = c_new

    if (return_palette_map):
        return palette_map

    img_arr = np.array(img_arr * 255, dtype=np.ubyte)
    return Image.fromarray(img_arr, "L")


def quantize_rgb(img: Image.Image, img_colors_per_channel: int,
                 dither=DITHER_MODES.NONE) -> Image.Image:
    img_r = img.getchannel("R")
    img_g = img.getchannel("G")
    img_b = img.getchannel("B")
    img_r = quantize_grayscale(img_r, img_colors_per_channel, dither)
    img_g = quantize_grayscale(img_g, img_colors_per_channel, dither)
    img_b = quantize_grayscale(img_b, img_colors_per_channel, dither)
    return Image.merge("RGB", (img_r, img_g, img_b))


def img_rgb_to_max_grayscale(img):
    img = np.array(img, dtype=np.int32).max(2)
    img = np.array(img, dtype=np.ubyte)
    return Image.fromarray(img, "L")


def preprocess_img(img: Image.Image,
                   scale_factor=1,
                   contrast=1,
                   brightness=1,
                   eq=0,
                   quantize_colors=256,
                   dither=DITHER_MODES.NONE,
                   sharpness=1,
                   enhance_edges=0,
                   grayscale=True):
    img = ImageOps.scale(img, scale_factor, Image.Resampling.BICUBIC)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = Image.blend(img, ImageOps.equalize(img), eq)
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = Image.blend(
        img,
        img.filter(
            ImageFilter.EDGE_ENHANCE_MORE),
        enhance_edges)
    if grayscale:
        img = quantize_grayscale(img.convert("L"), quantize_colors, dither)
    else:
        img = quantize_rgb(img, quantize_colors, dither)
    return img
