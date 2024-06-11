from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from enum import Enum

class DITHER_MODES(Enum):
    NONE = 1
    BAYER = 2
    JJN = 3
    FS = 4

def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0]])
    
    n_sq = n ** 2
    m = generate_bayer_matrix(n/2)
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

def apply_threshold_map(c, m, r, x, y):
    n = len(m)
    return c + r * m[x % n][y % n]

def estimate_jjn_error_diff(c, c_new, img_arr, x, y):
    c_err = c - c_new
    cumulative_error = 0
    for k_y in range(0, jjn_k.shape[0]):
        for k_x in range(0, jjn_k.shape[1]):
            if (y + k_y >= img_arr.shape[0]):
                continue
            if (x + k_x - 2 >= img_arr.shape[1] or x + k_x - 2 < 0):
                continue
            cumulative_error += c_err * jjn_k[k_y][k_x]
    return cumulative_error

def apply_jjn_error_diff(c, c_new, img_arr, x, y):
    c_err = c - c_new
    for k_y in range(0, jjn_k.shape[0]):
        for k_x in range(0, jjn_k.shape[1]):
            if (y + k_y >= img_arr.shape[0]):
                continue
            if (x + k_x - 2 >= img_arr.shape[1] or x + k_x - 2 < 0):
                continue
            img_arr[y + k_y][x + k_x - 2] += c_err * jjn_k[k_y][k_x]

def apply_fs_error_diff(c, c_new, img_arr, x, y):
    c_err = c - c_new
    for k_y in range(0, fs_k.shape[0]):
        for k_x in range(0, fs_k.shape[1]):
            if (y + k_y >= img_arr.shape[0]):
                continue
            if (x + k_x - 1 >= img_arr.shape[1] or x + k_x - 1 < 0):
                continue
            img_arr[y + k_y][x + k_x - 1] += c_err * fs_k[k_y][k_x]

def quantize_grayscale(img: Image.Image, img_colors: int,
                       dither=DITHER_MODES.NONE, return_palette_map=False,
                       palette: np.ndarray=None) -> Image.Image | np.ndarray:
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
    palette_map = np.zeros(img_arr.shape)

    for y in range(0, img_arr.shape[0]):
        for x in range(0, img_arr.shape[1]):
            c = img_arr[y][x]
            c_new = c
            if (dither == DITHER_MODES.BAYER):
                c_new = apply_threshold_map(c, dither_bayer_m, color_step, x, y)

            c_new_idx = int(c_new / color_step)
            c_new_idx = min(len(palette) - 1, max(0, c_new_idx))
            palette_map[y][x] = c_new_idx
            c_new = palette[c_new_idx]
            
            if (dither == DITHER_MODES.JJN):
                apply_jjn_error_diff(c, c_new, img_arr, x, y)
            
            if (dither == DITHER_MODES.FS):
                apply_fs_error_diff(c, c_new, img_arr, x, y)

            img_arr[y][x] = c_new
    
    if (return_palette_map):
        return palette_map

    img_arr = np.array(img_arr * 255, dtype=np.ubyte)
    return Image.frombytes("L", (img_arr.shape[1], img_arr.shape[0]), img_arr)

def preprocess_img(img: Image.Image,
                   scale_factor=1,
                   contrast=1,
                   brightness=1,
                   eq=0,
                   quantize_colors=255,
                   dither=DITHER_MODES.NONE):
    img = ImageOps.scale(img, scale_factor, Image.Resampling.BICUBIC)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = Image.blend(img, ImageOps.equalize(img), eq)
    img = quantize_grayscale(img, quantize_colors, dither)
    img = img.convert("L")
    return img
