import numpy as np
from PIL import Image
import random
from img_processing import *


def find_match(win, char_to_brightness_map, randomize=False):
    max_dist = np.linalg.norm(np.ones(win.shape))
    max_sim = 0
    max_char = ''
    max_br = []
    close_char = []
    close_br = []
    for char, br in char_to_brightness_map.items():
        sim = 1 - np.linalg.norm(win - br) / max_dist

        if (sim > max_sim):
            max_sim = sim
            max_char = char
            max_br = br
        if (sim > 0.5):
            close_char.append(char)
            close_br.append(br)

    if (len(close_char) > 1 and randomize):
        rand_char_id = random.randrange(len(close_char))
        return close_char[rand_char_id], close_br[rand_char_id]
    return max_char, max_br


def scale_kernel(kernel, width, height, offset):
    scaled_kernel = np.copy(kernel)

    scaled_kernel = scaled_kernel.repeat(height, axis=0)
    scaled_kernel = scaled_kernel.repeat(width, axis=1)
    scaled_kernel = scaled_kernel / np.sum(scaled_kernel)
    return scaled_kernel, offset * width


def pick_cls_prediction(win, cls, char_to_brightness_map, randomize=False):
    if (not randomize):
        prediction = cls.predict([win.flatten()])[0]
        return prediction, char_to_brightness_map[prediction]

    prediction = cls.predict_proba([win.flatten()])[0]
    rand = random.random()
    prob_sum = 0
    for c, prob in zip(cls.classes_, prediction):
        prob_sum += prob
        if rand <= prob_sum:
            return c, char_to_brightness_map[c]


def quantize_grayscale_wxh(img: Image.Image,
                           char_to_brightness_map,
                           brightness_hw_shape,
                           dither=DITHER_MODES.NONE,
                           cls=None,
                           randomize=False) -> list[list[int,
                                                  int]]:
    img_arr = np.array(img) / 255

    if (dither == DITHER_MODES.JJN):
        scaled_jjn_k, jjn_k_offset = scale_kernel(
            jjn_k, brightness_hw_shape[1], brightness_hw_shape[0], 2)
    if (dither == DITHER_MODES.FS):
        scaled_fs_k, fs_k_offset = scale_kernel(
            fs_k, brightness_hw_shape[1], brightness_hw_shape[0], 1)
    if (dither == DITHER_MODES.ATKINSON):
        scaled_atkinson_k, atkinson_k_offset = scale_kernel(
            atkinson_k, brightness_hw_shape[1], brightness_hw_shape[0], 1)

    char_arr = []
    for up_y in range(
            brightness_hw_shape[0],
            img_arr.shape[0],
            brightness_hw_shape[0]):
        char_arr.append([])
        for up_x in range(
                brightness_hw_shape[1],
                img_arr.shape[1],
                brightness_hw_shape[1]):
            x = up_x - brightness_hw_shape[1]
            y = up_y - brightness_hw_shape[0]
            win = img_arr[y:up_y, x:up_x]
            if cls is None:
                min_char, min_br = find_match(
                    win, char_to_brightness_map, randomize)
            else:
                min_char, min_br = pick_cls_prediction(
                    win, cls, char_to_brightness_map, randomize)

            if (dither == DITHER_MODES.JJN):
                apply_error_diff_window(
                    min_br,
                    min_br.shape[1],
                    min_br.shape[0],
                    img_arr,
                    x,
                    y,
                    scaled_jjn_k,
                    jjn_k_offset)

            if (dither == DITHER_MODES.FS):
                apply_error_diff_window(
                    min_br,
                    min_br.shape[1],
                    min_br.shape[0],
                    img_arr,
                    x,
                    y,
                    scaled_fs_k,
                    fs_k_offset)

            if (dither == DITHER_MODES.ATKINSON):
                apply_error_diff_window(
                    min_br,
                    min_br.shape[1],
                    min_br.shape[0],
                    img_arr,
                    x,
                    y,
                    scaled_atkinson_k,
                    atkinson_k_offset)

            char_arr[-1].append(min_char)

    return char_arr
