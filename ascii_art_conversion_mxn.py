import numpy as np
from PIL import Image
import random
from img_processing import *

def apply_error_diff_window(c_new, c_width, c_height, img_arr, x, y, kernel, kernel_off_x):
    c = img_arr[y:y+c_height,x:x+c_width]
    c_err = np.linalg.norm(c - c_new)
    if np.linalg.norm(c) < np.linalg.norm(c_new):
        c_err = -c_err
    c_err_kernel = c_err * kernel
    x_left = x - kernel_off_x
    if x_left < 0:
        c_err_kernel = c_err_kernel[:,-x_left:]
        x_left = 0
    x_right = x_left + c_err_kernel.shape[1]
    if x_right > img_arr.shape[1]:
        c_err_kernel = c_err_kernel[:,:c_err_kernel.shape[1]-(x_right-img_arr.shape[1])]
        x_right = img_arr.shape[1]
    top = y
    bottom = y + c_err_kernel.shape[0]
    if bottom > img_arr.shape[0]:
        c_err_kernel = c_err_kernel[:c_err_kernel.shape[0]-(bottom-img_arr.shape[0]),:]
        bottom = img_arr.shape[0]
    img_arr[top:bottom, x_left:x_right] += c_err_kernel

def find_match(win, char_to_brightness_map, dither=DITHER_MODES.NONE, allow_err=0):
    min_dist = np.inf
    min_char = ''
    min_br = []
    close_char = []
    close_br = []
    for char, br in char_to_brightness_map.items():
        # if (dither==DITHER_MODES.JJN):
        #     win_cp = np.copy(win)
        #     for w_y in range(br.shape[0]):
        #         for w_x in range(br.shape[1]):
        #             apply_error_diff(br[w_y][w_x], win_cp, w_x, w_y, jjn_k, 2)
        # if (dither==DITHER_MODES.FS):
        #     win_cp = np.copy(win)
        #     for w_y in range(br.shape[0]):
        #         for w_x in range(br.shape[1]):
        #             apply_error_diff(br[w_y][w_x], win_cp, w_x, w_y, fs_k, 1)
        dist = np.linalg.norm(win-br)

        if dist < min_dist:
            min_dist = dist
            min_char = char
            min_br = br
        if dist <= allow_err:
            close_char.append(char)
            close_br.append(br)

    if allow_err > 0 and len(close_char) > 0:
        rand_char_id = random.randrange(len(close_char))
        return close_char[rand_char_id], close_br[rand_char_id] 
    return min_char, min_br

def scale_kernel(kernel, width, height, offset):
    scaled_kernel = np.copy(kernel)
    scaled_kernel = scaled_kernel.repeat(height,axis=0)
    scaled_kernel = scaled_kernel.repeat(width, axis=1)
    scaled_kernel = scaled_kernel / np.sum(scaled_kernel)
    return scaled_kernel, offset * width


def quantize_grayscale_mxn(img: Image.Image, char_to_brightness_map,
                           brightness_hw_shape, dither=DITHER_MODES.NONE, allow_err=0.0) -> list[list[int, int]]:
    img_arr = np.array(img) / 255

    if (dither == DITHER_MODES.JJN):
        scaled_jjn_k, jjn_k_offset = scale_kernel(jjn_k, brightness_hw_shape[1], brightness_hw_shape[0], 2)
    if (dither == DITHER_MODES.FS):
        scaled_fs_k, fs_k_offset = scale_kernel(fs_k, brightness_hw_shape[1], brightness_hw_shape[0], 1)

    char_arr = []
    for up_y in range(brightness_hw_shape[0], img_arr.shape[0], brightness_hw_shape[0]):
        char_arr.append([])
        for up_x in range(brightness_hw_shape[1], img_arr.shape[1], brightness_hw_shape[1]):
            x = up_x - brightness_hw_shape[1]
            y = up_y - brightness_hw_shape[0]
            win = np.copy(img_arr[y:up_y, x:up_x])
            min_char, min_br = find_match(win, char_to_brightness_map, dither, allow_err)

            if (dither == DITHER_MODES.JJN):
                apply_error_diff_window(min_br, min_br.shape[1], min_br.shape[0], img_arr, x, y, scaled_jjn_k, jjn_k_offset)
            
            if (dither == DITHER_MODES.FS):
                apply_error_diff_window(min_br, min_br.shape[1], min_br.shape[0], img_arr, x, y, scaled_fs_k, fs_k_offset)
                        
            char_arr[-1].append(min_char)
            
    return char_arr