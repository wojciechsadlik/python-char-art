import numpy as np
from PIL import Image
import random
from img_processing import *

def find_match(win, char_to_brightness_map, dither=DITHER_MODES.NONE, allow_err=0):
    min_dist = np.inf
    min_char = ''
    min_br = []
    close_char = []
    close_br = []
    for char, br in char_to_brightness_map.items():
        if (dither==DITHER_MODES.JJN):
            win_cp = np.copy(win)
            for w_y in range(br.shape[0]):
                for w_x in range(br.shape[1]):
                    apply_jjn_error_diff(win_cp[w_y][w_x], br[w_y][w_x], win_cp, w_x, w_y)
        if (dither==DITHER_MODES.FS):
            win_cp = np.copy(win)
            for w_y in range(br.shape[0]):
                for w_x in range(br.shape[1]):
                    apply_fs_error_diff(win_cp[w_y][w_x], br[w_y][w_x], win_cp, w_x, w_y)
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


def quantize_grayscale_mxn(img: Image.Image, char_to_brightness_map,
                           brightness_shape, dither=DITHER_MODES.NONE, allow_err=0.0) -> list[list[int, int]]:
    img_arr = np.array(img) / 255
    img_bounds = (img_arr.shape[0]-img_arr.shape[0]%brightness_shape[0],
                  img_arr.shape[1]-img_arr.shape[1]%brightness_shape[1])
    img_arr = img_arr[:img_bounds[0],:img_bounds[1]]
    char_arr = []
    for y in range(0, img_arr.shape[0], brightness_shape[0]):
        char_arr.append([])
        for x in range(0, img_arr.shape[1], brightness_shape[1]):
            win = np.copy(img_arr[y:y+brightness_shape[0], x:x+brightness_shape[1]])
            min_char, min_br = find_match(win, char_to_brightness_map, dither, allow_err)

            if (dither == DITHER_MODES.JJN):
                for w_y in range(brightness_shape[0]):
                    for w_x in range(brightness_shape[1]):
                        apply_jjn_error_diff(win[w_y][w_x], min_br[w_y][w_x], img_arr, x+w_x, y+w_y)
            
            
            if (dither == DITHER_MODES.FS):
                for w_y in range(brightness_shape[0]):
                    for w_x in range(brightness_shape[1]):
                        apply_fs_error_diff(win[w_y][w_x], min_br[w_y][w_x], img_arr, x+w_x, y+w_y)
                        
            char_arr[-1].append(min_char)
            
    return char_arr