import random
import math
import numpy as np
from PIL import ImageChops
from ascii_art_non_mono_utils import *

def generate_random_line(line, palette, font):
    line_size = line.size
    _, text_draw = new_img_draw(line.size)
    text_arr = []
    bbox = text_draw.textbbox((0,0), ''.join(text_arr), font=font)

    while line_size[0] > bbox[2]:
        text_arr.append(palette[random.randint(0, len(palette)-1)])
        bbox = text_draw.textbbox((0,0), ''.join(text_arr), font=font)

    text_arr.pop()

    return text_arr


def lazy_random_search(img, palette, font):
    lines = split_lines(img, palette, font)
    best_diff = math.inf
    best_text_arr = []

    while True:
        text_img, text_draw = new_img_draw(img.size)
        text_arr = []
        for l in lines:
            r_l_text = generate_random_line(l, palette, font)
            text_arr.append(''.join(r_l_text) + '\n')
        text_draw.multiline_text((0,0), ''.join(text_arr), font=font, fill=255)
        diff = np.average(ImageChops.difference(img, text_img))
        if diff < best_diff:
            best_text_arr = text_arr
            best_diff = diff
        yield best_text_arr
        

def generate_greedy_line(line, palette, font):
    line_size = line.size
    text_img, text_draw = new_img_draw(line_size)
    text_arr = []
    bbox = text_draw.textbbox((0,0), ''.join(text_arr), font=font)

    while line_size[0] > bbox[2]:
        best_c = palette[0]
        text_arr.append(palette[0])
        text_draw.text((0,0), ''.join(text_arr), font=font, fill=255)
        best_c_diff = np.average(ImageChops.difference(line, text_img))
        text_arr.pop()
        clear_img(text_img, text_draw)
        for i in range(1, len(palette)):
            text_arr.append(palette[i])
            text_draw.text((0,0), ''.join(text_arr), font=font, fill=255)
            diff = np.average(ImageChops.difference(line, text_img))
            if diff < best_c_diff:
                best_c = palette[i]
                best_c_diff = diff
            text_arr.pop()
            clear_img(text_img, text_draw)
        text_arr.append(best_c)
        bbox = text_draw.textbbox((0,0), ''.join(text_arr), font=font)
        
    text_arr.pop()

    return text_arr

def greedy_algorithm(img, palette, font):
    lines = split_lines(img, palette, font)
    
    text_arr = []
    for l in lines:
        l_text_arr = generate_greedy_line(l, palette, font)
        text_arr.append(''.join(l_text_arr) + '\n')

    return text_arr