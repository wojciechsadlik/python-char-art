import random
import math
from PIL import Image, ImageDraw, ImageChops, ImageFilter, ImageOps
import numpy as np
from IPython import display
from skimage import metrics

def new_img_draw(size, fill=0):
    img = Image.new("L", size, fill)
    draw = ImageDraw.Draw(img)
    return img, draw

def clear_img(img, draw, fill=0):
    draw.rectangle(((0,0), img.size), fill=fill)

def split_lines(img, palette, font):
    _, draw = new_img_draw(img.size)
    text = []
    bbox = draw.textbbox((0,0), ''.join(text), font=font)
    lines = []
    
    bbox = draw.textbbox((0,0), ''.join(palette), font=font)
    
    line_width = img.size[0]
    line_height = bbox[3]
    lines = []
    i = 0
    while i * line_height < img.size[1]:
        lines.append(img.crop((0, i * line_height, line_width, (i+1) * line_height)))
        i += 1
    return lines

def palette_id_arr_to_text_arr(p_id_arr, palette):
    text_arr = []
    for id in p_id_arr:
        text_arr.append(palette[id])
    return text_arr

def text_arr_to_palette_id_arr(text_arr, palette):
    p_id_arr = []
    for c in text_arr:
        p_id_arr.append(palette.index(c))
    return p_id_arr

def draw_text_arr(img_draw, text_arr, font):
    img_draw.multiline_text((0, 0), ''.join(text_arr), font=font, fill=255, spacing=1)

def evaluate_text_arr(text_arr, img, font):
    text_img, text_draw = new_img_draw(img.size, 0)
    draw_text_arr(text_draw, text_arr, font)
    return metrics.structural_similarity(np.array(text_img), np.array(img), win_size=7)

def evaluate_palette_id_arr(p_id_arr, palette, img, font):
    text_arr = palette_id_arr_to_text_arr(p_id_arr, palette)
    return evaluate_text_arr(text_arr, img, font)

def evaluate_palette_id_population(population, palette, img, font):
    pop_fit = []
    for el in population:
        pop_fit.append(evaluate_palette_id_arr(el, palette, img, font))
    return pop_fit

def align_population_lengths(population, length, fill_id):
    for i in range(len(population)):
        while len(population[i]) < length:
            population[i].append(fill_id)

def sort_population(population, palette, img, font):
    fits = evaluate_palette_id_population(population, palette, img, font)
    sorted_population = sorted(zip(fits, population), reverse=True)
    sorted_fits = list(map(lambda x: x[0], sorted_population))
    sorted_population = list(map(lambda x: x[1], sorted_population))
    return sorted_population, sorted_fits

def insert_into_sorted_population(population, fits, new_el, palette, img, font):
    new_fit = evaluate_palette_id_arr(new_el, palette, img, font)
    for i, f in enumerate(fits):
        if new_fit > f:
            fits.insert(i, new_fit)
            fits.pop()
            population.insert(i, new_el)
            population.pop()
            return
        
def calculate_longest_individual(img, palette, font):
    img_draw = ImageDraw.Draw(img)
    min_p_width = math.inf
    for p in palette:
        bbox = img_draw.textbbox((0,0), p, font=font)
        if bbox[2] < min_p_width:
            min_p_width = bbox[2]
    
    return math.ceil(img.size[0] / min_p_width)
