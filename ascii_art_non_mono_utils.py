import random
from PIL import Image, ImageDraw, ImageChops, ImageFilter
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
    cmp_bbox = (0, 0, img.size[0], img.size[1])
    text_img = text_img.filter(ImageFilter.MaxFilter())
    text_img = text_img.filter(ImageFilter.GaussianBlur(1))
    cmp_img = img.crop(cmp_bbox)
    return metrics.structural_similarity(np.array(text_img), np.array(cmp_img), win_size=7)

def evaluate_palette_id_arr(p_id_arr, palette, img, font):
    text_arr = palette_id_arr_to_text_arr(p_id_arr, palette)
    return evaluate_text_arr(text_arr, img, font)

def evaluate_palette_id_population(population, palette, img, font):
    pop_fit = []
    for el in population:
        pop_fit.append(evaluate_palette_id_arr(el, palette, img, font))
    return pop_fit

def align_population_lengths(population, palette_len):
    max_len = len(population[0])
    for i in range(1, len(population)):
        if (len(population[i])) > max_len:
            max_len = len(population[i])

    for i in range(len(population)):
        while len(population[i]) < max_len:
            population[i].append(random.randrange(0, palette_len))

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