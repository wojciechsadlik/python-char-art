import random
from PIL import Image, ImageDraw, ImageChops
import numpy as np

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
    img_size = img.size
    lines = []
    while bbox[3] < img_size[1]:
        text.append(''.join(palette) + '\n')
        bbox = draw.textbbox((0,0), ''.join(text), font=font)

    line_width = img.size[0]
    line_height = img.size[1] // len(text)
    lines = []
    for i in range(len(text)):
        lines.append(img.crop((0, i * line_height, line_width, (i+1) * line_height)))
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

def evaluate_text_arr(text_arr, img, font):
    text_img, text_draw = new_img_draw(img.size, int(np.mean(img)))
    text_draw.text((0,0), ''.join(text_arr), font=font, fill=255)
    return -np.mean(ImageChops.difference(text_img, img))

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