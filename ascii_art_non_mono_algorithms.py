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
        text_arr.append(palette[random.randrange(0, len(palette))])
        bbox = text_draw.textbbox((0,0), ''.join(text_arr), font=font)

    text_arr.pop()

    return text_arr


def lazy_random_search(img, palette, font):
    lines = split_lines(img, palette, font)
    best_fit = -math.inf
    best_text_arr = []

    while True:
        text_arr = []
        for l in lines:
            r_l_text = generate_random_line(l, palette, font)
            text_arr.append(''.join(r_l_text) + '\n')
        fit = evaluate_text_arr(text_arr, img, font)
        if fit > best_fit:
            best_text_arr = text_arr
            best_fit = fit
        yield best_text_arr
        

def generate_greedy_line(line, palette, font):
    line_size = line.size
    _, text_draw = new_img_draw(line_size)
    text_arr = []
    bbox = text_draw.textbbox((0,0), ''.join(text_arr), font=font)

    while line_size[0] > bbox[2]:
        best_c = palette[0]
        text_arr.append(palette[0])
        best_c_fit = evaluate_text_arr(text_arr, line, font)
        text_arr.pop()
        for i in range(1, len(palette)):
            text_arr.append(palette[i])
            fit = evaluate_text_arr(text_arr, line, font)
            if fit > best_c_fit:
                best_c = palette[i]
                best_c_fit = fit
            text_arr.pop()
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


def generate_line_population(line, palette, font, count):
    population = []
    population.append(text_arr_to_palette_id_arr(generate_greedy_line(line, palette, font), palette))
    for _ in range(1, count):
        population.append(text_arr_to_palette_id_arr(generate_random_line(line, palette, font), palette))

    align_population_lengths(population, len(palette))

    population, fits = sort_population(population, palette, line, font)

    return population, fits


def generate_harmony_line(line, palette, font, iterations=100, pop_count=10, mem_rate=0.8, pa_rate=0.3, bw=2):
    population, fits = generate_line_population(line, palette, font, pop_count)
    el_len = len(population[0])
    
    for i in range(iterations):
        new_harm = []
        while len(new_harm) < el_len:
            if (random.random() < mem_rate):
                new_pitch = population[random.randrange(0, pop_count)][len(new_harm)]
                if (random.random() < pa_rate):
                    new_pitch += random.randint(-bw//2, bw//2)
                    new_pitch = new_pitch % len(palette)
                new_harm.append(new_pitch)
            else:
                new_harm.append(random.randrange(0, len(palette)))
        
        insert_into_sorted_population(population, fits, new_harm, palette, line, font)

        if i % 1000 == 0:
            print(min(fits), np.mean(fits))

    return population, fits
            

