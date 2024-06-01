import random
import math
from multiprocessing import Pool, cpu_count
from itertools import starmap, repeat
import numpy as np
from PIL import ImageChops
from ascii_art_non_mono_utils import *

USE_CPU = 1 + cpu_count() // 2

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
    with Pool(USE_CPU) as p:
        text_arr = p.starmap(generate_greedy_line, zip(lines, repeat(palette), repeat(font)))

    for i in range(len(text_arr)):
        text_arr[i] = ''.join(text_arr[i]) + '\n'
    return text_arr


def generate_line_population(line, palette, font, count, include_greedy=False):
    population = []
    if (include_greedy):
        population.append(text_arr_to_palette_id_arr(generate_greedy_line(line, palette, font), palette))
    for _ in range(len(population), count):
        population.append(text_arr_to_palette_id_arr(generate_random_line(line, palette, font), palette))

    align_population_lengths(population, calculate_longest_individual(line, palette, font), 0)

    population, fits = sort_population(population, palette, line, font)

    return population, fits

def new_harmony_line(palette, population, mem_rate=0.8, pa_rate=0.3, bw=2):
    new_harm = []
    while len(new_harm) < len(population[0]):
        if (random.random() < mem_rate):
            new_pitch = population[random.randrange(0, len(population))][len(new_harm)]
            if (random.random() < pa_rate):
                new_pitch += random.randint(-bw, bw) // 2
                new_pitch = new_pitch % len(palette)
            new_harm.append(new_pitch)
        else:
            new_harm.append(random.randrange(0, len(palette)))
    
    return new_harm

def lazy_generate_harmony_line(line, palette, font, pop_count=10, mem_rate=0.8, pa_rate=0.3, bw=2, include_greedy=False):
    population, fits = generate_line_population(line, palette, font, pop_count, include_greedy)
    generation = 0
    
    while True:
        generation += 1
        new_harm = new_harmony_line(palette, population, mem_rate, pa_rate, bw)
        insert_into_sorted_population(population, fits, new_harm, palette, line, font)
        yield population[0], fits[0]

class HarmonyLineSearch:
    def __init__(self, line, palette, font, pop_count, mem_rate, pa_rate, bw, include_greedy=False):
        self.line = line
        self.palette = palette
        self.font = font
        self.mem_rate = mem_rate
        self.pa_rate = pa_rate
        self.bw = bw
        self.population, self.fits = generate_line_population(line, palette, font, pop_count, include_greedy)
        self.generation = 0

    def next_line(self, incr):
        self.generation += incr
        for _ in range(incr):
            new_harm = new_harmony_line(self.palette, self.population, self.mem_rate, self.pa_rate, self.bw)
            insert_into_sorted_population(self.population, self.fits, new_harm, self.palette, self.line, self.font)
        return vars(self)
    
    def get_solution(self):
        return self.population[0]
    
def lazy_harmony_search(img, palette, font, pop_count=10, mem_rate=0.8, pa_rate=0.3, bw=2, include_greedy=False):
    lines = split_lines(img, palette, font)
    line_generators = []
    for l in lines:
        line_generators.append(lazy_generate_harmony_line(l, palette, font, pop_count, mem_rate, pa_rate, bw, include_greedy))

    while True:
        text_arr = []
        l_p_ids = [l[0] for l in map(next, line_generators)]
        l_text_arr = starmap(palette_id_arr_to_text_arr, zip(l_p_ids, repeat(palette)))
        text_arr = list(l + '\n' for l in map(''.join, l_text_arr))
        yield text_arr

def pool_harmony_search(img, palette, font, generations, pop_count=10, mem_rate=0.8, pa_rate=0.3, bw=2, include_greedy=False):
    lines = split_lines(img, palette, font)
    line_finders = []
    for l in lines:
        line_finders.append(HarmonyLineSearch(l, palette, font, pop_count, mem_rate, pa_rate, bw, include_greedy))

    while True:
        with Pool(USE_CPU) as p:
            next_states = p.starmap(HarmonyLineSearch.next_line, zip(line_finders, repeat(generations)))
        for f, s in zip(line_finders, next_states):
            f.__dict__.update(s)
        l_p_ids = map(HarmonyLineSearch.get_solution, line_finders)
        l_text_arr = starmap(palette_id_arr_to_text_arr, zip(l_p_ids, repeat(palette)))
        text_arr = list(l + '\n' for l in map(''.join, l_text_arr))
        yield text_arr
