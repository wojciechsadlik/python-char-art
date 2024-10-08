import random
import math
import copy
from multiprocessing import Pool, cpu_count
from itertools import starmap, repeat
import numpy as np
from PIL import ImageChops
from non_mono_char_art_utils import *

USE_CPU = 1 + cpu_count() // 2


def generate_random_line(line, palette, font):
    line_size = line.size
    _, text_draw = new_img_draw(line.size)
    text_arr = []
    bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)

    while line_size[0] > bbox[2]:
        text_arr.append(palette[random.randrange(0, len(palette))])
        bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)

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
    bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)

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
        bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)

    text_arr.pop()

    return text_arr


def greedy_algorithm(img, palette, font):
    lines = split_lines(img, palette, font)

    text_arr = []
    with Pool(USE_CPU) as p:
        text_arr = p.starmap(
            generate_greedy_line, zip(
                lines, repeat(palette), repeat(font)))

    for i in range(len(text_arr)):
        text_arr[i] = ''.join(text_arr[i]) + '\n'
    return text_arr


def generate_line_population(line, palette, font, count, include_greedy=False):
    population = []
    if (include_greedy):
        population.append(
            text_arr_to_palette_id_arr(
                generate_greedy_line(
                    line, palette, font), palette))
    for _ in range(len(population), count):
        population.append(
            text_arr_to_palette_id_arr(
                generate_random_line(
                    line, palette, font), palette))

    align_population_lengths(
        population,
        calculate_longest_individual(
            line,
            palette,
            font),
        palette_length=len(palette))

    population, fits = sort_population(population, palette, line, font)

    return population, fits


def lazy_generate_harmony_line(
        line,
        palette,
        font,
        pop_count=10,
        mem_rate=0.8,
        pa_rate=0.3,
        bw=2,
        include_greedy=False):
    population, fits = generate_line_population(
        line, palette, font, pop_count, include_greedy)
    generation = 0

    while True:
        generation += 1
        new_harm = HarmonyLineSearch.new_harmony_line(
            palette, population, mem_rate, pa_rate, bw)
        insert_into_sorted_population(
            population, fits, new_harm, palette, line, font)
        yield population[0], fits[0]


class HarmonyLineSearch:
    def __init__(
            self,
            line,
            palette,
            font,
            pop_count,
            mem_rate,
            pa_rate,
            bw,
            include_greedy=False):
        self.line = line
        self.palette = palette
        self.font = font
        self.mem_rate = mem_rate
        self.pa_rate = pa_rate
        self.bw = bw
        self.population, self.fits = generate_line_population(
            line, palette, font, pop_count, include_greedy)
        self.generation = 0

    def new_harmony_line(palette, population, mem_rate=0.8, pa_rate=0.3, bw=2):
        new_harm = []
        while len(new_harm) < len(population[0]):
            if (random.random() < mem_rate):
                new_pitch = population[random.randrange(
                    0, len(population))][len(new_harm)]
                if (random.random() < pa_rate):
                    new_pitch += random.randint(-bw, bw) // 2
                    new_pitch = new_pitch % len(palette)
                new_harm.append(new_pitch)
            else:
                new_harm.append(random.randrange(0, len(palette)))

        return new_harm

    def next_line(self, incr):
        self.generation += incr
        for _ in range(incr):
            new_harm = HarmonyLineSearch.new_harmony_line(
                self.palette, self.population, self.mem_rate, self.pa_rate, self.bw)
            insert_into_sorted_population(
                self.population,
                self.fits,
                new_harm,
                self.palette,
                self.line,
                self.font)
        return vars(self)

    def get_solution(self):
        return self.population[0]


def lazy_harmony_search(
        img,
        palette,
        font,
        pop_count=10,
        mem_rate=0.8,
        pa_rate=0.3,
        bw=2,
        include_greedy=False):
    lines = split_lines(img, palette, font)
    line_generators = []
    for l in lines:
        line_generators.append(
            lazy_generate_harmony_line(
                l,
                palette,
                font,
                pop_count,
                mem_rate,
                pa_rate,
                bw,
                include_greedy))

    while True:
        text_arr = []
        l_p_ids = [l[0] for l in map(next, line_generators)]
        l_text_arr = starmap(
            palette_id_arr_to_text_arr, zip(
                l_p_ids, repeat(palette)))
        text_arr = list(l + '\n' for l in map(''.join, l_text_arr))
        yield text_arr


class GeneticLineSearch:
    def __init__(
            self,
            line,
            palette,
            font,
            pop_count,
            mutation_rate=0.3,
            mutation_bw=2,
            include_greedy=False):
        self.line = line
        self.palette = palette
        self.font = font
        self.mutation_rate = mutation_rate
        self.mutation_bw = mutation_bw
        self.pop_count = pop_count
        self.population, self.fits = generate_line_population(
            line, palette, font, pop_count, include_greedy)
        self.generation = 0

    def genetic_mutation(el, val_range, mutation_rate=0.3, mutation_bw=2):
        for i in range(len(el)):
            if (random.random() < mutation_rate):
                el[i] = el[i] + random.randint(-mutation_bw, mutation_bw) // 2
                el[i] = el[i] % val_range

    def new_genetic_line_population(
            palette,
            population,
            mutation_rate=0.3,
            mutation_bw=2):
        el_len = len(population[0])
        cross_point1 = el_len // 3
        cross_point2 = 2 * cross_point1
        new_population = []
        i_ps = list(range(len(population)))
        random.shuffle(i_ps)
        while len(i_ps) > 0:
            p1 = population[i_ps.pop()]
            p2 = population[i_ps.pop()]

            new1 = p1[0:cross_point1] + \
                p2[cross_point1:cross_point2] + p1[cross_point2:]
            new2 = p2[0:cross_point1] + \
                p1[cross_point1:cross_point2] + p2[cross_point2:]

            GeneticLineSearch.genetic_mutation(
                new1, len(palette), mutation_rate, mutation_bw)
            GeneticLineSearch.genetic_mutation(
                new2, len(palette), mutation_rate, mutation_bw)

            new_population.append(new1)
            new_population.append(new2)

        return new_population

    def next_line(self, incr):
        self.generation += incr
        for _ in range(incr):
            new_population = GeneticLineSearch.new_genetic_line_population(
                self.palette, self.population, self.mutation_rate, self.mutation_bw)
            for el in new_population:
                insert_into_sorted_population(
                    self.population, self.fits, el, self.palette, self.line, self.font)
                self.population = self.population[:self.pop_count]
        return vars(self)

    def get_solution(self):
        return self.population[0]


class ParticleSwarmLineSearch:
    def __init__(
            self,
            line,
            palette,
            font,
            pop_count,
            innertion,
            cog_coeff,
            soc_coeff,
            include_greedy=False):
        self.line = line
        self.palette = palette
        self.font = font
        self.innertion = innertion
        self.cog_coeff = cog_coeff
        self.soc_coeff = soc_coeff
        self.particles, self.fits = generate_line_population(
            line, palette, font, pop_count, include_greedy)
        self.particles = np.array(self.particles, dtype=np.float32)
        self.pos_len = len(self.particles[0])
        self.best_particle_pos = copy.deepcopy(self.particles)
        self.best_particle_pos_fit = copy.deepcopy(self.fits)
        self.best_global_pos = copy.deepcopy(self.particles[0])
        self.best_global_pos_fit = self.fits[0]
        self.generation = 0
        self.init_velocities()

    def init_velocities(self):
        self.velocities = []
        for _ in range(len(self.particles)):
            v = 2 * len(self.palette) * \
                np.random.random(self.pos_len) - len(self.palette)
            self.velocities.append(v)

    def recalculate_velocity(
            self,
            velocity,
            particle_pos,
            best_particle_pos,
            best_global_pos):
        r_p = np.random.random(velocity.shape)
        r_g = np.random.random(velocity.shape)
        velocity = self.innertion * velocity
        velocity = self.cog_coeff * r_p * (best_particle_pos - particle_pos)
        velocity = self.soc_coeff * r_g * (best_global_pos - particle_pos)
        if (np.linalg.norm(velocity) == 0):
            velocity = 2 * len(self.palette) * \
                np.random.random(self.pos_len) - len(self.palette)
        return velocity

    def next_line(self, incr):
        self.generation += incr
        for _ in range(incr):
            for i in range(len(self.particles)):
                self.velocities[i] = self.recalculate_velocity(
                    self.velocities[i],
                    self.particles[i],
                    self.best_particle_pos[i],
                    self.best_global_pos)
                self.particles[i] += self.velocities[i]

                try:
                    self.fits[i] = evaluate_palette_id_arr(
                        map(int, self.particles[i]), self.palette, self.line, self.font)
                except IndexError:
                    self.fits[i] = 0
                if self.fits[i] > self.best_particle_pos_fit[i]:
                    self.best_particle_pos[i] = copy.deepcopy(
                        self.particles[i])
                    self.best_particle_pos_fit[i] = self.fits[i]
                    if self.fits[i] > self.best_global_pos_fit:
                        self.best_global_pos = copy.deepcopy(self.particles[i])
                        self.best_global_pos_fit = self.fits[i]
        return vars(self)

    def get_solution(self):
        return list(map(int, self.best_global_pos))


def pool_evolutionary_search(finder_class, line_finders, generations, palette):
    while True:
        with Pool(USE_CPU) as p:
            next_states = p.starmap(
                finder_class.next_line, zip(
                    line_finders, repeat(generations)))
        for f, s in zip(line_finders, next_states):
            f.__dict__.update(s)
        l_p_ids = map(finder_class.get_solution, line_finders)
        l_text_arr = starmap(
            palette_id_arr_to_text_arr, zip(
                l_p_ids, repeat(palette)))
        text_arr = list(l + '\n' for l in map(''.join, l_text_arr))
        yield text_arr
