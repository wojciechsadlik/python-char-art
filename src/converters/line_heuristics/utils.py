import random
import math
from PIL import Image, ImageDraw, ImageChops, ImageFont
import numpy as np

SPACING: int = 2


def new_img_draw(size: tuple[int, int],
                 fill: int = 0) -> tuple[Image.Image,
                                         ImageDraw.ImageDraw]:
    img = Image.new("L", size, fill)
    draw = ImageDraw.Draw(img)
    return img, draw


def split_lines(img: Image.Image,
                symbols: list[str],
                font: ImageFont.FreeTypeFont,
                spacing: int = SPACING) -> list[Image.Image]:
    _, draw = new_img_draw(img.size)
    bbox = draw.textbbox((0, 0), ''.join(symbols), font=font)
    line_top_margin = bbox[1]
    line_width = img.size[0]
    line_height = bbox[3]
    lines: list[Image.Image] = []
    top = 0
    bottom = line_height
    while img.size[1] - bottom > line_height // 2:
        lines.append(img.crop((0, top, line_width, bottom)))
        top = bottom - line_top_margin + spacing
        bottom = top + line_height
    return lines


def symbols_id_arr_to_text_arr(
        p_id_arr: list[int],
        symbols: list[str]) -> list[str]:
    return [symbols[int(id)] for id in p_id_arr]


def text_arr_to_symbols_id_arr(
        text_arr: list[str],
        symbols: list[str]) -> list[int]:
    return [symbols.index(c) for c in text_arr]


def draw_text_arr(
        img_draw: ImageDraw.ImageDraw,
        text_arr: list[str],
        font: ImageFont.FreeTypeFont,
        spacing: int = SPACING) -> None:
    img_draw.multiline_text((0, 0), ''.join(text_arr),
                            font=font, fill=255, spacing=spacing)


def evaluate_text_arr(
        text_arr: list[str],
        img: Image.Image,
        font: ImageFont.FreeTypeFont) -> float:
    text_img, text_draw = new_img_draw(img.size)
    draw_text_arr(text_draw, text_arr, font)
    return float(1 - np.mean(ImageChops.difference(text_img, img)) / 255)


def evaluate_symbols_id_arr(
        p_id_arr: list[int],
        symbols: list[str],
        img: Image.Image,
        font: ImageFont.FreeTypeFont) -> float:
    text_arr = symbols_id_arr_to_text_arr(p_id_arr, symbols)
    return evaluate_text_arr(text_arr, img, font)


def evaluate_symbols_id_population(population: list[list[int]],
                                   symbols: list[str],
                                   img: Image.Image,
                                   font: ImageFont.FreeTypeFont) -> list[float]:
    return [evaluate_symbols_id_arr(el, symbols, img, font)
            for el in population]


def align_population_lengths(population: list[list[int]],
                             length: int,
                             symbols_length: int = 1,
                             fill_id: int = None) -> None:
    for i in range(len(population)):
        while len(population[i]) < length:
            if fill_id is not None:
                population[i].append(fill_id)
            else:
                population[i].append(random.randrange(0, symbols_length))


def sort_population(population: list[list[int]],
                    symbols: list[str],
                    img: Image.Image,
                    font: ImageFont.FreeTypeFont) -> tuple[list[list[int]],
                                                           list[float]]:
    fits = evaluate_symbols_id_population(population, symbols, img, font)
    sorted_population = sorted(zip(fits, population), reverse=True)
    sorted_fits = [x[0] for x in sorted_population]
    sorted_pop = [x[1] for x in sorted_population]
    return sorted_pop, sorted_fits


def insert_into_sorted_population(population: list[list[int]],
                                  fits: list[float],
                                  new_el: list[int],
                                  symbols: list[str],
                                  img: Image.Image,
                                  font: ImageFont.FreeTypeFont) -> None:
    new_fit = evaluate_symbols_id_arr(new_el, symbols, img, font)
    for i, f in enumerate(fits):
        if new_fit > f:
            fits.insert(i, new_fit)
            fits.pop()
            population.insert(i, new_el)
            population.pop()
            return


def calculate_longest_individual(
        img: Image.Image,
        symbols: list[str],
        font: ImageFont.FreeTypeFont) -> int:
    img_draw = ImageDraw.Draw(img)
    min_p_width = math.inf
    for p in symbols:
        bbox = img_draw.textbbox((0, 0), p, font=font)
        if bbox[2] < min_p_width:
            min_p_width = bbox[2]
    return int(math.ceil(img.size[0] / min_p_width))


def generate_random_line(
        line: Image.Image,
        symbols: list[str],
        font: ImageFont.FreeTypeFont) -> list[str]:
    line_size = line.size
    _, text_draw = new_img_draw(line.size)
    text_arr: list[str] = []
    bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)
    while line_size[0] > bbox[2]:
        text_arr.append(symbols[random.randrange(0, len(symbols))])
        bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)
    if text_arr:
        text_arr.pop()
    return text_arr


def generate_greedy_line(
        line: Image.Image,
        symbols: list[str],
        font: ImageFont.FreeTypeFont) -> list[str]:
    line_size = line.size
    _, text_draw = new_img_draw(line_size)
    text_arr: list[str] = []
    bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)
    while line_size[0] > bbox[2]:
        best_c = symbols[0]
        text_arr.append(symbols[0])
        best_c_fit = evaluate_text_arr(text_arr, line, font)
        text_arr.pop()
        for i in range(1, len(symbols)):
            text_arr.append(symbols[i])
            fit = evaluate_text_arr(text_arr, line, font)
            if fit > best_c_fit:
                best_c = symbols[i]
                best_c_fit = fit
            text_arr.pop()
        text_arr.append(best_c)
        bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)
    if text_arr:
        text_arr.pop()
    return text_arr


def generate_line_population(line: Image.Image,
                             symbols: list[str],
                             font: ImageFont.FreeTypeFont,
                             count: int,
                             include_greedy: bool = False
                             ) -> tuple[list[list[int]], list[float]]:
    population: list[list[int]] = []
    if include_greedy:
        population.append(
            text_arr_to_symbols_id_arr(
                generate_greedy_line(
                    line, symbols, font), symbols))
    for _ in range(len(population), count):
        population.append(
            text_arr_to_symbols_id_arr(
                generate_random_line(
                    line, symbols, font), symbols))
    align_population_lengths(
        population,
        calculate_longest_individual(
            line,
            symbols,
            font),
        symbols_length=len(symbols))
    population, fits = sort_population(population, symbols, line, font)
    return population, fits
