import math
import random
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sewar.full_ref import mse

from converters.tile.tile_converter import TileConverter
from image.array_ops import slice_horizontally
from rendering.image import render_symbols_img


def get_line_height(symbols: list[str], font: ImageFont.FreeTypeFont) -> int:
    img = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), "".join(symbols), font=font)
    return bbox[3] - bbox[1]


def get_char_width(font: ImageFont.FreeTypeFont) -> float:
    img = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), "M", font=font)
    return bbox[2] - bbox[0]


def new_img_draw(size: tuple[int, int], fill: int = 0) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("L", size, fill)
    draw = ImageDraw.Draw(img)
    return img, draw


def symbols_id_arr_to_text_arr(p_id_arr: list[int], symbols: list[str]) -> list[str]:
    return [symbols[int(id)] for id in p_id_arr]


def text_arr_to_symbols_id_arr(text_arr: list[str], symbols: list[str]) -> list[int]:
    return [symbols.index(c) for c in text_arr]


def draw_text_arr(
    img_draw: ImageDraw.ImageDraw, text_arr: list[str], font: ImageFont.FreeTypeFont
) -> None:
    img_draw.multiline_text((0, 0), "".join(text_arr), font=font, fill=255)


def similarity(src_img: Image.Image, res_img: Image.Image) -> float:
    target_size = (
        min(src_img.width, res_img.width),
        min(src_img.height, res_img.height),
    )

    if src_img.size != target_size:
        src_img = src_img.resize(target_size, Image.Resampling.BICUBIC)

    if res_img.size != target_size:
        res_img = res_img.resize(target_size, Image.Resampling.BICUBIC)

    src_arr = np.array(src_img.convert("L")) / 255
    res_arr = np.array(res_img.convert("L")) / 255
    return -mse(src_arr, res_arr)


def evaluate_symbol_arr(
    src_img: Image.Image,
    symbol_arr: list[list[str]],
    font: ImageFont.FreeTypeFont,
    wh,
) -> float:
    res_img = render_symbols_img(symbol_arr, font, wh=wh)
    return similarity(src_img, res_img)


def evaluate_symbols_id_arr(
    p_id_arr: list[int],
    symbols: list[str],
    src_img: Image.Image,
    font: ImageFont.FreeTypeFont,
) -> float:
    symbol_arr = symbols_id_arr_to_text_arr(p_id_arr, symbols)
    return evaluate_symbol_arr(src_img, [symbol_arr], font, src_img.size)


def evaluate_symbols_id_population(
    population: list[list[int]],
    symbols: list[str],
    img: Image.Image,
    font: ImageFont.FreeTypeFont,
) -> list[float]:
    return [evaluate_symbols_id_arr(el, symbols, img, font) for el in population]


def align_population_lengths(
    population: list[list[int]],
    length: int,
    symbols_length: int = 1,
    fill_id: int = None,
) -> None:
    for i in range(len(population)):
        while len(population[i]) < length:
            if fill_id is not None:
                population[i].append(fill_id)
            else:
                population[i].append(random.randrange(0, symbols_length))


def sort_population(
    population: list[list[int]],
    symbols: list[str],
    img: Image.Image,
    font: ImageFont.FreeTypeFont,
) -> tuple[list[list[int]], list[float]]:
    fits = evaluate_symbols_id_population(population, symbols, img, font)
    sorted_population = sorted(
        zip(fits, population), key=lambda f_p: f_p[0], reverse=True)
    sorted_fits = [x[0] for x in sorted_population]
    sorted_pop = [x[1] for x in sorted_population]
    return sorted_pop, sorted_fits


def insert_into_sorted_population(
    population: list[list[int]],
    fits: list[float],
    new_el: list[int],
    symbols: list[str],
    img: Image.Image,
    font: ImageFont.FreeTypeFont,
) -> None:
    new_fit = evaluate_symbols_id_arr(new_el, symbols, img, font)
    for i, f in enumerate(fits):
        if new_fit > f:
            fits.insert(i, new_fit)
            fits.pop()
            population.insert(i, new_el)
            population.pop()
            return


def calculate_longest_individual(
    img: Image.Image, symbols: list[str], font: ImageFont.FreeTypeFont
) -> int:
    img_draw = ImageDraw.Draw(img)
    min_p_width = math.inf
    for p in symbols:
        bbox = img_draw.textbbox((0, 0), p, font=font)
        if bbox[2] < min_p_width:
            min_p_width = bbox[2]
    return int(math.floor(img.size[0] / min_p_width))


def generate_tile_line(
    line: Image.Image,
    tile_converter: TileConverter,
    col_width: Optional[int] = None,
) -> list[str]:
    col_width = col_width or line.height // 2
    line_arr = np.array(line)
    tile_arrs = slice_horizontally(line_arr, col_width)
    return [tile_converter.tile2symbol(tile) for tile in tile_arrs]


def generate_random_line(
    line: Image.Image, symbols: list[str], font: ImageFont.FreeTypeFont
) -> list[str]:
    line_size = line.size
    _, text_draw = new_img_draw(line.size)
    text_arr: list[str] = []
    bbox = text_draw.textbbox((0, 0), "".join(text_arr), font=font)
    while bbox[2] < line_size[0] + 4:
        text_arr.append(symbols[random.randrange(0, len(symbols))])
        bbox = text_draw.textbbox((0, 0), "".join(text_arr), font=font)
    if text_arr:
        text_arr.pop()
    return text_arr


def generate_line_population(
    line: Image.Image,
    symbols: list[str],
    font: ImageFont.FreeTypeFont,
    count: int,
    include_greedy: bool = False,
    tile_converter: Optional[TileConverter] = None,
    col_width: Optional[int] = None,
) -> tuple[list[list[int]], list[float]]:
    population: list[list[int]] = []

    if include_greedy:
        from converters.line_heuristics.greedy import generate_greedy_line
        population.append(
            text_arr_to_symbols_id_arr(
                generate_greedy_line(line, symbols, font), symbols)
        )

    for _ in range(len(population), count):
        if tile_converter is not None:
            population.append(
                text_arr_to_symbols_id_arr(
                    generate_tile_line(line, tile_converter, col_width), symbols
                )
            )
        else:
            population.append(
                text_arr_to_symbols_id_arr(
                    generate_random_line(line, symbols, font), symbols)
            )

    align_population_lengths(
        population,
        calculate_longest_individual(line, symbols, font),
        symbols_length=len(symbols),
    )
    population, fits = sort_population(population, symbols, line, font)
    return population, fits
