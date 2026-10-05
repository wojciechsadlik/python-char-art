import logging
import math
import random
from typing import Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sewar.full_ref import mse

from converters.evaluation import line_similarity
from diagnostics.artifact_manager import get_artifact_manager
from converters.tile.tile_converter import TileConverter
from image.array_ops import slice_horizontally
from rendering.image import render_symbols_img

logger = logging.getLogger(__name__)


def get_line_height(symbols: list[str], font: ImageFont.FreeTypeFont) -> int:
    img = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), "".join(symbols), font=font)
    return bbox[3] - bbox[1]


def get_char_width(symbols: list[str], font: ImageFont.FreeTypeFont) -> float:
    if not symbols:
        return 0.0
    img = Image.new("L", (1, 1))
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), "".join(symbols), font=font)
    return (bbox[2] - bbox[0]) / len(symbols)


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


def evaluate_symbol_arr(
    src_img: Image.Image,
    symbol_arr: list[list[str]],
    font: ImageFont.FreeTypeFont,
) -> float:
    res_img = render_symbols_img(symbol_arr, font)
    return line_similarity(src_img, res_img)


def evaluate_symbols_id_arr(
    p_id_arr: list[int],
    symbols: list[str],
    src_img: Image.Image,
    font: ImageFont.FreeTypeFont,
) -> float:
    symbol_arr = symbols_id_arr_to_text_arr(p_id_arr, symbols)
    return evaluate_symbol_arr(src_img, [symbol_arr], font)


def evaluate_symbols_id_population(
    population: list[list[int]],
    symbols: list[str],
    img: Image.Image,
    font: ImageFont.FreeTypeFont,
) -> list[float]:
    return [evaluate_symbols_id_arr(el, symbols, img, font) for el in population]


def sort_population(
    population: list[list[int]],
    symbols: list[str],
    img: Image.Image,
    font: ImageFont.FreeTypeFont,
) -> tuple[list[list[int]], list[float]]:
    fits = evaluate_symbols_id_population(population, symbols, img, font)
    sorted_population = sorted(
        zip(fits, population), key=lambda f_p: f_p[0], reverse=True
    )
    sorted_fits = [x[0] for x in sorted_population]
    sorted_pop = [x[1] for x in sorted_population]

    best_line = symbols_id_arr_to_text_arr(sorted_pop[0], symbols)

    get_artifact_manager().save_line(
        symbol_line=best_line,
        font=font,
        size=img.size,
        fitness=sorted_fits[0],
    )

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
            if i == 0:
                line = symbols_id_arr_to_text_arr(new_el, symbols)

                get_artifact_manager().save_line(
                    symbol_line=line,
                    font=font,
                    size=img.size,
                    fitness=new_fit,
                )

            fits.insert(i, new_fit)
            fits.pop()
            population.insert(i, new_el)
            population.pop()
            return


def generate_tile_line(
    line: Image.Image,
    tile_converter: TileConverter,
    max_cols: int,
) -> list[str]:
    line_arr = np.array(line)
    tile_arrs = slice_horizontally(line_arr, max_cols)
    return [tile_converter.tile2symbol(tile) for tile in tile_arrs]


def generate_random_line(symbols: list[str], max_cols: int) -> list[str]:
    return [random.choice(symbols) for _ in range(max_cols)]


def generate_line_population(
    line: Image.Image,
    symbols: list[str],
    font: ImageFont.FreeTypeFont,
    count: int,
    max_cols: int,
    include_greedy: bool = False,
    tile_converter: Optional[TileConverter] = None,
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
                    generate_tile_line(line, tile_converter, max_cols), symbols
                )
            )
        else:
            population.append(
                text_arr_to_symbols_id_arr(
                    generate_random_line(symbols, max_cols), symbols)
            )

    population, fits = sort_population(population, symbols, line, font)
    return population, fits
