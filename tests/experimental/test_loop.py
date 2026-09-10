from dataclasses import dataclass, field
from itertools import repeat
from multiprocessing import Pool, cpu_count
import math
from typing import Callable

from PIL import Image as PILImage
from PIL.Image import Resampling, Image
from PIL.ImageFont import FreeTypeFont, truetype
import numpy as np
from sewar.full_ref import msssim

from converters.img_converter import ImgConverter
from converters.line_heuristics.utils import get_char_width, get_line_height
from image.processing import preprocess_img
from rendering.image import render_symbols_img


@dataclass
class Params:
    font_path: str
    font_size: int
    max_cols: int
    max_lines: int
    gens_per_step: int = 1000
    converter_args: dict = field(default_factory=dict)
    preprocess_args: dict = field(default_factory=dict)


def similarity(src_img: Image, res_img: Image):
    target_size = (min(src_img.width, res_img.width),
                   min(src_img.height, res_img.height))
    min_dim = min(target_size)
    if min_dim < 176:
        scale = 176 / min_dim
        target_size = (int(target_size[0] * scale),
                       int(target_size[1] * scale))

    if src_img.size != target_size:
        src_img = src_img.resize(target_size, Resampling.BICUBIC)

    if res_img.size != target_size:
        res_img = res_img.resize(target_size, Resampling.BICUBIC)

    src_img = np.array(src_img.convert("L"))
    res_img = np.array(res_img.convert("L"))
    return np.real(msssim(src_img, res_img))


def test_img(img_path: str, converter: ImgConverter, params: Params) -> float:
    font = truetype(params.font_path, params.font_size)
    line_height = int(params.font_size * 1.333)
    col_width = int(line_height / 2)

    with PILImage.open(img_path) as src_img:
        src_img_loaded = src_img.copy()

        prep_img = preprocess_img(src_img_loaded,
                                  grayscale=True,
                                  **params.preprocess_args)

        res_arr = converter.img2symbols(
            prep_img,
            max_cols=params.max_cols,
            max_lines=params.max_lines,
            col_width=col_width,
            line_height=line_height)
        res_img = render_symbols_img(res_arr, font)

        res_img = res_img.convert("L")
        return similarity(src_img_loaded, res_img)


def test_converter_parallel(
        img_paths: list[str],
        converter: ImgConverter,
        params: Params) -> list[float]:
    with Pool(cpu_count()) as p:
        sim_scores = p.starmap(
            test_img,
            zip(img_paths, repeat(converter), repeat(params))
        )
    return sim_scores


def test_loop(img_paths: list[str],
              make_converter: Callable[[Params], ImgConverter],
              params_space: list[Params]
              ) -> list[list[float]]:
    similarity_scores = []
    for params in params_space:
        converter = make_converter(params)
        results = test_converter_parallel(img_paths, converter, params)

        print(f"Min: {min(results):.4f} | Max: {max(results):.4f} | "
              f"Mean: {np.mean(results):.4f} | Std: {np.std(results):.4f}")
        similarity_scores.append(results)

    return similarity_scores


def lazy_converter_loop(img_paths: list[str],
                        converter: ImgConverter,
                        font: FreeTypeFont,
                        max_cols: int,
                        max_lines: int,
                        col_width: int,
                        line_height: int,
                        gens: int,
                        gens_per_step: int) -> list[list[float]]:
    similarity_scores = []
    img_generators = []
    for img_path in img_paths:
        img = PILImage.open(img_path).convert("L")
        img_generators.append((
            img,
            converter.img2symbols_lazy(img,
                                        max_cols=max_cols,
                                        max_lines=max_lines,
                                        col_width=col_width,
                                        line_height=line_height,
                                        gens_per_step=gens_per_step)))

    for _ in range(gens // gens_per_step):
        results = []
        for img, generator in img_generators:
            try:
                res_arr = next(generator)
            except StopIteration:
                break
            res_img = render_symbols_img(res_arr, font)
            results.append(similarity(img, res_img))
        if not results:
            break
        print(f"Min: {min(results):.4f} | Max: {max(results):.4f} | "
            f"Mean: {np.mean(results):.4f} | Std: {np.std(results):.4f}")
        similarity_scores.append(results)

    return similarity_scores
