from dataclasses import dataclass
from itertools import repeat
from multiprocessing import Pool, cpu_count
import math

from PIL import Image as PILImage
from PIL.Image import Resampling, Image
from PIL.ImageFont import truetype
import numpy as np
from sewar.full_ref import msssim

from converters.tile.base import TileConverter
from converters.line_heuristics.base import LineConverter
from image.processing import preprocess_img
from rendering.image import render_symbols_img


Converter = TileConverter | LineConverter


@dataclass
class Params:
    converter_args: dict
    preprocess_args: dict
    font_path: str
    font_size: int
    win_width: int = None
    max_width: int = None
    max_height: int = None


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


def test_img(img_path: str, converter: Converter, params: Params) -> float:
    font = truetype(params.font_path, params.font_size)

    with PILImage.open(img_path) as src_img:
        src_img_loaded = src_img.copy()

        prep_img = preprocess_img(src_img_loaded,
                                  grayscale=True,
                                  **params.preprocess_args)

        scale_h = scale_w = 1.0
        if isinstance(converter, LineConverter):
            if params.max_width:
                font_width_px = params.font_size * 0.5
                scale_w = params.max_width * font_width_px / prep_img.width
            if params.max_height:
                font_height_px = params.font_size * 1.333
                scale_h = params.max_height * font_height_px / prep_img.height
            prep_img = preprocess_img(prep_img, scale_factor=min(scale_h, scale_w))
            res_arr = converter.process_image(prep_img)
            res_img = render_symbols_img(res_arr, font, wh=prep_img.size)
        elif isinstance(converter, TileConverter):
            win_w = params.win_width
            if not win_w:
                win_w = math.ceil(params.max_width / prep_img.width)
            win_h = win_w * 2
            if params.max_width:
                scale_w = params.max_width * win_w / prep_img.width
            if params.max_height:
                scale_h = params.max_height * win_h / prep_img.height
            prep_img = preprocess_img(prep_img, scale_factor=min(scale_h, scale_w))
            res_arr = converter.process_image(prep_img, (win_w, win_h))
            res_img = render_symbols_img(res_arr, font)
        res_img = res_img.convert("L")
        return similarity(src_img_loaded, res_img)


def test_converter_parallel(
        img_paths: list[str],
        converter: Converter,
        params: Params) -> list[float]:
    with Pool(cpu_count()) as p:
        sim_scores = p.starmap(
            test_img,
            zip(img_paths, repeat(converter), repeat(params))
        )
    return sim_scores


def test_loop(img_paths: list[str], make_converter: Converter,
              params_space: list[Params]) -> list[list[float]]:
    similarity_scores = []
    for params in params_space:
        converter = make_converter(params)
        results = test_converter_parallel(img_paths, converter, params)

        print(f"Min: {min(results):.4f} | Max: {max(results):.4f} | "
              f"Mean: {np.mean(results):.4f} | Std: {np.std(results):.4f}")
        similarity_scores.append(results)

    return similarity_scores
