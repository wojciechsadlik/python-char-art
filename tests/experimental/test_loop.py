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
from converters.img_converter import ImgConverter
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

        img_converter = ImgConverter(converter)

        if isinstance(converter, LineConverter):
            res_arr = img_converter.img2symbols(
                prep_img,
                max_cols=params.max_width,
                max_lines=params.max_height)
            res_img = render_symbols_img(res_arr, font, wh=prep_img.size)

        elif isinstance(converter, TileConverter):
            win_w = params.win_width
            if not win_w and params.max_width:
                win_w = math.ceil(params.max_width / prep_img.width)
            win_h = win_w * 2

            res_arr = img_converter.img2symbols(
                prep_img,
                win_wh=(win_w, win_h),
                max_cols=params.max_width,
                max_lines=params.max_height)
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
