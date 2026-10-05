from dataclasses import dataclass, field
from itertools import repeat
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import math
from typing import Callable

from PIL import Image as PILImage
from PIL.Image import Resampling, Image
from PIL.ImageFont import FreeTypeFont, truetype
import numpy as np
from sewar.full_ref import msssim

from converters.img_converter import ImgConverter
from converters.evaluation import img_similarity
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


def test_img(img_path: str, converter: ImgConverter, params: Params) -> float:
    font = truetype(params.font_path, params.font_size)

    with PILImage.open(img_path) as src_img:
        src_img_loaded = src_img.copy()

        prep_img = preprocess_img(src_img_loaded,
                                  grayscale=True,
                                  **params.preprocess_args)

        res_arr = converter.img2symbols(
            prep_img,
            max_cols=params.max_cols,
            max_lines=params.max_lines)
        res_img = render_symbols_img(res_arr, font)

        res_img = res_img.convert("L")
        return img_similarity(src_img_loaded, res_img)


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
                                       gens_per_step=gens_per_step)))

    def _advance_generator(item):
        img, generator = item
        try:
            res_arr = next(generator)
        except StopIteration:
            return None
        res_img = render_symbols_img(res_arr, font)
        return img_similarity(img, res_img)

    with ThreadPoolExecutor() as executor:
        for _ in range(gens // gens_per_step):
            results = list(executor.map(_advance_generator, img_generators))
            
            results = [r for r in results if r is not None]
            
            if not results:
                break
                
            print(f"Min: {min(results):.4f} | Max: {max(results):.4f} | "
                  f"Mean: {np.mean(results):.4f} | Std: {np.std(results):.4f}")
            similarity_scores.append(results)

    return similarity_scores
