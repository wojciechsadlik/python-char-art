from copy import deepcopy
import logging
from typing import Generator, Optional
from PIL import Image, ImageFont
import numpy as np

from diagnostics.artifact_manager import get_artifact_manager
from image.array_ops import slice_vertically
from converters.line_heuristics.line_converter import LineConverter

from converters.line_heuristics.utils import (
    symbols_id_arr_to_text_arr, 
    get_char_width, 
    get_line_height
)
from converters.evaluation import img_similarity
from rendering.image import render_symbols_img

logger = logging.getLogger(__name__)


class ImgConverter:
    def __init__(
        self,
        converter: LineConverter,
        font: ImageFont.FreeTypeFont,
    ) -> None:
        self.converter = converter
        self.font = font

    def _evaluate_candidates_globally(
        self,
        img: Image.Image,
        current_state: list[list[str]],
        line_idx: int,
        candidates: list[list[str]]
    ) -> list[str]:
        bst_candidate = current_state[line_idx]

        baseline_img = render_symbols_img(current_state, self.font)
        bst_fit = img_similarity(img, baseline_img)
        
        test_state = deepcopy(current_state)

        for candidate in candidates:
            test_state[line_idx] = candidate
            rend_test = render_symbols_img(test_state, self.font)
            fit = img_similarity(img, rend_test)
            if fit > bst_fit:
                bst_candidate = candidate
                bst_fit = fit

        return deepcopy(bst_candidate)

    def _initialize_line_streams(
        self,
        img: Image.Image,
        cols: int,
        lines: int
    ) -> tuple[list[Image.Image], list[LineConverter], list[Generator], list[list[str]], list[bool]]:
        img_arr = np.array(img)
        line_arrs = slice_vertically(img_arr, lines)
        lines_img = [Image.fromarray(line_arr) for line_arr in line_arrs]

        line_converters = [
            deepcopy(self.converter)
            for _ in lines_img
        ]

        line_gens = [
            converter.line2symbols_lazy(line, max_cols=cols)
            for converter, line in zip(line_converters, lines_img)
        ]

        latest_image_state: list[list[str]] = [[] for _ in lines_img]
        active_mask = [True] * len(lines_img)

        return lines_img, line_converters, line_gens, latest_image_state, active_mask

    def img2symbols_lazy(
        self,
        img: Image.Image,
        max_cols: int,
        max_lines: int,
        gens_per_step: int = 5,
    ) -> Generator[list[list[str]], None, None]:

        symbols = getattr(self.converter, "symbols", None)
        if not symbols and getattr(self.converter, "tile_converter", None):
            symbols = getattr(self.converter.tile_converter, "symbols", None)
            
        if not symbols:
            symbols = ["M", "g", "@", "l"]

        char_w = get_char_width(symbols, self.font)
        char_h = get_line_height(symbols, self.font)
        
        target_ratio = (img.width / img.height) * (char_h / char_w)
        
        lines_if_max_cols = int(max_cols / target_ratio)
        if lines_if_max_cols <= max_lines:
            cols = max_cols
            lines = max(1, lines_if_max_cols)
        else:
            cols = max(1, int(max_lines * target_ratio))
            lines = max_lines

        lines_img, line_converters, line_gens, bst_image_state, active_mask = self._initialize_line_streams(
            img, cols, lines
        )

        for i, gen in enumerate(line_gens):
            get_artifact_manager().set_line(i, src_line=lines_img[i])
            try:
                bst_image_state[i] = deepcopy(next(gen)) 
            except StopIteration:
                active_mask[i] = False

        yield [list(line_res) for line_res in bst_image_state]

        while any(active_mask):
            for i, (gen, converter) in enumerate(zip(line_gens, line_converters)):
                if not active_mask[i]:
                    continue

                get_artifact_manager().set_line(i, src_line=lines_img[i])

                for _ in range(gens_per_step):
                    try:
                        next(gen)
                    except StopIteration:
                        active_mask[i] = False
                        break
                
                candidates = converter.get_candidates()
                
                eval_candidates = [bst_image_state[i]]
                for c in candidates:
                    if c not in eval_candidates:
                        eval_candidates.append(c)
                
                if eval_candidates and len(eval_candidates) > 1:
                    bst_image_state[i] = self._evaluate_candidates_globally(
                        img, bst_image_state, i, eval_candidates,
                    )

            yield [list(line_res) for line_res in bst_image_state]

    def img2symbols(
        self,
        img: Image.Image,
        max_cols: int,
        max_lines: int,
        gens_per_step: Optional[int] = 5,
    ) -> list[list[str]]:
        final_result: list[list[str]] = []
        for frame in self.img2symbols_lazy(
            img,
            max_cols=max_cols,
            max_lines=max_lines,
            gens_per_step=gens_per_step
        ):
            final_result = frame
        return final_result
