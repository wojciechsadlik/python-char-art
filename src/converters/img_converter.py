from copy import deepcopy
import logging
from typing import Generator, Optional
from PIL import Image, ImageFont
import numpy as np

from diagnostics.artifact_manager import get_artifact_manager
from image.array_ops import slice_vertically
from converters.line_heuristics.line_converter import LineConverter

from converters.line_heuristics.utils import symbols_id_arr_to_text_arr
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

    def scale_img(
        self,
        img: Image.Image,
        max_cols: Optional[int] = None,
        col_width: Optional[int] = None,
        max_lines: Optional[int] = None,
        line_height: Optional[int] = None,
    ) -> Image.Image:
        if max_cols is None and max_lines is None:
            return img

        scale_w, scale_h = 1.0, 1.0

        if col_width is None and line_height is not None:
            col_width = line_height // 2
        if line_height is None and col_width is not None:
            line_height = col_width * 2

        if max_cols is not None:
            scale_w = (max_cols * col_width) / img.width

        if max_lines is not None:
            scale_h = (max_lines * line_height) / img.height

        scale_factor = min(scale_w, scale_h)

        new_w = int(img.width * scale_factor)
        new_h = int(img.height * scale_factor)

        logger.info(
            "Rescaling image from %s to %s (scale factor: %.3f)",
            img.size, (new_w, new_h), scale_factor
        )

        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    def _evaluate_candidates_globally(
        self,
        img: Image.Image,
        current_state: list[list[str]],
        line_idx: int,
        candidates: list[list[str]]
    ) -> list[str]:
        bst_candidate = current_state[line_idx]
        bst_fit = float("-inf")
        test_state = deepcopy(current_state)

        for candidate in candidates:
            test_state[line_idx] = candidate
            rend_test = render_symbols_img(test_state, self.font, wh=img.size)
            fit = img_similarity(img, rend_test)
            if fit > bst_fit:
                bst_candidate = candidate
                bst_fit = fit

        return bst_candidate

    def _initialize_line_streams(
        self,
        img: Image.Image,
        col_width: Optional[int],
        line_height: Optional[int]
    ) -> tuple[list[Image.Image], list[LineConverter], list[Generator], list[list[str]], list[bool]]:
        img_arr = np.array(img)
        line_arrs = slice_vertically(img_arr, line_height)
        lines = [Image.fromarray(line_arr) for line_arr in line_arrs]

        line_converters = [
            deepcopy(self.converter)
            for _ in lines
        ]

        line_gens = [
            converter.line2symbols_lazy(line, col_width=col_width)
            for converter, line in zip(line_converters, lines)
        ]

        latest_image_state: list[list[str]] = [[] for _ in lines]
        active_mask = [True] * len(lines)

        return lines, line_converters, line_gens, latest_image_state, active_mask

    def img2symbols_lazy(
        self,
        img: Image.Image,
        max_cols: Optional[int] = None,
        col_width: Optional[int] = None,
        max_lines: Optional[int] = None,
        line_height: Optional[int] = None,
        gens_per_step: int = 5,
    ) -> Generator[list[list[str]], None, None]:
        logger.info(
            "Scaling and slicing image. Constraints: max_cols=%s, col_width=%s, max_lines=%s, line_height=%s",
            max_cols, col_width, max_lines, line_height
        )

        img = self.scale_img(
            img,
            max_cols=max_cols,
            col_width=col_width,
            max_lines=max_lines,
            line_height=line_height
        )

        lines, line_converters, line_gens, latest_image_state, active_mask = self._initialize_line_streams(
            img, col_width, line_height
        )

        for i, gen in enumerate(line_gens):
            get_artifact_manager().set_line(i, src_line=lines[i])
            try:
                latest_image_state[i] = next(gen)
            except StopIteration:
                active_mask[i] = False

        yield [list(line_res) for line_res in latest_image_state]

        while any(active_mask):
            for i, (gen, converter) in enumerate(zip(line_gens, line_converters)):
                if not active_mask[i]:
                    continue

                get_artifact_manager().set_line(i, src_line=lines[i])

                for _ in range(gens_per_step):
                    try:
                        next(gen)
                    except StopIteration:
                        active_mask[i] = False
                        break

                prev_best = latest_image_state[i]
                
                candidates = converter.get_candidates()
                
                eval_candidates = [prev_best] if prev_best else []
                for c in candidates:
                    if c not in eval_candidates:
                        eval_candidates.append(c)
                
                if eval_candidates and len(eval_candidates) > 1:
                    latest_image_state[i] = self._evaluate_candidates_globally(
                        img, latest_image_state, i, eval_candidates
                    )
                elif eval_candidates:
                    latest_image_state[i] = eval_candidates[0]

            yield [list(line_res) for line_res in latest_image_state]

    def img2symbols(
        self,
        img: Image.Image,
        max_cols: Optional[int] = None,
        col_width: Optional[int] = None,
        max_lines: Optional[int] = None,
        line_height: Optional[int] = None,
        gens_per_step: Optional[int] = 5,
    ) -> list[list[str]]:
        final_result: list[list[str]] = []
        for frame in self.img2symbols_lazy(
            img,
            max_cols=max_cols,
            col_width=col_width,
            max_lines=max_lines,
            line_height=line_height,
            gens_per_step=gens_per_step
        ):
            final_result = frame
        return final_result
