from typing import Generator, Optional
from PIL import Image
import numpy as np

from image.array_ops import slice_vertically
from converters.line_heuristics.line_converter import LineConverter


class ImgConverter:
    def __init__(
        self,
        converter: LineConverter
    ) -> None:
        self.converter = converter

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

        scale_w = 1.0
        scale_h = 1.0

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

        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    def img2symbols_lazy(
        self,
        img: Image.Image,
        max_cols: Optional[int] = None,
        col_width: Optional[int] = None,
        max_lines: Optional[int] = None,
        line_height: Optional[int] = None,
        gens_per_step: int = 5,
    ) -> Generator[list[list[str]], None, None]:
        img = self.scale_img(img,
                             max_cols=max_cols,
                             col_width=col_width,
                             max_lines=max_lines,
                             line_height=line_height)
        img_arr = np.array(img)

        line_arrs = slice_vertically(img_arr, line_height)
        lines = [Image.fromarray(line_arr) for line_arr in line_arrs]

        line_gens = [
            self.converter.line2symbols_lazy(line, col_width=col_width)
            for line in lines
        ]

        latest_image_state: list[list[str]] = [[] for _ in lines]
        active_mask = [True] * len(lines)

        for i, gen in enumerate(line_gens):
            try:
                latest_image_state[i] = next(gen)
            except StopIteration:
                active_mask[i] = False

        yield [list(line_res) for line_res in latest_image_state]

        while any(active_mask):
            for i, gen in enumerate(line_gens):
                if not active_mask[i]:
                    continue

                for _ in range(gens_per_step):
                    try:
                        latest_image_state[i] = next(gen)
                    except StopIteration:
                        active_mask[i] = False
                        break

            yield [list(line_res) for line_res in latest_image_state]

    def img2symbols(
        self,
        img: Image.Image,
        max_cols: Optional[int] = None,
        col_width: Optional[int] = None,
        max_lines: Optional[int] = None,
        line_height: Optional[int] = None,
    ) -> list[list[str]]:
        final_result: list[list[str]] = []
        for frame in self.img2symbols_lazy(
            img,
            max_cols=max_cols,
            col_width=col_width,
            max_lines=max_lines,
            line_height=line_height,
            gens_per_step=10000,
        ):
            final_result = frame
        return final_result
