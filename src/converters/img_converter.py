import math
from typing import Union, Generator, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from image.array_ops import map_img_arr, slice_vertically
from converters.tile.base import TileConverter
from converters.line_heuristics.base import LineConverter


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


class ImgConverter:
    def __init__(self, converter: Union[LineConverter, TileConverter]) -> None:
        self.converter = converter

    def scale_img(
        self,
        img: Image.Image,
        win_wh: Optional[tuple[int, int]] = (1, 2),
        max_cols: Optional[int] = None,
        max_lines: Optional[int] = None,
    ) -> Image.Image:
        if max_cols is None and max_lines is None:
            return img

        scale_w = 1.0
        scale_h = 1.0

        if isinstance(self.converter, LineConverter):
            if max_cols is not None:
                font_width_px = get_char_width(self.converter.font)
                scale_w = (max_cols * font_width_px) / img.width

            if max_lines is not None:
                line_height_px = get_line_height(self.converter.symbols,
                                                 self.converter.font)
                scale_h = (max_lines * line_height_px) / img.height

        elif isinstance(self.converter, TileConverter):
            win_w = win_wh[0]
            win_h = win_wh[1]
            if max_cols is not None:
                scale_w = (max_cols * win_w) / img.width
            if max_lines is not None:
                scale_h = (max_lines * win_h) / img.height

        scale_factor = min(scale_w, scale_h)

        new_w = int(img.width * scale_factor)
        new_h = int(img.height * scale_factor)

        return img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    def img2symbols_lazy(
        self,
        img: Image.Image,
        win_wh: Optional[tuple[int, int]] = (1, 2),
        max_cols: Optional[int] = None,
        max_lines: Optional[int] = None,
        gens_per_step: int = 5,
    ) -> Generator[list[list[str]], None, None]:
        img = self.scale_img(
            img, win_wh=win_wh, max_cols=max_cols, max_lines=max_lines
        )
        img_arr = np.array(img)

        if isinstance(self.converter, TileConverter):
            symbols_grid = map_img_arr(
                img_arr,
                width=win_wh[0],
                height=win_wh[1],
                callback=self.converter.tile2symbol,
            )
            yield symbols_grid
            return

        if isinstance(self.converter, LineConverter):
            line_height = get_line_height(
                self.converter.symbols, self.converter.font
            )

            line_arrs = slice_vertically(img_arr, line_height)
            lines = [Image.fromarray(line_arr) for line_arr in line_arrs]

            line_gens = [
                self.converter.line2symbols_lazy(line) for line in lines
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
        win_wh: Optional[tuple[int, int]] = (1, 2),
        max_cols: Optional[int] = None,
        max_lines: Optional[int] = None,
    ) -> list[list[str]]:
        final_result: list[list[str]] = []
        for frame in self.img2symbols_lazy(img,
                                           win_wh=win_wh,
                                           max_cols=max_cols,
                                           max_lines=max_lines,
                                           gens_per_step=10000):
            final_result = frame
        return final_result
