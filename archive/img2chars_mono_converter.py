from PIL import Image, ImageFont
import numpy as np
import random
import copy
from typing import Protocol
from palette.symbol2value import (
    make_1d_bin_palette,
    make_2d_bin_palette,
    make_symbol2value_map)
from rendering.ansi_colorizer import AnsiColorizer, reset_code
from image.processing import (
    DITHER_MODES,
    quantize_grayscale,
    img_rgb_to_max_grayscale,
    quantize_rgb)
from modules.img2chars_mono_2d import img2palette_ids as img2palette_ids_2d
from modules.img2chars_mono_nd import find_match


class PickStrategyType(Protocol):
    def __call__(
        self,
        palette_cell: list[str] | None,
        img_win: np.ndarray | None,
        symbol2value: dict[str, np.ndarray] | None) -> str: ...


def pick_random(palette_cell: list[str], *_):
    return palette_cell[random.randrange(len(palette_cell))]


def pick_closest(palette_cell, img_win, symbol2value):
    cell_sym2val = {s: v for s, v in symbol2value.items() if s in palette_cell}
    return find_match(img_win, cell_sym2val)


class Img2MonoCharsConverter:
    def __init__(self,
                 symbols: list[str],
                 font: ImageFont.ImageFont,
                 general_mapping_palette_shape: tuple | None = None,
                 detailed_mapping_win_shape: tuple | None = None,
                 ansi_colorizer: AnsiColorizer | None = None,
                 dither: DITHER_MODES = DITHER_MODES.NONE,
                 normalize_palette: bool = True,
                 pick_strategy: PickStrategyType = pick_random):
        self.ansi_colorizer = ansi_colorizer
        self.use_general_mapping = general_mapping_palette_shape is not None
        self.use_detailed_mapping = detailed_mapping_win_shape is not None
        self.dither = dither
        self.pick_strategy = pick_strategy

        if self.use_general_mapping:
            if (len(general_mapping_palette_shape) == 1):
                self.general_palette_dims = 1
                self.bin_palette, self.bin_vals = make_1d_bin_palette(
                    symbols=symbols,
                    font=font,
                    bins=general_mapping_palette_shape[0],
                    normalize=normalize_palette
                )
            else:
                self.general_palette_dims = 2
                self.bin_palette, self.bin_vals = make_2d_bin_palette(
                    symbols=symbols,
                    font=font,
                    bins=general_mapping_palette_shape,
                    normalize=normalize_palette
                )

        if self.use_detailed_mapping:
            self.detailed_mapping_win_h = detailed_mapping_win_shape[0]
            self.detailed_mapping_win_w = detailed_mapping_win_shape[1]
            self.detailed_mapping = make_symbol2value_map(
                symbols,
                font,
                val_height=self.detailed_mapping_win_h,
                val_width=self.detailed_mapping_win_w,
                normalize=normalize_palette
            )

    def img2palette_ids_1d(self, img: Image.Image):
        if img.mode == "RGB":
            img = img_rgb_to_max_grayscale(img)
        return quantize_grayscale(
            img=img,
            img_colors=len(self.bin_palette),
            dither=self.dither,
            return_palette_map=True,
            palette=self.bin_vals)

    def img2palette_ids_2d(self, img: Image.Image):
        return img2palette_ids_2d(
            img=img,
            bin_palette=self.bin_palette,
            dither=self.dither)

    def palette_ids2char_arr(self,
                             palette_ids,
                             img_br: np.ndarray,
                             img_rgb: np.ndarray = None,
                             colorize=False) -> list[list[str]]:
        char_arr = []
        for y, row in enumerate(palette_ids):
            char_arr.append([])
            for x, palette_idx in enumerate(row):
                y_low = y_high = y
                x_low = x_high = x
                if self.use_detailed_mapping:
                    y_low = y * self.detailed_mapping_win_h
                    y_high = y_low + self.detailed_mapping_win_h
                    x_low = x * self.detailed_mapping_win_w
                    x_high = x_low + self.detailed_mapping_win_w
                elif self.general_palette_dims == 2:
                    y_low, y_high = y * 2, y * 2 + 1

                if self.general_palette_dims == 1:
                    palette_cell = self.bin_palette[palette_idx]
                else:
                    palette_cell = self.bin_palette[
                        palette_idx[0]][palette_idx[1]]

                char = palette_cell[0]
                if (len(palette_cell) > 1):
                    img_br_win = img_br[y_low:y_high, x_low:x_high]
                    if (self.use_detailed_mapping):
                        char = self.pick_strategy(
                            palette_cell, img_br_win, self.detailed_mapping)
                    else:
                        char = pick_random(palette_cell)

                if colorize:
                    if self.use_detailed_mapping:
                        pix_rgb = np.mean(
                            img_rgb[y_low:y_high, x_low:x_high], axis=(0, 1))
                    elif self.general_palette_dims == 2:
                        pix_rgb = (img_rgb[y * 2][x] +
                                   img_rgb[y * 2 + 1][x]) / 2
                    else:
                        pix_rgb = img_rgb[y][x]

                    char = self.ansi_colorizer.create_ansi_prefix(
                        pix_rgb) + char + reset_code()
                char_arr[-1].append(char)

            if colorize:
                char_arr[-1].append(reset_code())

        return char_arr

    def convert(self, img: Image.Image) -> list[list[str]]:
        colorize = self.ansi_colorizer is not None

        if colorize and img.mode != "RGB":
            img = img.convert("RGB")
        elif not colorize:
            img = img.convert("L")

        if colorize and self.ansi_colorizer.use_ansi_256_colors:
            img = quantize_rgb(img, 6, self.dither)

        if self.use_general_mapping:
            if self.general_palette_dims == 1:
                if self.use_detailed_mapping:
                    ds_img = img.resize((
                        img.size[0] // self.detailed_mapping_win_w,
                        img.size[1] // self.detailed_mapping_win_h
                    ))
                else:
                    ds_img = img.resize((
                        img.size[0],
                        img.size[1] // 2
                    ))
                palette_ids_arr = self.img2palette_ids_1d(ds_img)
            elif self.general_palette_dims == 2:
                if self.use_detailed_mapping:
                    ds_img = img.resize((
                        img.size[0] // self.detailed_mapping_win_w,
                        (img.size[1] // self.detailed_mapping_win_h) * 2
                    ))
                else:
                    ds_img = copy.deepcopy(img)
                palette_ids_arr = self.img2palette_ids_2d(ds_img)

            if colorize:
                img_br = np.array(img_rgb_to_max_grayscale(img)) / 255
                img_rgb = np.array(img, dtype=np.float32)

                return self.palette_ids2char_arr(
                    palette_ids_arr, img_br, img_rgb, colorize)

            img_br = np.array(img) / 255
            return self.palette_ids2char_arr(palette_ids_arr, img_br)
