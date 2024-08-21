from PIL import Image, ImageFont
import numpy as np
import random
import copy
from typing import Callable
from generate_char_palette import (generate_1_1_palette,
    generate_1_2_palette, generate_brightness_map)
from ansi_colorizer import AnsiColorizer
from img_processing import DITHER_MODES, quantize_grayscale
from mono_char_art_conversion_1x2 import quantize_grayscale_1x2




class MonoCharArtConverter:
    def detailed_search(self, chars: list[str], win: np.ndarray) -> str:
        max_dist = np.linalg.norm(np.ones(win.shape))
        max_sim = 0
        max_char = ''
        for char, br in self.detailed_mapping.items():
            if char not in chars:
                continue

            sim = 1 - np.linalg.norm(win - br) / max_dist

            if (sim > max_sim):
                max_sim = sim
                max_char = char
        return max_char


    def __init__(self,
                 char_set: list[str],
                 font: ImageFont.ImageFont,
                 general_mapping_palette_shape: tuple | None = None,
                 detailed_mapping_wh: tuple | None = None,
                 colorize_settings: AnsiColorizer | None = None,
                 dither: DITHER_MODES = DITHER_MODES.NONE
                ):
        self.colorize_settings = colorize_settings
        self.use_general_mapping = general_mapping_palette_shape is not None
        self.use_detailed_mapping = detailed_mapping_wh is not None
        self.dither = dither

        if self.use_general_mapping:
            if (len(general_mapping_palette_shape) == 1):
                self.general_palette_dims = 1
                self.palette, self.palette_br = generate_1_1_palette(
                    char_set,
                    font,
                    general_mapping_palette_shape[0],
                    normalize=True
                )
            else:
                self.general_palette_dims = 2
                self.palette, self.palette_br = generate_1_2_palette(
                    char_set,
                    font,
                    (
                        general_mapping_palette_shape[0],
                        general_mapping_palette_shape[1],
                    ),
                    normalize=True
                )
        
        if self.use_detailed_mapping:
            self.detailed_mapping_wh = detailed_mapping_wh
            self.detailed_mapping = generate_brightness_map(
                char_set,
                font,
                detailed_mapping_wh,
                normalize=True
            )


    def pix2palette_id_mapping(self, img: Image.Image) -> np.ndarray:
        if (self.colorize_settings is None or img.mode != "RGB"):
            return quantize_grayscale(
                img.convert("L"),
                len(self.palette),
                self.dither,
                True,
                self.palette_br
            )
        else:
            raise Exception("TODO")
        pass


    def two_pix2palette_id_mapping(self, img: Image.Image) -> np.ndarray:
        if (self.colorize_settings is None or img.mode != "RGB"):
            return quantize_grayscale_1x2(
                img.convert("L"),
                (len(self.palette), len(self.palette[0])),
                self.dither,
                True,
                np.array(self.palette_br)
            )
        else:
            raise Exception("TODO")


    def palette_id_mapping2char_arr(self, img_mapped: np.ndarray, img: np.ndarray) -> list[list[str]]:
        char_arr = []
        for y in range(img_mapped.shape[0]):
            char_arr.append([])
            for x in range(img_mapped.shape[1]):
                palette_id = img_mapped[y][x]
                if self.general_palette_dims == 1:
                    palette_cell = self.palette[palette_id]
                else:
                    palette_cell = self.palette[palette_id[0]][palette_id[1]]
                
                if (len(palette_cell) > 1):
                    if self.use_detailed_mapping:
                        win_y_range = (y*self.detailed_mapping_wh[1],
                                       (y+1)*self.detailed_mapping_wh[1])
                        win_x_range = (x*self.detailed_mapping_wh[0],
                                       (x+1)*self.detailed_mapping_wh[0])

                        win = img[win_y_range[0]:win_y_range[1],
                                  win_x_range[0]:win_x_range[1]]
                        char = self.detailed_search(palette_cell, win)
                    else:
                        char = palette_cell[random.randrange(0, len(palette_cell))]
                else:
                    char = palette_cell[0]

                char_arr[-1].append(char)

        return char_arr


    def convert(self, img: Image.Image) -> list[list[str]]:
        if self.use_general_mapping:
            if self.general_palette_dims == 1:
                if self.use_detailed_mapping:
                    proc_img = img.resize((
                        img.size[0] // self.detailed_mapping_wh[0],
                        img.size[1] // self.detailed_mapping_wh[1]
                    ))
                else:
                    proc_img = img.resize((
                        img.size[0],
                        img.size[1] // 2
                    ))
                img_mapped = self.pix2palette_id_mapping(proc_img)
            elif self.general_palette_dims == 2:
                if self.use_detailed_mapping:
                    proc_img = img.resize((
                        img.size[0] // self.detailed_mapping_wh[0],
                        (img.size[1] // self.detailed_mapping_wh[1]) * 2
                    ))
                else:
                    proc_img = copy.copy(img)
                img_mapped = self.two_pix2palette_id_mapping(proc_img)

            img = np.array(img) / 255
            return self.palette_id_mapping2char_arr(img_mapped, img)

        return []