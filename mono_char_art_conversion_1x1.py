import numpy as np
from PIL import Image
import random
from img_processing import DITHER_MODES, quantize_grayscale, quantize_rgb
from ansi_colors_parser import *

class Ansi256ColorizeSettings:
    def __init__(self, colored_fg, colored_bg, fg_brightness_scale, bg_brightness_scale):
        self.colored_fg = colored_fg
        self.colored_bg = colored_bg
        self.fg_brightness_scale = fg_brightness_scale
        self.bg_brightness_scale = bg_brightness_scale


def img2char_arr_1x1(img: Image.Image, palette, brightness_palette,
                  dither=DITHER_MODES.NONE, scale_vertically=True, colorize_settings: Ansi256ColorizeSettings=None) -> list[list[str]]:
    if scale_vertically:
        img = img.resize((img.size[0], img.size[1]//2))

    img_brightness_mapping_arr = quantize_grayscale(img.convert("L"), len(palette), dither, True, brightness_palette)
    if (colorize_settings is None or img.mode != "RGB"):
        return img_arr2char_arr_1x1(img_brightness_mapping_arr, palette, len(palette))
    
    img_rgb = quantize_rgb(img, 6, dither)
    img_rgb_arr = np.array(img_rgb)
    return img_arr2char_arr_1x1(img_brightness_mapping_arr, palette, len(palette), img_rgb_arr, colorize_settings)


def img_arr2char_arr_1x1(img_arr: np.ndarray, palette: list[str], img_shades=256, img_rgb_arr=None, colorize_settings: Ansi256ColorizeSettings=None) -> list[list[str]]:
    palette_interval = img_shades / len(palette)
    char_arr = []
    for i in range(img_arr.shape[0]):
        char_arr.append([])
        for j in range(img_arr.shape[1]):
            palette_cell = palette[int(img_arr[i][j]//palette_interval)]
            if (len(palette_cell) > 1):
                char_arr[-1].append(palette_cell[random.randrange(len(palette_cell))])
            else:
                char_arr[-1].append(palette_cell[0])
            if img_rgb_arr is not None and colorize_settings is not None:
                if colorize_settings.colored_fg:
                    rgb_pix = img_rgb_arr[i][j]
                    rgb_pix = colorize_settings.fg_brightness_scale * rgb_pix
                    rgb_pix = np.clip(rgb_pix, 0.0, 255.0)
                    char_arr[-1][-1] = (set_char_fg_color_code(rgb_to_ansi_256_id(rgb_pix[0], rgb_pix[1], rgb_pix[2]))
                                        + char_arr[-1][-1])
                if colorize_settings.colored_bg:
                    rgb_pix_bg = img_rgb_arr[i][j]
                    rgb_pix_bg = colorize_settings.bg_brightness_scale * rgb_pix_bg
                    rgb_pix_bg = np.clip(rgb_pix_bg, 0.0, 255.0)
                    char_arr[-1][-1] = (set_char_bg_color_code(rgb_to_ansi_256_id(rgb_pix_bg[0], rgb_pix_bg[1], rgb_pix_bg[2]))
                                        + char_arr[-1][-1])
    return char_arr


