import numpy as np
from PIL import Image
import random
from img_processing import DITHER_MODES, quantize_grayscale


def img2ascii_arr(img: Image.Image, palette, brightness_palette,
                  dither=DITHER_MODES.NONE, scale_vertically=True) -> list[list[str]]:
    if scale_vertically:
        img = img.resize((img.size[0], img.size[1]//2))

    img_arr = quantize_grayscale(img.convert("L"), len(palette), dither, True, brightness_palette)
    
    return img_arr2ascii_arr(img_arr, palette, len(palette))

def img_arr2ascii_arr(img_arr: np.ndarray, palette: list[str], img_colors=256) -> list[list[str]]:
    palette_interval = img_colors / len(palette)
    char_arr = []
    for i in range(img_arr.shape[0]):
        char_arr.append([])
        for j in range(img_arr.shape[1]):
            palette_cell = palette[int(img_arr[i][j]//palette_interval)]
            if (len(palette_cell) > 1):
                char_arr[-1].append(palette_cell[random.randrange(len(palette_cell))])
            else:
                char_arr[-1].append(palette_cell[0])
    return char_arr


