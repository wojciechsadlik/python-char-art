import numpy as np
from PIL import Image
import random
from img_processing import DITHER_MODES, quantize_grayscale
from ascii_palettes import ascii_palette, ascii_brightness


def img2ascii_arr(img: Image.Image, dither=DITHER_MODES.NONE,
                  scale_vertically=True, palette: list[str] = ascii_palette,
                  brightness_palette=ascii_brightness) -> list[list[str]]:
    if scale_vertically:
        img = img.resize((img.size[0], img.size[1]//2))

    for i in range(len(brightness_palette)):
        brightness_palette[i] = brightness_palette[i] / brightness_palette[-1]

    img_arr = quantize_grayscale(img.convert("L"), len(palette), dither, True, brightness_palette)
    
    return img_arr2ascii_arr(img_arr, len(palette), palette)

def img_arr2ascii_arr(img_arr: np.ndarray, img_colors=256, palette: list[str] = ascii_palette) -> list[list[str]]:
    palette_interval = img_colors / len(palette)
    ascii_arr = []
    for i in range(img_arr.shape[0]):
        ascii_arr.append([])
        for j in range(img_arr.shape[1]):
            palette_cell = palette[int(img_arr[i][j]//palette_interval)]
            if (len(palette_cell) > 1):
                ascii_arr[-1].append(palette_cell[random.randint(0, len(palette_cell)-1)])
            else:
                ascii_arr[-1].append(palette_cell)
    return ascii_arr


