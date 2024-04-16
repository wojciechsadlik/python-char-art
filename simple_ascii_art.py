import numpy as np
from PIL import Image
import random
from img_processing import DITHER_MODES, quantize_grayscale, quantize_grayscale_v2
from ascii_palettes import ascii_palette, ascii_palette_v2, ascii_brightness, ascii_brightness_v2


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


def img2ascii_arr_v2(img: Image.Image, palette: list[list[str]] = ascii_palette_v2,
                     dither=DITHER_MODES.NONE, brightness_palette=ascii_brightness_v2) -> list[list[str]]:

    for i in range(len(brightness_palette)):
        for j in range(len(brightness_palette[i])):
            brightness_palette[i][j] = (
                brightness_palette[i][j][0] / brightness_palette[-1][-1][0],
                brightness_palette[i][j][1] / brightness_palette[-1][-1][1]
            )
    
    img_arr = quantize_grayscale_v2(img.convert("L"), (len(palette), len(palette[0])), dither, True, np.array(brightness_palette))

    return img_arr2ascii_arr_v2(img_arr, (len(palette), len(palette[0])), palette)

def img_arr2ascii_arr_v2(img_arr: np.ndarray, img_colors=(256,256), palette: list[list[str]] = ascii_palette_v2) -> list[list[str]]:
    palette_y_interval = img_colors[0] / len(palette)
    palette_x_interval = img_colors[1] / len(palette[0])
    ascii_arr = []
    for y in range(1, img_arr.shape[0], 2):
        ascii_arr.append([])
        for x in range(img_arr.shape[1]):
            palette_cell = palette[int(img_arr[y-1][x]//palette_y_interval)][int(img_arr[y][x]//palette_x_interval)]
            if (len(palette_cell) > 1):
                ascii_arr[-1].append(palette_cell[random.randint(0, len(palette_cell)-1)])
            else:
                ascii_arr[-1].append(palette_cell)
    return ascii_arr