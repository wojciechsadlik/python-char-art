from PIL import Image
import numpy as np
from img_processing import DITHER_MODES, quantize_grayscale

BRAILLE_UNICODE_START = 0x2800


def get_braille_chars():
    braille_unicode_start = BRAILLE_UNICODE_START
    braille_unicode_end = 0x28FF

    chars = []
    for c in range(braille_unicode_start, braille_unicode_end + 1):
        chars.append(chr(c))
    return chars


def img2braille_arr(img: Image.Image,
                    dither=DITHER_MODES.NONE) -> list[list[str]]:
    img_arr = quantize_grayscale(img.convert("L"), 2, dither, True)
    return img_arr2braille_arr(img_arr, img_colors=2)


def img_arr2braille_arr(img_arr: np.ndarray,
                        img_colors=256) -> list[list[str]]:
    color_threshold = img_colors // 2
    braille_arr = []
    for y in range(3, img_arr.shape[0], 4):
        braille_arr.append([])
        for x in range(1, img_arr.shape[1], 2):
            braille_unicode_offset = int(
                img_arr[y - 3][x - 1] < color_threshold)
            braille_unicode_offset += int(img_arr[y - 2]
                                          [x - 1] < color_threshold) << 1
            braille_unicode_offset += int(img_arr[y - 1]
                                          [x - 1] < color_threshold) << 2
            braille_unicode_offset += int(img_arr[y - 3]
                                          [x] < color_threshold) << 3
            braille_unicode_offset += int(img_arr[y - 2]
                                          [x] < color_threshold) << 4
            braille_unicode_offset += int(img_arr[y - 1]
                                          [x] < color_threshold) << 5

            braille_unicode_offset += int(img_arr[y]
                                          [x - 1] < color_threshold) << 6
            braille_unicode_offset += int(img_arr[y][x] < color_threshold) << 7

            braille_unicode_offset ^= 0b011111111
            braille_arr[-1].append(chr(BRAILLE_UNICODE_START +
                                   braille_unicode_offset))

    return braille_arr
