from collections.abc import Callable
import numpy as np


def slice_vertically(img_arr: np.ndarray, height: int) -> list[np.ndarray]:
    img_height = img_arr.shape[0]

    if (remainder := img_height % height) != 0:
        pad_h = height - remainder
        pad_shape = [(0, pad_h)] + [(0, 0)] * (img_arr.ndim - 1)
        img_arr = np.pad(
            img_arr,
            pad_shape,
            constant_values=0)

    return [img_arr[y: y + height] for y in range(0, img_arr.shape[0], height)]


def slice_horizontally(img_arr: np.ndarray, width: int) -> list[np.ndarray]:
    img_width = img_arr.shape[1]

    if (remainder := img_width % width) != 0:
        pad_w = width - remainder
        pad_shape = [(0, 0), (0, pad_w)] + [(0, 0)] * (img_arr.ndim - 2)
        img_arr = np.pad(
            img_arr,
            pad_shape,
            constant_values=0)

    return [img_arr[:, x: x + width]
            for x in range(0, img_arr.shape[1], width)]


def map_img_arr[T](img_arr: np.ndarray, width: int, height: int,
                   callback: Callable[[np.ndarray], T]) -> list[list[T]]:
    res = []
    for row in slice_vertically(img_arr, height):
        res_row = [callback(tile) for tile in slice_horizontally(row, width)]
        res.append(res_row)
    return res
