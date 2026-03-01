from collections.abc import Callable
import numpy as np


def slice_vertically(img_arr: np.ndarray, height: int) -> list[np.ndarray]:
    img_height = img_arr.shape[0]
    num_parts = int(img_height / height)
    arr = [img_arr[i * height:(i + 1) * height]
           for i in range(num_parts)]
    if img_height % height:
        height_diff = height - img_height % height
        zero_rows = np.zeros_like(img_arr, shape=(
            height_diff, *img_arr.shape[1:]))
        arr.append(np.vstack([img_arr[num_parts * height:], zero_rows]))
    return arr


def slice_horizontally(img_arr: np.ndarray, width: int) -> list[np.ndarray]:
    img_width = img_arr.shape[1]
    num_parts = int(img_width / width)
    arr = [img_arr[:, i * width:(i + 1) * width]
           for i in range(num_parts)]
    if img_width % width:
        width_diff = width - img_width % width
        zero_rows = np.zeros_like(img_arr, shape=(
            img_arr.shape[0], width_diff, *img_arr.shape[2:]))
        arr.append(np.hstack([img_arr[:, num_parts * width:], zero_rows]))
    return arr


def map_img_arr[T](img_arr: np.ndarray, width: int, height: int,
                   callback: Callable[[np.ndarray], T]) -> list[list[T]]:
    res = []
    for row in slice_vertically(img_arr, height):
        res_row = []
        for slice in slice_horizontally(row, width):
            res_row.append(callback(slice))
        res.append(res_row)
    return res
