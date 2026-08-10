import numpy as np
from scipy.ndimage import zoom

from converters.tile.tile_converter import TileConverter


BRAILLE_UNICODE_START = 0x2800


class BrailleTileConverter(TileConverter):
    def __init__(self, img_colors: int = 2):
        self.color_threshold = img_colors // 2

    def tile2symbol(self, tile: np.ndarray) -> str:
        target_shape = (4, 2)
        zoom_factors = (target_shape[0] / tile.shape[0],
                        target_shape[1] / tile.shape[1])

        rescaled_tile = zoom(tile, zoom_factors, order=0, grid_mode=True)

        dots = rescaled_tile < self.color_threshold

        offset = int(dots[0, 0])
        offset += int(dots[1, 0]) << 1
        offset += int(dots[2, 0]) << 2

        offset += int(dots[0, 1]) << 3
        offset += int(dots[1, 1]) << 4
        offset += int(dots[2, 1]) << 5

        offset += int(dots[3, 0]) << 6
        offset += int(dots[3, 1]) << 7

        offset ^= 0b11111111

        return chr(BRAILLE_UNICODE_START + offset)
