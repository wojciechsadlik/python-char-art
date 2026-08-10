import numpy as np

from PIL.ImageFont import FreeTypeFont
from scipy.ndimage import zoom

from converters.tile.tile_converter import TileConverter
from palette.symbol2value import make_symbol2value_map


def nearest(tile: np.ndarray, val: np.ndarray):
    dist = np.mean((tile - val) ** 2)
    return np.exp(-dist)


class Tile2SymbNearest(TileConverter):
    def __init__(self,
                 symbols: list[str],
                 font: FreeTypeFont,
                 val_wh=(3, 6),
                 bg_color=0,
                 fg_color=255,
                 metrics=nearest):
        self.symbol2value = make_symbol2value_map(
            symbols=symbols,
            font=font,
            val_width=val_wh[0],
            val_height=val_wh[1],
            bg_color=bg_color,
            fg_color=fg_color,
            normalize=True,
            grayscale=True)
        self.metrics = metrics
        self.val_wh = val_wh

    def tile2symbol(self, tile: np.ndarray) -> str:
        tile = tile / 255
        zoom_factors = (self.val_wh[1] / tile.shape[0],
                        self.val_wh[0] / tile.shape[1])
        tile = zoom(tile, zoom_factors, order=0, grid_mode=True)
        max_sim = 0
        max_sym = ''
        for symbol, val in self.symbol2value.items():
            similarity = self.metrics(tile, val)
            if (similarity > max_sim):
                max_sim = similarity
                max_sym = symbol

        return max_sym
