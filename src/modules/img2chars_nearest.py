import numpy as np
from PIL.ImageFont import FreeTypeFont
from modules.palette_generation import make_symbol2value_map


def nearest(tile: np.ndarray, val: np.ndarray):
    max_dist = np.linalg.norm(np.ones(tile.shape))
    return 1 - np.linalg.norm(tile - val) / max_dist


class Img2CharsNearest:
    def __init__(self,
                 symbols: list[str],
                 font: FreeTypeFont,
                 wh=(3, 6),
                 bg_color=0,
                 fg_color=255,
                 metrics=nearest):
        self.symbol2value = make_symbol2value_map(
            symbols=symbols,
            font=font,
            val_width=wh[0],
            val_height=wh[1],
            bg_color=bg_color,
            fg_color=fg_color,
            normalize=True,
            grayscale=True)
        self.metrics = metrics

    def tile2symbol(self, tile: np.ndarray) -> str:
        max_sim = 0
        max_sym = ''
        for char, val in self.symbol2value.items():
            similarity = self.metrics(tile, val)
            if (similarity > max_sim):
                max_sim = similarity
                max_sym = char

        return max_sym
