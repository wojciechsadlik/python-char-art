import random

import numpy as np
from PIL.ImageFont import FreeTypeFont

from converters.tile.base import TileConverter
from palette.symbol2value import make_symbol2value_map


def make_1d_bin_palette(
        symbols,
        font,
        bins=12,
        bg_color=0,
        fg_color=255,
        normalize=False):

    symbol2brightness = make_symbol2value_map(
        symbols=symbols,
        font=font,
        val_width=1,
        val_height=1,
        bg_color=bg_color,
        fg_color=fg_color,
        grayscale=True,
        normalize=normalize)
    symbol2brightness = {s: b[0][0] for s, b in symbol2brightness.items()}

    symbols = [s for s in symbol2brightness.keys()]

    br_max = max(symbol2brightness.values())
    symbol_bins = [[] for _ in range(bins)]
    br_step = br_max / bins
    for sym in symbols:
        sym_br = symbol2brightness[sym]
        bin_index = min(int(sym_br / br_step), bins - 1)
        symbol_bins[bin_index].append(sym)

    for i in range(1, bins):
        if (len(symbol_bins[i]) == 0):
            max_last = max(symbol_bins[i - 1],
                           key=lambda s: symbol2brightness[s])
            symbol_bins[i].append(max_last)

    mean_bin_brs = []
    for bin in symbol_bins:
        mean_bin_brs.append(
            np.mean([symbol2brightness[s] for s in bin]))

    return symbol_bins, mean_bin_brs


class Tile2Symb1dBin(TileConverter):
    def __init__(self,
                 symbols: list[str],
                 font: FreeTypeFont,
                 bins=12,
                 bg_color=0,
                 fg_color=255,
                 img_colors=256):
        self.bin_palette, _ = make_1d_bin_palette(
            symbols=symbols,
            font=font,
            bins=bins,
            bg_color=bg_color,
            fg_color=fg_color,
            normalize=True)
        self.palette_interval = img_colors / len(self.bin_palette)

    def tile2symbol(self, tile: np.ndarray) -> str:
        tile_val = tile.mean()
        palette_cell = self.bin_palette[int(tile_val / self.palette_interval)]
        symbol = palette_cell[0]
        if (len(palette_cell) > 1):
            symbol = palette_cell[random.randrange(len(palette_cell))]
        return symbol
