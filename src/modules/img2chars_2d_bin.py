import numpy as np
from PIL.ImageFont import FreeTypeFont
import random
from modules.palette_generation import make_symbol2value_map


def make_2d_bin_palette(
        symbols,
        font,
        bins=(9, 9),
        bg_color=0,
        fg_color=255,
        normalize=False):

    symbol2brightness = make_symbol2value_map(
        symbols=symbols,
        font=font,
        val_width=1,
        val_height=2,
        bg_color=bg_color,
        fg_color=fg_color,
        grayscale=True,
        normalize=normalize)

    x_bins = bins[0]
    y_bins = bins[1]

    symbol_bins = [[[] for _ in range(x_bins)] for _ in range(y_bins)]
    br_max = np.max(np.array(list(symbol2brightness.values())), axis=0)
    br_y_step = br_max[0][0] / y_bins
    br_x_step = br_max[1][0] / x_bins
    for sym, sym_br in symbol2brightness.items():
        bin_y_index = min(int(sym_br[0][0] / br_y_step), y_bins - 1)
        bin_x_index = min(int(sym_br[1][0] / br_x_step), x_bins - 1)
        symbol_bins[bin_y_index][bin_x_index].append(sym)

    def splice_cell(top, left):
        if not top:
            return max(left, key=lambda s: symbol2brightness[s][1])
        if not left:
            return max(top, key=lambda s: symbol2brightness[s][0])
        res = top[0]
        res_mean = np.mean(symbol2brightness[res])
        for s in top:
            s_mean = np.mean(symbol2brightness[s])
            if (res_mean < s_mean):
                res = s
                res_mean = s_mean
        for s in left:
            s_mean = np.mean(symbol2brightness[s])
            if (res_mean < s_mean):
                res = s
                res_mean = s_mean
        return res

    for y in range(y_bins):
        for x in range(x_bins):
            if len(symbol_bins[y][x]) == 0:
                if x > 0 and y > 0:
                    symbol_bins[y][x].append(splice_cell(
                        symbol_bins[y - 1][x], symbol_bins[y][x - 1]))
                elif x > 0:
                    symbol_bins[y][x].append(
                        splice_cell([], symbol_bins[y][x - 1]))
                elif y > 0:
                    symbol_bins[y][x].append(
                        splice_cell(symbol_bins[y - 1][x], []))

    mean_bin_brs = []
    for row in symbol_bins:
        mean_bin_brs.append([])
        for bin in row:
            mean_br = np.mean([symbol2brightness[s] for s in bin], axis=0)
            mean_bin_brs[-1].append(mean_br)

    return symbol_bins, mean_bin_brs


class Img2Chars2dBin:
    def __init__(self,
                 symbols: list[str],
                 font: FreeTypeFont,
                 bins=(9, 9),
                 bg_color=0,
                 fg_color=255,
                 img_colors=256):
        self.bin_palette, _ = make_2d_bin_palette(
            symbols=symbols,
            font=font,
            bins=bins,
            bg_color=bg_color,
            fg_color=fg_color,
            normalize=True)
        self.palette_interval = (
            img_colors / bins[0],
            img_colors / bins[1]
        )

    def tile2symbol(self, tile: np.ndarray) -> str:
        tile_top_val = tile[:tile.shape[0] // 2].mean()
        tile_btm_val = tile[tile.shape[0] // 2:].mean()
        y_palette_idx = int(tile_top_val / self.palette_interval[0])
        x_palette_idx = int(tile_btm_val / self.palette_interval[1])
        palette_cell = self.bin_palette[y_palette_idx][x_palette_idx]
        symbol = palette_cell[0]
        if (len(palette_cell) > 1):
            symbol = palette_cell[random.randrange(len(palette_cell))]
        return symbol
