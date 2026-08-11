from typing import Generator, Optional
import numpy as np
from PIL import Image

from converters.tile.tile_converter import TileConverter
from image.array_ops import slice_horizontally


class LineConverter:
    def __init__(
        self,
        tile_converter: Optional[TileConverter] = None
    ) -> None:
        self.tile_converter = tile_converter

    def tile_line(self,
                  line: Image.Image,
                  col_width: Optional[int] = None
                  ) -> list[str]:
        if self.tile_converter is None:
            raise ValueError("No TileConverter assigned to LineConverter.")
        col_width = col_width or line.height // 2
        line_arr = np.array(line)
        tile_arrs = slice_horizontally(line_arr, col_width)
        return [self.tile_converter.tile2symbol(tile) for tile in tile_arrs]

    def line2symbols_lazy(self,
                          line: Image.Image,
                          col_width: Optional[int] = None
                          ) -> Generator[list[str], None, None]:
        if self.tile_converter is not None:
            yield self.tile_line(line, col_width=col_width)
        else:
            raise NotImplementedError(
                "override or provide a tile_converter"
            )

    def line2symbols(self, 
                     line: Image.Image,
                     col_width: Optional[int] = None
                     ) -> list[str]:
        best_symbols: list[str] = []
        for symbols in self.line2symbols_lazy(line, col_width=col_width):
            best_symbols = symbols
        return best_symbols
