from abc import ABC, abstractmethod

from PIL import Image, ImageFont

from converters.line_heuristics.utils import split_lines


class LineConverter(ABC):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont) -> None:
        self.symbols: list[str] = symbols
        self.font: ImageFont.FreeTypeFont = font

    @abstractmethod
    def line2symbols(self, line: Image.Image) -> list[str]:
        pass

    def process_image(self, img: Image.Image) -> list[list[str]]:
        lines = split_lines(img, self.symbols, self.font)
        text_arr: list[str] = []
        for line in lines:
            symbols = self.line2symbols(line)
            text_arr.append(symbols)
        return text_arr
