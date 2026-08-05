from abc import ABC, abstractmethod
from typing import Generator
from PIL import Image, ImageFont


class LineConverter(ABC):
    def __init__(self, symbols: list[str], font: ImageFont.FreeTypeFont) -> None:
        self.symbols = symbols
        self.font = font

    @abstractmethod
    def line2symbols_lazy(self, line: Image.Image) -> Generator[list[str], None, None]:
        pass

    def line2symbols(self, line: Image.Image) -> list[str]:
        best_symbols: list[str] = []
        for symbols in self.line2symbols_lazy(line):
            best_symbols = symbols
        return best_symbols
