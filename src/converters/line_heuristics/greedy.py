from typing import Generator
from PIL import Image
from converters.line_heuristics.line_converter import LineConverter
from converters.line_heuristics.utils import generate_greedy_line


class GreedyLineSearch(LineConverter):
    def __init__(self, symbols, font):
        self.symbols = symbols
        self.font = font

    def line2symbols_lazy(self, line: Image.Image, **kwargs) -> Generator[list[str], None, None]:
        yield generate_greedy_line(line, self.symbols, self.font)
