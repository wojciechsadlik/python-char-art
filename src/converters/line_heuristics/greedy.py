from typing import Generator
from PIL import Image
from converters.line_heuristics.base import LineConverter
from converters.line_heuristics.utils import generate_greedy_line


class GreedyLineSearch(LineConverter):
    def line2symbols_lazy(self, line: Image.Image) -> Generator[list[str], None, None]:
        yield generate_greedy_line(line, self.symbols, self.font)
