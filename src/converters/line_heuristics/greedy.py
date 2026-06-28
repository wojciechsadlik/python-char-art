from PIL import Image
from converters.line_heuristics.base import LineConverter
from converters.line_heuristics.utils import generate_greedy_line


class GreedyLineSearch(LineConverter):
    def line2symbols(self, line: Image.Image) -> list[str]:
        return generate_greedy_line(line, self.symbols, self.font)
