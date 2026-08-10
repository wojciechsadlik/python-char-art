from typing import Generator
from PIL import Image, ImageFont
from converters.line_heuristics.line_converter import LineConverter
from converters.line_heuristics.utils import generate_random_line, evaluate_symbol_arr


class RandomLineSearch(LineConverter):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont,
            generations: int = 100) -> None:
        self.symbols = symbols
        self.font = font
        self.generations = generations

    def line2symbols_lazy(self, line: Image.Image, **kwargs) -> Generator[list[str], None, None]:
        best_fit = -float('inf')
        best_symbols: list[str] = []

        for _ in range(self.generations):
            current_symbols = generate_random_line(
                line, self.symbols, self.font)
            current_fit = evaluate_symbol_arr(
                line, [current_symbols], self.font, line.size)

            if current_fit > best_fit:
                best_fit = current_fit
                best_symbols = current_symbols

            yield best_symbols
