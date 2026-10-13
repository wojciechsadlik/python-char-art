from typing import Generator
from PIL import Image, ImageFont

from converters.line_heuristics.line_converter import LineConverter
from converters.line_heuristics.utils import evaluate_symbol_arr, generate_random_line
from palette.symbol2value import symbols_sorted


def generate_greedy_line(
    line: Image.Image, symbols: list[str],
    font: ImageFont.FreeTypeFont, max_cols: int
) -> list[str]:
    text_arr: list[str] = generate_random_line(symbols, max_cols)

    for i in range(len(text_arr)):
        best_char = text_arr[i]
        best_fit = evaluate_symbol_arr(line, [text_arr], font)

        for sym in symbols:
            if sym == best_char:
                continue

            text_arr[i] = sym
            fit = evaluate_symbol_arr(line, [text_arr], font)

            if fit > best_fit:
                best_fit = fit
                best_char = sym

        text_arr[i] = best_char

    return text_arr


class GreedyLineSearch(LineConverter):
    def __init__(
        self,
        symbols: list[str],
        font: ImageFont.FreeTypeFont,
    ) -> None:
        super().__init__()
        self.font = font
        self.symbols = symbols_sorted(symbols, font)

    def line2symbols_lazy(
        self, line: Image.Image, max_cols: int, **kwargs
    ) -> Generator[list[str], None, None]:
        res = generate_greedy_line(line, self.symbols, self.font, max_cols)
        self.current_candidates = [res]
        yield res
