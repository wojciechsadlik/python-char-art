from typing import Generator
from PIL import Image, ImageFont

from converters.line_heuristics.line_converter import LineConverter
from converters.line_heuristics.utils import evaluate_symbol_arr, new_img_draw


def generate_greedy_line(
        line: Image.Image,
        symbols: list[str],
        font: ImageFont.FreeTypeFont) -> list[str]:
    _, text_draw = new_img_draw(line.size)
    text_arr: list[str] = []
    bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)
    while bbox[2] < line.size[0] + 4:
        best_c = symbols[0]
        text_arr.append(symbols[0])
        best_c_fit = evaluate_symbol_arr(line, [text_arr], font, line.size)
        text_arr.pop()
        for i in range(1, len(symbols)):
            text_arr.append(symbols[i])
            fit = evaluate_symbol_arr(line, [text_arr], font, line.size)
            if fit > best_c_fit:
                best_c = symbols[i]
                best_c_fit = fit
            text_arr.pop()
        text_arr.append(best_c)
        bbox = text_draw.textbbox((0, 0), ''.join(text_arr), font=font)

    if text_arr:
        text_arr.pop()

    return text_arr


class GreedyLineSearch(LineConverter):
    def __init__(
        self,
        symbols: list[str],
        font: ImageFont.FreeTypeFont,
    ) -> None:
        super().__init__()
        self.font = font
        self.symbols = symbols

    def line2symbols_lazy(
        self, line: Image.Image, **kwargs
    ) -> Generator[list[str], None, None]:
        yield generate_greedy_line(line, self.symbols, self.font)
