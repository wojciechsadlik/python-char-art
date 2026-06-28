from PIL import Image, ImageFont
from base import LineConverter
from utils import generate_random_line, evaluate_text_arr


class RandomLineSearch(LineConverter):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont,
            generations: int = 50) -> None:
        super().__init__(symbols, font)
        self.generations = generations

    def line2symbols(self, line: Image.Image) -> list[str]:
        best_fit = -float('inf')
        best_symbols: list[str] = []

        for _ in range(self.generations):
            current_symbols = generate_random_line(
                line, self.symbols, self.font)
            current_fit = evaluate_text_arr(
                current_symbols, line, self.font)

            if current_fit > best_fit:
                best_fit = current_fit
                best_symbols = current_symbols

        return best_symbols
