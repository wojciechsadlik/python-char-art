import random
from typing import Generator, Optional
from PIL import Image, ImageFont

from converters.line_heuristics.line_converter import LineConverter
from converters.line_heuristics.utils import (
    generate_line_population,
    insert_into_sorted_population,
    symbols_id_arr_to_text_arr,
)
from converters.tile.tile_converter import TileConverter
from palette.symbol2value import symbols_sorted


class HarmonyLineSearch(LineConverter):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont,
            generations: int = 50,
            pop_count: int = 100,
            mem_rate: float = 0.8,
            pa_rate: float = 0.3,
            pa: Optional[int] = None,
            include_greedy: bool = False,
            tile_converter: Optional[TileConverter] = None) -> None:
        super().__init__(tile_converter=tile_converter)
        self.symbols = symbols_sorted(symbols, font)
        self.font = font
        self.generations = generations
        self.pop_count = pop_count
        self.mem_rate = mem_rate
        self.pa_rate = pa_rate
        self.pa = pa
        if pa is None:
            self.pa = max(len(symbols) // 8, 2)
        self.include_greedy = include_greedy

    @staticmethod
    def new_harmony_line(symbols: list[str],
                         population: list[list[int]],
                         mem_rate: float = 0.8,
                         pa_rate: float = 0.3,
                         pa: int = 2) -> list[int]:
        new_harm: list[int] = []
        while len(new_harm) < len(population[0]):
            if random.random() < mem_rate:
                new_pitch = random.choices(population, weights=range(
                    len(population), 0, -1))[0][len(new_harm)]
                if random.random() < pa_rate:
                    new_pitch += random.randint(-pa, pa)
                    new_pitch = new_pitch % len(symbols)
                new_harm.append(new_pitch)
            else:
                new_harm.append(random.randrange(0, len(symbols)))
        return new_harm

    def line2symbols_lazy(self, line: Image.Image, col_width: Optional[int] = None, **kwargs) -> Generator[list[str], None, None]:
        population, fits = generate_line_population(
            line,
            self.symbols,
            self.font,
            self.pop_count,
            include_greedy=self.include_greedy,
            tile_converter=self.tile_converter,
            col_width=col_width,
        )

        yield symbols_id_arr_to_text_arr(population[0], self.symbols)

        for _ in range(self.generations):
            for _ in range(self.pop_count):
                new_harm = self.new_harmony_line(
                    self.symbols, population, self.mem_rate, self.pa_rate, self.pa)
                insert_into_sorted_population(
                    population, fits, new_harm, self.symbols, line, self.font)

            yield symbols_id_arr_to_text_arr(population[0], self.symbols)
