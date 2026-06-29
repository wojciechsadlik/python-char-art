import random
from PIL import Image, ImageFont
from converters.line_heuristics.base import LineConverter
from converters.line_heuristics.utils import generate_line_population, insert_into_sorted_population, symbols_id_arr_to_text_arr


class HarmonyLineSearch(LineConverter):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont,
            generations: int = 50,
            pop_count: int = 100,
            mem_rate: float = 0.8,
            pa_rate: float = 0.3,
            pa: int = 2,
            include_greedy: bool = False) -> None:
        super().__init__(symbols, font)
        self.generations = generations
        self.pop_count = pop_count
        self.mem_rate = mem_rate
        self.pa_rate = pa_rate
        self.pa = pa
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
                new_pitch = random.choices(population, weights=range(len(population), 0, -1))[0][len(new_harm)]
                if random.random() < pa_rate:
                    new_pitch += random.randint(-pa, pa)
                    new_pitch = new_pitch % len(symbols)
                new_harm.append(new_pitch)
            else:
                new_harm.append(random.randrange(0, len(symbols)))
        return new_harm

    def line2symbols(self, line: Image.Image) -> list[str]:
        population, fits = generate_line_population(
            line, self.symbols, self.font, self.pop_count, self.include_greedy)
        best_fit = fits[0]
        print(-1, best_fit)
        for gen in range(self.generations):
            for _ in range(self.pop_count):
                new_harm = self.new_harmony_line(
                    self.symbols, population, self.mem_rate, self.pa_rate, self.pa)
                insert_into_sorted_population(
                    population, fits, new_harm, self.symbols, line, self.font)
            if fits[0] > best_fit:
                best_fit = fits[0]
                print(gen, best_fit)

        return symbols_id_arr_to_text_arr(population[0], self.symbols)
