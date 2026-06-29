import random
from PIL import Image, ImageFont
from converters.line_heuristics.base import LineConverter
from converters.line_heuristics.utils import generate_line_population, insert_into_sorted_population, symbols_id_arr_to_text_arr


class GeneticLineSearch(LineConverter):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont,
            generations: int = 50,
            pop_count: int = 20,
            mutation_rate: float = 0.3,
            mutation_bw: int = 2,
            include_greedy: bool = False) -> None:
        super().__init__(symbols, font)
        self.generations = generations
        self.pop_count = pop_count
        self.mutation_rate = mutation_rate
        self.mutation_bw = mutation_bw
        self.include_greedy = include_greedy

    @staticmethod
    def genetic_mutation(
            el: list[int],
            val_range: int,
            mutation_rate: float = 0.3,
            mutation_bw: int = 2) -> None:
        for i in range(len(el)):
            if random.random() < mutation_rate:
                el[i] = el[i] + random.randint(-mutation_bw, mutation_bw) // 2
                el[i] = el[i] % val_range

    @staticmethod
    def new_genetic_line_population(symbols: list[str],
                                    population: list[list[int]],
                                    mutation_rate: float = 0.3,
                                    mutation_bw: int = 2) -> list[list[int]]:
        el_len = len(population[0])
        cross_point1 = el_len // 3
        cross_point2 = 2 * cross_point1
        new_population: list[list[int]] = []
        i_ps = list(range(len(population)))
        random.shuffle(i_ps)
        while len(i_ps) > 1:
            p1 = population[i_ps.pop()]
            p2 = population[i_ps.pop()]
            new1 = p1[0:cross_point1] + \
                p2[cross_point1:cross_point2] + p1[cross_point2:]
            new2 = p2[0:cross_point1] + \
                p1[cross_point1:cross_point2] + p2[cross_point2:]
            GeneticLineSearch.genetic_mutation(
                new1, len(symbols), mutation_rate, mutation_bw)
            GeneticLineSearch.genetic_mutation(
                new2, len(symbols), mutation_rate, mutation_bw)
            new_population.append(new1)
            new_population.append(new2)
        return new_population

    def line2symbols(self, line: Image.Image) -> list[str]:
        population, fits = generate_line_population(
            line, self.symbols, self.font, self.pop_count, self.include_greedy)
        best_fit = fits[0]
        for gen in range(self.generations):
            new_population = self.new_genetic_line_population(
                self.symbols, population, self.mutation_rate, self.mutation_bw)
            for el in new_population:
                insert_into_sorted_population(
                    population, fits, el, self.symbols, line, self.font)
            population = population[:self.pop_count]
            if fits[0] > best_fit:
                best_fit = fits[0]
                print(gen, best_fit)
        return symbols_id_arr_to_text_arr(population[0], self.symbols)
