import copy
import numpy as np
from PIL import Image, ImageFont
from base import LineConverter
from utils import generate_line_population, evaluate_symbols_id_arr, symbols_id_arr_to_text_arr


class ParticleSwarmLineSearch(LineConverter):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont,
            generations: int = 50,
            pop_count: int = 10,
            innertion: float = 0.5,
            cog_coeff: float = 1.5,
            soc_coeff: float = 1.5,
            include_greedy: bool = False) -> None:
        super().__init__(symbols, font)
        self.generations = generations
        self.pop_count = pop_count
        self.innertion = innertion
        self.cog_coeff = cog_coeff
        self.soc_coeff = soc_coeff
        self.include_greedy = include_greedy

    def line2symbols(self, line: Image.Image) -> list[str]:
        particles, fits = generate_line_population(
            line, self.symbols, self.font, self.pop_count, self.include_greedy)
        particles_np = np.array(particles, dtype=np.float32)
        pos_len = len(particles_np[0])

        best_particle_pos = copy.deepcopy(particles_np)
        best_particle_pos_fit = copy.deepcopy(fits)
        best_global_pos = copy.deepcopy(particles_np[0])
        best_global_pos_fit = fits[0]

        velocities = [2 *
                      len(self.symbols) *
                      np.random.random(pos_len) -
                      len(self.symbols) for _ in range(len(particles_np))]

        for _ in range(self.generations):
            for i in range(len(particles_np)):
                r_p = np.random.random(velocities[i].shape)
                r_g = np.random.random(velocities[i].shape)

                v = self.innertion * velocities[i]
                v += self.cog_coeff * r_p * \
                    (best_particle_pos[i] - particles_np[i])
                v += self.soc_coeff * r_g * (best_global_pos - particles_np[i])

                if np.linalg.norm(v) == 0:
                    v = 2 * len(self.symbols) * \
                        np.random.random(pos_len) - len(self.symbols)
                velocities[i] = v
                particles_np[i] += velocities[i]

                try:
                    fits[i] = evaluate_symbols_id_arr(
                        list(map(int, particles_np[i])), self.symbols, line, self.font)
                except IndexError:
                    fits[i] = 0.0

                if fits[i] > best_particle_pos_fit[i]:
                    best_particle_pos[i] = copy.deepcopy(particles_np[i])
                    best_particle_pos_fit[i] = fits[i]
                    if fits[i] > best_global_pos_fit:
                        best_global_pos = copy.deepcopy(particles_np[i])
                        best_global_pos_fit = fits[i]

        best_p_id_arr = list(map(int, best_global_pos))
        return symbols_id_arr_to_text_arr(best_p_id_arr, self.symbols)
