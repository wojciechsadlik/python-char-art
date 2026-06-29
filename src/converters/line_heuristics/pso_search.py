import copy
import numpy as np
from PIL import Image, ImageFont
from converters.line_heuristics.base import LineConverter
from converters.line_heuristics.utils import generate_line_population, evaluate_symbols_id_arr, symbols_id_arr_to_text_arr


class ParticleSwarmLineSearch(LineConverter):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont,
            generations: int = 50,
            pop_count: int = 10,
            innertion: float = 0.9,
            cog_coeff: float = 1.5,
            soc_coeff: float = 1.5,
            include_greedy: bool = False) -> None:
        super().__init__(symbols, font)
        self.generations = generations
        self.pop_count = pop_count
        self.init_innertion = innertion
        self.cog_coeff = cog_coeff
        self.soc_coeff = soc_coeff
        self.include_greedy = include_greedy

    def line2symbols(self, line: Image.Image) -> list[str]:
        particles, fits = generate_line_population(
            line, self.symbols, self.font, self.pop_count, self.include_greedy)
        particles_np = np.array(particles, dtype=np.float32)
        particles_stag_counter = [0] * self.pop_count
        pos_len = len(particles_np[0])

        best_particle_pos = copy.deepcopy(particles_np)
        best_particle_pos_fit = copy.deepcopy(fits)
        best_global_pos = copy.deepcopy(particles_np[0])
        best_global_pos_fit = fits[0]
        print(-1, best_global_pos_fit)

        vmin = 0.5
        vmax = 0.4 * len(self.symbols)

        velocities = [(np.random.random(pos_len) - 0.5) * vmax
                      for _ in range(len(particles_np))]


        for gen in range(self.generations):
            innertion = self.init_innertion * (self.generations - gen) / self.generations
            for i in range(len(particles_np)):
                if particles_stag_counter[i] > 30:
                    particles_np[i] = np.random.random(particles_np[i].shape) \
                                      * len(self.symbols)
                    particles_stag_counter[i] = 0

                r_p = np.random.random(velocities[i].shape)
                r_g = np.random.random(velocities[i].shape)

                v = innertion * velocities[i]
                v += self.cog_coeff * r_p * \
                    (best_particle_pos[i] - particles_np[i])
                v += self.soc_coeff * r_g * (best_global_pos - particles_np[i])

                if np.linalg.norm(v) < vmin:
                    particles_np[i] = (np.random.random(pos_len) - 0.5) * vmax
                    
                # if np.linalg.norm(v) > vmax:
                #     v *= vmax / np.linalg.norm(v)

                velocities[i] = v
                particles_np[i] += velocities[i]
                # particles_np[i] = particles_np[i] % len(self.symbols)

                try:
                    fits[i] = evaluate_symbols_id_arr(
                        list(map(int, particles_np[i])), self.symbols, line, self.font)
                except IndexError:
                    fits[i] = -float('inf')

                if fits[i] > best_particle_pos_fit[i]:
                    best_particle_pos[i] = copy.deepcopy(particles_np[i])
                    best_particle_pos_fit[i] = fits[i]
                    # print(gen, i, best_particle_pos_fit[i])
                    if fits[i] > best_global_pos_fit:
                        best_global_pos = copy.deepcopy(particles_np[i])
                        best_global_pos_fit = fits[i]
                        print(gen, best_global_pos_fit)
                else:
                    particles_stag_counter[i] += 1

            # print(gen)
            # print('avg velocity', np.mean(np.linalg.norm(velocities, axis=1)))
            # print('avg fit', np.mean(best_particle_pos_fit))

        best_p_id_arr = list(map(int, best_global_pos))
        return symbols_id_arr_to_text_arr(best_p_id_arr, self.symbols)
