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
            generations: int = 100,
            pop_count: int = 50,
            innertion: float = 0.3,
            cog_coeff: float = 0.9,
            soc_coeff: float = 1.2,
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
        particles_np = np.array(particles, dtype=np.int32)
        particles_stag_counter = [0] * self.pop_count
        pos_len = len(particles_np[0])
        num_symbols = len(self.symbols)

        best_particle_pos = copy.deepcopy(particles_np)
        best_particle_pos_fit = copy.deepcopy(fits)
        best_global_pos = copy.deepcopy(particles_np[0])
        best_global_pos_fit = fits[0]

        velocities = np.zeros((self.pop_count, pos_len, num_symbols), dtype=np.float32)

        for gen in range(self.generations):
            innertion = self.init_innertion * (self.generations - gen) / self.generations
            for i in range(self.pop_count):
                if particles_stag_counter[i] > 30:
                    particles_np[i] = np.random.randint(0, num_symbols, size=pos_len)
                    particles_stag_counter[i] = 0

                r_p = np.random.random((pos_len, 1))
                r_g = np.random.random((pos_len, 1))

                pbest_onehot = np.zeros((pos_len, num_symbols), dtype=np.float32)
                pbest_onehot[np.arange(pos_len), best_particle_pos[i]] = 1.0

                gbest_onehot = np.zeros((pos_len, num_symbols), dtype=np.float32)
                gbest_onehot[np.arange(pos_len), best_global_pos] = 1.0

                v = innertion * velocities[i]
                v += self.cog_coeff * r_p * pbest_onehot
                v += self.soc_coeff * r_g * gbest_onehot
                velocities[i] = v

                v_shifted = v - np.max(v, axis=1, keepdims=True)
                exp_v = np.exp(v_shifted)
                probs = exp_v / np.sum(exp_v, axis=1, keepdims=True)

                for j in range(pos_len):
                    particles_np[i, j] = np.random.choice(num_symbols, p=probs[j])

                try:
                    fits[i] = evaluate_symbols_id_arr(
                        particles_np[i].tolist(), self.symbols, line, self.font)
                except IndexError:
                    fits[i] = -float('inf')

                if fits[i] > best_particle_pos_fit[i]:
                    best_particle_pos[i] = copy.deepcopy(particles_np[i])
                    best_particle_pos_fit[i] = fits[i]
                    # print(gen, i, best_particle_pos_fit[i])
                    if fits[i] > best_global_pos_fit:
                        best_global_pos = copy.deepcopy(particles_np[i])
                        best_global_pos_fit = fits[i]
                        # print(gen, best_global_pos_fit)
                else:
                    particles_stag_counter[i] += 1

        best_p_id_arr = best_global_pos.tolist()
        return symbols_id_arr_to_text_arr(best_p_id_arr, self.symbols)
