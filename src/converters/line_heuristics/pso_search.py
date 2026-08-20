import copy
from typing import Generator, Optional
import numpy as np
from PIL import Image, ImageFont

from converters.line_heuristics.line_converter import LineConverter
from converters.line_heuristics.utils import (
    evaluate_symbols_id_arr,
    generate_line_population,
    symbols_id_arr_to_text_arr,
)
from converters.tile.tile_converter import TileConverter
from diagnostics.artifact_manager import get_artifact_manager
from palette.symbol2value import symbols_sorted


class ParticleSwarmLineSearch(LineConverter):
    def __init__(
            self,
            symbols: list[str],
            font: ImageFont.FreeTypeFont,
            generations: int = 100,
            pop_count: int = 20,
            innertion: float = 1.0,
            cog_coeff: float = 2.0,
            soc_coeff: float = 2.0,
            noise_std: float = 1.0,
            confidence_thresh: float = 0.95,
            include_greedy: bool = False,
            tile_converter: Optional[TileConverter] = None) -> None:
        super().__init__(tile_converter=tile_converter)
        self.symbols = symbols_sorted(symbols, font)
        self.font = font
        self.generations = generations
        self.pop_count = pop_count
        self.init_innertion = innertion
        self.cog_coeff = cog_coeff
        self.soc_coeff = soc_coeff
        self.noise_std = noise_std
        self.confidence_thresh = confidence_thresh
        self.include_greedy = include_greedy

    def line2symbols_lazy(self,
                          line: Image.Image,
                          col_width: Optional[int] = None,
                          **kwarg
                          ) -> Generator[list[str], None, None]:
        particles, fits = generate_line_population(
            line,
            self.symbols,
            self.font,
            self.pop_count,
            include_greedy=self.include_greedy,
            tile_converter=self.tile_converter,
            col_width=col_width,
        )
        particles_np = np.array(particles, dtype=np.int32)
        pos_len = len(particles_np[0])
        num_symbols = len(self.symbols)

        best_particle_pos = copy.deepcopy(particles_np)
        best_particle_pos_fit = copy.deepcopy(fits)
        best_global_pos = copy.deepcopy(particles_np[0])
        best_global_pos_fit = fits[0]

        velocities = np.zeros(
            (self.pop_count, pos_len, num_symbols), dtype=np.float32)

        sorted_indices = np.argsort(best_particle_pos_fit)[::-1]
        self.current_candidates = [
            symbols_id_arr_to_text_arr(best_particle_pos[idx].tolist(), self.symbols)
            for idx in sorted_indices[:4]
        ]

        yield symbols_id_arr_to_text_arr(best_global_pos.tolist(), self.symbols)

        for gen in range(self.generations):
            get_artifact_manager().current_gen = gen
            innertion = self.init_innertion * \
                (self.generations - gen) / self.generations
            for i in range(self.pop_count):
                r_p = np.random.random()
                r_g = np.random.random()

                pbest_onehot = np.zeros(
                    (pos_len, num_symbols), dtype=np.float32)
                pbest_onehot[np.arange(pos_len), best_particle_pos[i]] = r_p

                gbest_onehot = np.zeros(
                    (pos_len, num_symbols), dtype=np.float32)
                gbest_onehot[np.arange(pos_len), best_global_pos] = r_g

                v = innertion * velocities[i]
                v += self.cog_coeff * r_p * pbest_onehot
                v += self.soc_coeff * r_g * gbest_onehot

                if self.noise_std > 0:
                    v += np.random.normal(0.0, self.noise_std, size=v.shape)

                velocities[i] = v

                v_shifted = v - np.max(v, axis=1, keepdims=True)
                exp_v = np.exp(v_shifted)
                probs = exp_v / np.sum(exp_v, axis=1, keepdims=True)

                if np.mean(np.max(probs, axis=1)) > self.confidence_thresh:
                    velocities[i] = np.zeros((pos_len, num_symbols), dtype=np.float32)
                    probs = np.full((pos_len, num_symbols), 1.0 / num_symbols, dtype=np.float32)

                for j in range(pos_len):
                    particles_np[i, j] = np.random.choice(
                        num_symbols, p=probs[j])

                try:
                    fits[i] = evaluate_symbols_id_arr(
                        particles_np[i].tolist(), self.symbols, line, self.font)
                except IndexError:
                    fits[i] = -float('inf')

                if fits[i] > best_particle_pos_fit[i]:
                    best_particle_pos[i] = copy.deepcopy(particles_np[i])
                    best_particle_pos_fit[i] = fits[i]
                    if fits[i] > best_global_pos_fit:
                        best_global_pos = copy.deepcopy(particles_np[i])
                        best_global_pos_fit = fits[i]

            sorted_indices = np.argsort(best_particle_pos_fit)[::-1]
            self.current_candidates = [
                symbols_id_arr_to_text_arr(best_particle_pos[idx].tolist(), self.symbols)
                for idx in sorted_indices[:4]
            ]
            
            yield symbols_id_arr_to_text_arr(best_global_pos.tolist(), self.symbols)
