from abc import ABC, abstractmethod

from PIL import Image
import numpy as np

from image.array_ops import map_img_arr


class TileConverter(ABC):
    @abstractmethod
    def tile2symbol(self, tile: np.ndarray) -> str:
        pass

    def process_image(self, img: Image.Image,
                      win_wh: tuple[int, int]) -> list[list[str]]:
        img_arr = np.array(img)
        return map_img_arr(
            img_arr,
            width=win_wh[0],
            height=win_wh[1],
            callback=self.tile2symbol
        )
