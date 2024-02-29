from PIL import Image
import numpy as np

def quantize(img: Image.Image, img_colors: int) -> Image.Image:
    quantize_palette = np.array([[c,c,c] for c in range(0, 256, 256//img_colors)]).flatten()
    palette_img = Image.new("P", (1,1))
    palette_img.putpalette(quantize_palette)
    return img.quantize(palette=palette_img)