from PIL import Image
import numpy as np

def quantize(img: Image.Image, img_colors: int) -> Image.Image:
    if (img_colors == 1):
        quantize_palette = np.array([0,0,0])
    else:
        quantize_palette = list(np.array([[c,c,c] for c in range(0, 256, 255//(img_colors-1))]).flatten())
    palette_img = Image.new("P", (1,1))
    palette_img.putpalette(quantize_palette)
    return img.quantize(palette=palette_img)
