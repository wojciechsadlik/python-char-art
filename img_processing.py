from PIL import Image, ImageOps, ImageEnhance
import numpy as np

def quantize(img: Image.Image, img_colors: int) -> Image.Image:
    if (img_colors == 1):
        quantize_palette = np.array([0,0,0])
    else:
        quantize_palette = list(np.array([[c,c,c] for c in range(0, 256, 255//(img_colors-1))]).flatten())
    palette_img = Image.new("P", (1,1))
    palette_img.putpalette(quantize_palette)
    return img.quantize(palette=palette_img)

def preprocess_img(img: Image.Image,
                   scale_factor=1,
                   contrast=1,
                   brightness=1,
                   eq=0,
                   quantize_colors=255):
    img = ImageOps.scale(img, scale_factor, Image.Resampling.BOX)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = Image.blend(img, ImageOps.equalize(img), eq)
    img = quantize(img, quantize_colors)
    img = img.convert("L")
    return img