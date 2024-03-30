from PIL import Image, ImageOps, ImageEnhance
import numpy as np

def generate_bayer_matrix(n):
    if n == 1:
        return np.array([[0]])
    
    n_sq = n ** 2
    m = generate_bayer_matrix(n/2)
    r0 = np.concatenate((n_sq * m, n_sq * m + 2), axis=1)
    r1 = np.concatenate((n_sq * m + 3, n_sq * m + 1), axis=1)
    m = np.concatenate((r0, r1), axis=0)
    return (1 / n_sq) * m

dither_bayer_m = generate_bayer_matrix(32) - 0.5

def apply_threshold_map(c, m, r, x, y):
    n = len(m)
    return c + r * m[x % n][y % n]

def quantize(img: Image.Image, img_colors: int) -> Image.Image:
    if (img_colors == 1):
        quantize_palette = np.array([0,0,0])
    else:
        quantize_palette = list(np.array([[c,c,c] for c in range(0, 256, 255//(img_colors-1))]).flatten())
    palette_img = Image.new("P", (1,1))
    palette_img.putpalette(quantize_palette)
    return img.quantize(palette=palette_img)

def quantize_grayscale(img: Image.Image, img_colors: int) -> Image.Image:
    if (img.mode != "L"):
        raise "wrong img mode"
    if (img_colors <= 0):
        raise "img_colors <= 0"

    m = dither_bayer_m
    img_arr = np.array(img)
    color_step = 256 / img_colors
    palette = np.linspace(0, 255, img_colors)
    for y in range(0, img_arr.shape[0]):
        for x in range(0, img_arr.shape[1]):
            c_dithered = int(apply_threshold_map(img_arr[y][x], m, color_step, x, y) / color_step)
            c_dithered = min(len(palette) - 1, max(0, c_dithered))
            img_arr[y][x] = palette[c_dithered]
    
    return Image.frombytes("L", img_arr.shape, img_arr)


def preprocess_img(img: Image.Image,
                   scale_factor=1,
                   contrast=1,
                   brightness=1,
                   eq=0,
                   quantize_colors=255):
    img = ImageOps.scale(img, scale_factor, Image.Resampling.HAMMING)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = Image.blend(img, ImageOps.equalize(img), eq)
    img = quantize_grayscale(img, quantize_colors)
    img = img.convert("L")
    return img