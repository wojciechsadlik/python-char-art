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

palette_fills = [(' ', 0.0),
('.', 0.024305555555555556),
(':', 0.04861111111111111),
(';', 0.07291666666666667),
('~', 0.09027777777777778),
('+', 0.125),
('Y', 0.1527777777777778),
('s', 0.1701388888888889),
('3', 0.1840277777777778),
('V', 0.1909722222222222),
('4', 0.20833333333333334),
('A', 0.2152777777777778),
('H', 0.22916666666666666),
('#', 0.2534722222222222),
('N', 0.2673611111111111),
('@', 0.2881944444444444)]

def quantize2(img: Image.Image) -> Image.Image:
    fills_to_brightness = [int(b[1] * 255) for b in palette_fills]
    print(fills_to_brightness)
    quantize_palette = list(np.array([[c,c,c] for c in fills_to_brightness]).flatten())
    palette_img = Image.new("P", (1,1))
    palette_img.putpalette(quantize_palette)
    return img.quantize(palette=palette_img)