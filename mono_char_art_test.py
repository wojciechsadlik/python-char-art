import argparse
from shutil import get_terminal_size
from PIL import Image, ImageFont, ImageChops
from mono_char_art_conversion_1x1 import img2char_arr_1x1
from mono_char_art_conversion_1x2 import img2char_arr_1x2
from mono_char_art_conversion_wxh import quantize_grayscale_wxh
from mono_char_art_conversion_mlp import train_classifier
from img_processing import preprocess_img, DITHER_MODES
from braille_art import get_braille_chars
from generate_char_palette import *
from mono_char_art_converter import MonoCharArtConverter

parser = argparse.ArgumentParser()
parser.add_argument('img_path', type=str, nargs='?', default="imgs/irad_grad.bmp")
parser.add_argument('--font_path', type=str, nargs='?', default="fonts/CascadiaMono.ttf")
parser.add_argument('--dither', type=str, nargs='?', default=DITHER_MODES.NONE)
parser.add_argument('--cols', type=int, nargs='?')
parser.add_argument('--lines', type=int, nargs='?')
parser.add_argument('--invert', type=bool, nargs='?', const=True, default=False)

args = parser.parse_args()
FONT = ImageFont.truetype(args.font_path, 32)
DITHER = DITHER_MODES(args.dither)
char_set = get_asciis()

img = Image.open(args.img_path).convert("RGB")
if (args.invert):
    img = ImageChops.invert(img)

term_cols, term_lines = get_terminal_size()
if (args.cols):
    term_cols = args.cols
if (args.lines):
    term_lines = args.lines


img_h, img_w = img.size
scale_factor = min(term_cols/img_w, term_lines/img_h)
proc_img = preprocess_img(img, scale_factor, sharpness=2.0, enhance_edges=0.05)

converter = MonoCharArtConverter(
    char_set=char_set,
    font=FONT,
    general_mapping_palette_shape=(12,),
    dither=DITHER
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print('\n')

converter = MonoCharArtConverter(
    char_set=char_set,
    font=FONT,
    general_mapping_palette_shape=(8,8),
    dither=DITHER
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print('\n')



W_H_WIN_SHAPE = (3,6)
scale_factor *= W_H_WIN_SHAPE[0]
proc_img = preprocess_img(img, scale_factor, sharpness=2.0, enhance_edges=0.05)

char_to_brightness_map = generate_brightness_map(char_set, FONT, W_H_WIN_SHAPE, normalize=True)
ascii_arr = quantize_grayscale_wxh(proc_img, char_to_brightness_map,
                                   (W_H_WIN_SHAPE[1], W_H_WIN_SHAPE[0]), DITHER, randomize=False)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')
    
print('\n')


converter = MonoCharArtConverter(
    char_set=char_set,
    font=FONT,
    general_mapping_palette_shape=(9,),
    detailed_mapping_wh=W_H_WIN_SHAPE,
    dither=DITHER
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print('\n')


converter = MonoCharArtConverter(
    char_set=char_set,
    font=FONT,
    general_mapping_palette_shape=(6,6),
    detailed_mapping_wh=W_H_WIN_SHAPE,
    dither=DITHER
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print('\n')


# noise = 0.05
# cls, char_to_brightness_map = train_classifier(char_set, FONT, W_H_WIN_SHAPE, (8,8,8), 50, noise)

# brightness_X = np.array([b.flatten() for b in char_to_brightness_map.values()])
# brightness_y = list(char_to_brightness_map.keys())

# repetitions = 50

# test_X = np.tile(brightness_X, (repetitions,1))
# test_y = list(char_to_brightness_map.keys()) * repetitions
# test_X += ((np.random.random(test_X.shape)-0.5) * noise)
# test_X = np.clip(test_X, a_min=0, a_max=1)

# print('classifier score no noise', cls.score(brightness_X, brightness_y))
# print('classifier score noise', cls.score(test_X, test_y))

# ascii_arr = quantize_grayscale_wxh(proc_img, char_to_brightness_map,
#                                    (W_H_WIN_SHAPE[1], W_H_WIN_SHAPE[0]), DITHER, cls=cls, randomize=True)
# for i in range(len(ascii_arr)):
#     for j in range(len(ascii_arr[i])):
#         print(ascii_arr[i][j], sep='', end='')
#     print('\n', end='')

# print('\n')