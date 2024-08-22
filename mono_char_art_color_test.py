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
from ansi_colorizer import AnsiColorizer, reset_code

parser = argparse.ArgumentParser()
parser.add_argument('img_path', type=str, nargs='?', default="imgs/rgb_grad.png")
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

colorize_settings = AnsiColorizer(
    colored_fg=True,
    colored_bg=True,
    fg_brightness_scale=1.1,
    bg_brightness_scale=0.5,
    use_ansi_256_colors=False
)

img_h, img_w = img.size
scale_factor = min(term_cols/img_w, term_lines/img_h)
proc_img = preprocess_img(img, scale_factor, sharpness=2.0, enhance_edges=0.05, grayscale=False)

converter = MonoCharArtConverter(
    char_set=char_set,
    font=FONT,
    general_mapping_palette_shape=(12,),
    colorize_settings=colorize_settings,
    dither=DITHER
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print(reset_code() + '\n', end='')

print('\n')

converter = MonoCharArtConverter(
    char_set=char_set,
    font=FONT,
    general_mapping_palette_shape=(8,8),
    colorize_settings=colorize_settings,
    dither=DITHER
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print(reset_code() + '\n', end='')

print('\n')



# W_H_WIN_SHAPE = (3,6)
# scale_factor *= W_H_WIN_SHAPE[0]
# proc_img = preprocess_img(img, scale_factor, sharpness=2.0, enhance_edges=0.05, grayscale=False)


# converter = MonoCharArtConverter(
#     char_set=char_set,
#     font=FONT,
#     general_mapping_palette_shape=(9,),
#     detailed_mapping_wh=W_H_WIN_SHAPE,
#     colorize_settings=colorize_settings,
#     dither=DITHER
# )
# ascii_arr = converter.convert(proc_img)
# for i in range(len(ascii_arr)):
#     for j in range(len(ascii_arr[i])):
#         print(ascii_arr[i][j], sep='', end='')
#     print(reset_code() + '\n', end='')

# print('\n')


# converter = MonoCharArtConverter(
#     char_set=char_set,
#     font=FONT,
#     general_mapping_palette_shape=(6,6),
#     detailed_mapping_wh=W_H_WIN_SHAPE,
#     colorize_settings=colorize_settings,
#     dither=DITHER
# )
# ascii_arr = converter.convert(proc_img)
# for i in range(len(ascii_arr)):
#     for j in range(len(ascii_arr[i])):
#         print(ascii_arr[i][j], sep='', end='')
#     print(reset_code() + '\n', end='')

# print('\n')
