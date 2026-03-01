import argparse
from shutil import get_terminal_size
from PIL import Image, ImageFont, ImageChops
from modules.img_processing import preprocess_img, DITHER_MODES
from modules.palette_generation import get_asciis
from modules.img2chars_mono_converter import Img2MonoCharsConverter, pick_closest

parser = argparse.ArgumentParser()
parser.add_argument(
    'img_path',
    type=str,
    nargs='?',
    default="imgs/irad_grad.bmp")
parser.add_argument(
    '--font_path',
    type=str,
    nargs='?',
    default="fonts/CascadiaMono.ttf")
parser.add_argument('--dither', type=str, nargs='?', default=DITHER_MODES.NONE)
parser.add_argument('--cols', type=int, nargs='?')
parser.add_argument('--lines', type=int, nargs='?')
parser.add_argument(
    '--invert',
    type=bool,
    nargs='?',
    const=True,
    default=False)

args = parser.parse_args()
FONT = ImageFont.truetype(args.font_path, 11)
DITHER = DITHER_MODES(args.dither)
symbols = get_asciis()

img = Image.open(args.img_path).convert("RGB")
if (args.invert):
    img = ImageChops.invert(img)

term_cols, term_lines = get_terminal_size()
if (args.cols):
    term_cols = args.cols
if (args.lines):
    term_lines = args.lines


img_h, img_w = img.size
scale_factor = min(term_cols / img_w, term_lines / img_h * 2)
proc_img = preprocess_img(
    img,
    scale_factor)

converter = Img2MonoCharsConverter(
    symbols=symbols,
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

converter = Img2MonoCharsConverter(
    symbols=symbols,
    font=FONT,
    general_mapping_palette_shape=(8, 8),
    dither=DITHER
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print('\n')

detailed_win_h, detailed_win_w = 4, 2
scale_factor = min(
    term_cols /
    img_w *
    detailed_win_w,
    term_lines /
    img_h *
    detailed_win_h)

proc_img = preprocess_img(
    img,
    scale_factor)
converter = Img2MonoCharsConverter(
    symbols=symbols,
    font=FONT,
    general_mapping_palette_shape=(8, 8),
    dither=DITHER,
    detailed_mapping_win_shape=(detailed_win_h, detailed_win_w),
    pick_strategy=pick_closest
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print('\n')
