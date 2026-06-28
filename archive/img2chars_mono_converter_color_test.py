import argparse
from shutil import get_terminal_size
from PIL import Image, ImageFont, ImageChops
from rendering.ansi_colorizer import AnsiColorizer, reset_code
from image.processing import preprocess_img, DITHER_MODES
from palette.symbol2value import get_asciis
from modules.img2chars_mono_converter import Img2MonoCharsConverter, pick_closest

parser = argparse.ArgumentParser()
parser.add_argument(
    'img_path',
    type=str,
    nargs='?',
    default="imgs/rgb_grad.png")
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
parser.add_argument(
    '--truecolor',
    type=bool,
    nargs='?',
    const=True,
    default=False)
parser.add_argument('--bg_brightness', type=float, nargs='?', default=0.5)
parser.add_argument('--fg_brightness', type=float, nargs='?', default=1.0)
parser.add_argument('--no_bg', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--no_fg', type=bool, nargs='?', const=True, default=False)

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

colorize_settings = AnsiColorizer(
    colored_fg=not args.no_fg,
    colored_bg=not args.no_bg,
    fg_brightness_scale=args.fg_brightness,
    bg_brightness_scale=args.bg_brightness,
    use_ansi_256_colors=not args.truecolor
)

img_h, img_w = img.size
scale_factor = min(term_cols / img_w, term_lines / img_h * 2)
proc_img = preprocess_img(
    img,
    scale_factor,
    grayscale=False)

converter = Img2MonoCharsConverter(
    symbols=symbols,
    font=FONT,
    general_mapping_palette_shape=(12,),
    dither=DITHER,
    ansi_colorizer=colorize_settings
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print(reset_code(), '\n')

converter = Img2MonoCharsConverter(
    symbols=symbols,
    font=FONT,
    general_mapping_palette_shape=(8, 8),
    dither=DITHER,
    ansi_colorizer=colorize_settings
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print(reset_code(), '\n')

scale_factor = min(term_cols / img_w * 2, term_lines / img_h * 4)
proc_img = preprocess_img(
    img,
    scale_factor,
    grayscale=False)
converter = Img2MonoCharsConverter(
    symbols=symbols,
    font=FONT,
    general_mapping_palette_shape=(8, 8),
    dither=DITHER,
    detailed_mapping_win_shape=(4, 2),
    pick_strategy=pick_closest,
    ansi_colorizer=colorize_settings
)
ascii_arr = converter.convert(proc_img)
for i in range(len(ascii_arr)):
    for j in range(len(ascii_arr[i])):
        print(ascii_arr[i][j], sep='', end='')
    print('\n', end='')

print(reset_code(), '\n')
