import argparse
import string
from PIL import Image, ImageFont, ImageDraw

from converters.line_heuristics.utils import get_char_width, get_line_height
from image.processing import preprocess_img, DITHER_MODES
from converters.tile.bin_2d import Tile2Symb2dBin
from converters.line_heuristics.harmony_search import HarmonyLineSearch
from converters.img_converter import ImgConverter
from diagnostics.artifact_manager import get_artifact_manager

get_artifact_manager(debug=True)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess an image into grayscale and convert it to symbol art."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input image file")
    parser.add_argument("-c", "--cols", type=int, default=80, help="Maximum output columns (width)")
    parser.add_argument("-l", "--lines", type=int, default=40, help="Maximum output lines (height)")
    parser.add_argument("-f", "--font-path", required=True, help="Path to TTF monospaced font")
    parser.add_argument("-s", "--font-size", type=int, default=16, help="Monospaced font size in px")
    parser.add_argument("-g", "--generations", type=int, default=50, help="Harmony Search generations per line")
    parser.add_argument(
        "--symbols",
        type=str,
        default=string.ascii_letters + string.digits + string.punctuation + " ",
        help="Custom set of symbols to use"
    )
    
    # Preprocessing options
    parser.add_argument("--contrast", type=float, default=1.0, help="Contrast factor")
    parser.add_argument("--quantize-colors", type=int, default=256, help="Number of color quantization levels")
    parser.add_argument("--enhance-edges", type=float, default=0.0, help="Edge enhancement blend factor (0.0 to 1.0)")
    parser.add_argument("--find-edges", type=float, default=0.0, help="Find edges blend factor (0.0 to 1.0)")

    args = parser.parse_args()

    img = Image.open(args.input)
    symbols = list(args.symbols)
    font = ImageFont.truetype(args.font_path, args.font_size)

    prep_img = preprocess_img(
        img,
        grayscale=True,
        contrast=args.contrast,
        quantize_colors=args.quantize_colors,
        dither=DITHER_MODES.FS,
        enhance_edges=args.enhance_edges,
        find_edges=args.find_edges
    )

    line_height = get_line_height(symbols, font)
    col_width = get_char_width(font)
    tile_converter = Tile2Symb2dBin(symbols=symbols, font=font)

    harmony_search = HarmonyLineSearch(
        symbols=symbols,
        font=font,
        generations=args.generations,
        pop_count=10,
        tile_converter=tile_converter
    )

    img_converter = ImgConverter(converter=harmony_search)

    symbol_matrix = img_converter.img2symbols(
        prep_img,
        max_cols=args.cols,
        max_lines=args.lines,
        line_height=line_height,
        col_width=col_width
    )

    text_output = "\n".join("".join(line) for line in symbol_matrix)
    print(text_output)


if __name__ == "__main__":
    main()
