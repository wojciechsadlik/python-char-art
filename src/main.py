import argparse
from PIL import Image, ImageFont

from image.processing import preprocess_img, DITHER_MODES
from palette.symbol2value import PALETTES, stream_width_aligned_permutations
from converters.tile.nearest import Tile2SymbNearest
from converters.line_heuristics.harmony_search import HarmonyLineSearch
from converters.img_converter import ImgConverter
from diagnostics.artifact_manager import get_artifact_manager
from rendering.ansi_colorizer import AnsiColorizer, colorize_symbols


get_artifact_manager(debug=False)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess an image into grayscale and convert it to symbol art."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input image file")
    parser.add_argument("-c", "--cols", type=int, default=80,
                        help="Maximum output columns (width)")
    parser.add_argument("-l", "--lines", type=int, default=24,
                        help="Maximum output lines (height)")
    parser.add_argument("-f", "--font-path", required=True,
                        help="Path to TTF font")
    parser.add_argument("-s", "--font-size", type=int,
                        default=16, help="Font size in px")
    parser.add_argument("-g", "--generations", type=int,
                        default=50, help="Harmony Search generations per line")
    parser.add_argument("--symbols", type=str,
                        default=PALETTES['asciis'], help="Custom set of symbols to use")
    parser.add_argument("--non-mono", action="store_true",
                        help="Use when using a non monospaced font")

    # Colorization Flags
    parser.add_argument("--colorize", action="store_true",
                        help="Use ANSI to colorize")
    parser.add_argument("--ansi-256", action="store_true",
                        help="Use ANSI 256 colors instead of standard 8/16 palette")

    # Preprocessing options
    parser.add_argument("--contrast", type=float,
                        default=1.0, help="Contrast factor")
    parser.add_argument("--quantize-colors", type=int,
                        default=256, help="Number of color quantization levels")
    parser.add_argument("--enhance-edges", type=float, default=0.0,
                        help="Edge enhancement blend factor (0.0 to 1.0)")
    parser.add_argument("--find-edges", type=float, default=0.0,
                        help="Find edges blend factor (0.0 to 1.0)")
    parser.add_argument("--invert", action="store_true",
                        help="Invert colors")

    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    symbols = list(args.symbols)
    font = ImageFont.truetype(args.font_path, args.font_size)

    prep_img = preprocess_img(
        img,
        grayscale=True,
        contrast=args.contrast,
        quantize_colors=args.quantize_colors,
        dither=DITHER_MODES.FS,
        enhance_edges=args.enhance_edges,
        find_edges=args.find_edges,
        invert=args.invert
    )

    if args.non_mono:
        symbols = list(stream_width_aligned_permutations(symbols, font))

    tile_converter = Tile2SymbNearest(symbols=symbols,
                                      font=font,
                                      val_wh=(3, 6) if not args.non_mono else (6, 6))

    harmony_search = HarmonyLineSearch(
        symbols=symbols,
        font=font,
        generations=args.generations,
        pop_count=5,
        tile_converter=tile_converter
    )

    img_converter = ImgConverter(converter=harmony_search, font=font)

    symbol_matrix = img_converter.img2symbols(
        prep_img,
        max_cols=args.cols,
        max_lines=args.lines
    )

    if args.colorize:
        ansi_colorizer = AnsiColorizer(
            fg_brightness_scale=1.0,
            bg_brightness_scale=0.8,
            use_ansi_256_colors=args.ansi_256)
        symbol_matrix = colorize_symbols(
            prep_img, symbol_matrix, font, ansi_colorizer)

    text_output = "\n".join("".join(line) for line in symbol_matrix)
    print(text_output)


if __name__ == "__main__":
    main()
