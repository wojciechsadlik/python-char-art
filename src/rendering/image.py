from PIL import Image, ImageDraw


def symbol_arr_to_str(symbol_arr: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in symbol_arr) + "\n"


def render_symbols_img(
        symbol_arr: list[list[str]],
        font,
        bg_color=(0, 0, 0),
        fg_color=(255, 255, 255)) -> Image.Image:

    text = symbol_arr_to_str(symbol_arr)

    img_d = ImageDraw.Draw(Image.new("1", (1, 1)))
    bbox = img_d.multiline_textbbox((0, 0), text=text, font=font)

    img = Image.new("RGB", (bbox[2], bbox[3]), bg_color)
    ImageDraw.Draw(img).multiline_text(
        (0, 0), text=text, font=font, fill=fg_color)

    return img
