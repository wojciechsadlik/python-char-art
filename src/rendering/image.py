from PIL import Image, ImageDraw, ImageFont


def symbol_arr_to_str(symbol_arr: list[list[str]]) -> str:
    return "\n".join("".join(row) for row in symbol_arr)


def render_symbols_img(
        symbol_arr: list[list[str]],
        font: ImageFont.FreeTypeFont,
        bg_color=(0, 0, 0),
        fg_color=(255, 255, 255),
        wh: tuple[float, float] = None) -> Image.Image:

    text = symbol_arr_to_str(symbol_arr)

    if wh is None:
        img_d = ImageDraw.Draw(Image.new("1", (1, 1)))
        bbox = img_d.multiline_textbbox((0, 0), text=text, font=font)
        wh = (bbox[2], bbox[3])

    img = Image.new("RGB", (wh[0], wh[1]), bg_color)
    ImageDraw.Draw(img).multiline_text(
        (wh[0] / 2, wh[1] / 2), anchor="mm", text=text, font=font, fill=fg_color)

    return img
