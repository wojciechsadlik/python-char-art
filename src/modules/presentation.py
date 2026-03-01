from PIL import Image, ImageDraw
from PIL.ImageFont import FreeTypeFont


def print_gallery(char_arrs, columns=2):
    gallery = []
    for i in range(len(char_arrs)):
        if (i % columns == 0):
            gallery.append([])
        gallery[-1].append(char_arrs[i])

    h_sep_len = len(char_arrs[0][0]) * columns + 3 * columns + 3
    joined_arr = []
    for row in gallery:
        h_sep = ['-' for _ in range(h_sep_len)]
        joined_arr.append(''.join(h_sep))

        for y in range(len(row[0])):
            new_row = ' | '
            for char_arr in row:
                new_row += ''.join(char_arr[y]) + ' | '
            joined_arr.append(new_row)

    h_sep = ['-' for _ in range(h_sep_len)]
    joined_arr.append(''.join(h_sep))

    print('\n'.join(joined_arr))


def symbol_arr_to_str(symbol_arr: list[list[str]]) -> str:
    res = ""
    for row in symbol_arr:
        for symb in row:
            res += symb
        res += "\n"
    return res


def render_symbols_img(
        symbol_arr: list[list[str]],
        font: FreeTypeFont,
        bg_color=(0, 0, 0),
        fg_color=(255, 255, 255)) -> Image:
    img = Image.new("RGB", (1, 1), bg_color)
    img_d = ImageDraw.Draw(img)
    text = symbol_arr_to_str(symbol_arr)
    bbox = font.getbbox(text)
    width, height = bbox[2], bbox[3]
    img = Image.new("RGB", (width, height), bg_color)
    img_d = ImageDraw.Draw(img)
    img_d.text((width / 2, height / 2), text, fg_color, anchor="mm")
