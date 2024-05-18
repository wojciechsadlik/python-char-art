import random
import numpy as np
from PIL import ImageChops
from ascii_art_non_mono_utils import new_img_draw, clear_img

def generate_random_line(line, palette, font):
    line_size = line.size
    bg_img, draw = new_img_draw(line.size)
    text = []
    bbox = draw.textbbox((0,0), ''.join(text), font=font)

    while line_size[0] > bbox[2]:
        text.append(palette[random.randint(0, len(palette)-1)])
        bbox = draw.textbbox((0,0), ''.join(text), font=font)

    text.pop()

    draw.text((0,0), ''.join(text), font=font, fill=255)
    return bg_img, text

def generate_greedy_line(line, palette, font):
    line_size = line.size
    bg_img, draw = new_img_draw(line_size)
    text = []
    bbox = draw.textbbox((0,0), ''.join(text), font=font)

    while line_size[0] > bbox[2]:
        best_c = palette[0]
        text.append(palette[0])
        draw.text((0,0), ''.join(text), font=font, fill=255)
        best_c_diff = np.average(ImageChops.difference(line, bg_img))
        text.pop()
        clear_img(draw, line_size)
        for i in range(1, len(palette)):
            text.append(palette[i])
            draw.text((0,0), ''.join(text), font=font, fill=255)
            diff = np.average(ImageChops.difference(line, bg_img))
            if diff < best_c_diff:
                best_c = palette[i]
                best_c_diff = diff
            text.pop()
            clear_img(draw, line_size)
        text.append(best_c)
        bbox = draw.textbbox((0,0), ''.join(text), font=font)
        
    text.pop()

    draw.text((0,0), ''.join(text), font=font, fill=255)
    return bg_img, text