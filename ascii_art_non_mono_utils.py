from PIL import Image, ImageDraw, ImageChops
import numpy as np

def new_img_draw(size, fill=0):
    img = Image.new("L", size, fill)
    draw = ImageDraw.Draw(img)
    return img, draw

def clear_img(img, draw, fill=0):
    draw.rectangle(((0,0), img.size), fill=fill)

def split_lines(img, palette, font):
    _, draw = new_img_draw(img.size)
    text = []
    bbox = draw.textbbox((0,0), ''.join(text), font=font)
    img_size = img.size
    lines = []
    while bbox[3] < img_size[1]:
        text.append(''.join(palette) + '\n')
        bbox = draw.textbbox((0,0), ''.join(text), font=font)

    line_width = img.size[0]
    line_height = img.size[1] // len(text)
    lines = []
    for i in range(len(text)):
        lines.append(img.crop((0, i * line_height, line_width, (i+1) * line_height)))
    return lines

def palette_ids_to_text_arr(p_id_arr, palette):
    text_arr = []
    for id in p_id_arr:
        text_arr.append(palette[id])
    return text_arr

def evaluate_text_arr(text_arr, img, font):
    text_img, text_draw = new_img_draw(img.size, int(np.mean(img)))
    text_draw.text((0,0), ''.join(text_arr), font=font, fill=255)
    return np.mean(ImageChops.difference(text_img, img))