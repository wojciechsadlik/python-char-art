from PIL import Image, ImageDraw

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