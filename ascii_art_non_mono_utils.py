from PIL import Image, ImageDraw

def new_img_draw(size, fill=0):
    img = Image.new("L", size, fill)
    draw = ImageDraw.Draw(img)
    return img, draw

def clear_img(draw, size, fill=0):
    draw.rectangle(((0,0), size), fill=fill)