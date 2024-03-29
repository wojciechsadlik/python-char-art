from PIL import Image, ImageDraw, ImageFont
import string

asciis = list(filter(lambda a: a.isprintable(), string.printable))

fnt = ImageFont.truetype("fonts/CascadiaMono-Bold.ttf", 16)

width, height = 0, 0
for a in asciis:
    width = max(width, fnt.getbbox(a)[2])
    height = max(height, fnt.getbbox(a)[3])


asciis = list(map(lambda a: (ord(a), a), asciis))
for a in asciis:
    img = Image.new(mode="L", size=(width, height), color=(0))
    img_d = ImageDraw.Draw(img)
    img_d.text((0,0), a[1], font=fnt, fill=(255))
    img.save("character_imgs/ascii/CascadiaMono-Bold/" + str(a[0]) + ".bmp", format="BMP")