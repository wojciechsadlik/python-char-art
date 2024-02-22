from PIL import Image, ImageDraw, ImageFont
import string

asciis = list(filter(lambda a: a.isprintable(), string.printable))

fnt = ImageFont.truetype("fonts/CascadiaMono.ttf", 20)

width, height = 0, 0
for a in asciis:
    width = max(width, fnt.getbbox(a)[2])
    height = max(height, fnt.getbbox(a)[3])


asciis = list(map(lambda a: (ord(a), a), asciis))
for a in asciis:
    img = Image.new(mode="1", size=(width, height), color=(1))
    img_d = ImageDraw.Draw(img)
    img_d.text((0,0), a[1], font=fnt, fill=(0))
    img.save("character_imgs/ascii/CascadiaMono/" + str(a[0]) + ".bmp", format="BMP")