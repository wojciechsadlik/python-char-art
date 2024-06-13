from PIL import Image, ImageDraw, ImageOps
import string
import numpy as np

def get_asciis():
    return list(filter(lambda a: a.isprintable(), string.printable))

def max_brighntess_val(brightnesses):
    max_b_glob = 0
    for b in brightnesses:
        max_b = max(b.flatten())
        if max_b_glob < max_b:
            max_b_glob = max_b
    return max_b_glob

def max_brightness_per_pos(brightnesses):
    max_br = np.zeros_like(brightnesses[0])
    for b in brightnesses:
        for y in range(b.shape[0]):
            for x in range(b.shape[1]):
                if b[y][x] > max_br[y][x]:
                    max_br[y][x] = b[y][x]
    return max_br

def normalize_brightness_map(brightnesses):
    return [b / max_brighntess_val(brightnesses) for b in brightnesses]

def generate_brightness_map(char_set, font, window_size, bg_color=0, char_color=255, normalize=False):
    width, height = 0, 0
    for char in char_set:
        width = max(width, font.getbbox(char)[2])
        height = max(height, font.getbbox(char)[3])

    brightnesses = []
    for char in char_set:
        img = Image.new(mode="L", size=(width, height), color=bg_color)
        img_d = ImageDraw.Draw(img)
        img_d.text((width,height), char, font=font, fill=char_color, anchor='rd')
        res_img = img.resize(window_size, Image.Resampling.BOX)
        brightnesses.append(np.array(res_img)/255)

    if normalize:
        brightnesses = normalize_brightness_map(brightnesses)

    return {c: b for c, b in zip(char_set, brightnesses)}

def generate_1_1_palette(char_set, font, bins=12, bg_color=0, char_color=255, normalize=False):
    char_to_brightness = generate_brightness_map(char_set, font, (1,1), bg_color, char_color, normalize)
    char_to_brightness = dict(map(lambda c_b: (c_b[0], c_b[1][0][0]), char_to_brightness.items()))
    sorted_map = sorted(char_to_brightness.items(), key=lambda c_b: c_b[1])

    br_max = max_brighntess_val(char_to_brightness.values())
    char_bins = [[] for _ in range(bins)]
    br_step = br_max / (bins-1)
    for char, char_br in sorted_map:
        bin_index = int(round(char_br / br_step))
        char_bins[bin_index].append(char)
    
    if (len(char_bins[0]) == 0):
        char_bins[0].append(sorted_map[0][1])

    for i in range(1, bins):
        if (len(char_bins[i]) == 0):
            char_bins[i].append(char_bins[i-1][-1])

    bin_to_brightness = []
    for bin in char_bins:
        bin_to_brightness.append(np.mean([char_to_brightness[c] for c in bin]))

    return char_bins, bin_to_brightness

def generate_1_2_palette(char_palette, font, bins=(9,9), bg_color=0, char_color=255, normalize=False):
    char_to_brightness = generate_brightness_map(char_palette, font, (1,2), bg_color, char_color, normalize)
    x_bins = bins[0]
    y_bins = bins[1]

    char_bins = [[[] for _ in range(x_bins)] for _ in range(y_bins)]
    br_max = max_brightness_per_pos(list(char_to_brightness.values()))
    br_y_step = br_max[0][0] / (y_bins-1)
    br_x_step = br_max[1][0] / (x_bins-1)
    for char, char_br in char_to_brightness.items():
        bin_y_index = int(round(char_br[0][0] / br_y_step))
        bin_x_index = int(round(char_br[1][0] / br_x_step))
        char_bins[bin_y_index][bin_x_index].append(char)

    def splice_cell(top, left):
        if len(top) > 0:
            res = top[0]
        else:
            res = left[0]
        for a in top:
            if (np.mean(char_to_brightness[res]) < np.mean(char_to_brightness[a])):
                res = a

        for a in left:
            if (np.mean(char_to_brightness[res]) < np.mean(char_to_brightness[a])):
                res = a
        return res

    for y in range(y_bins):
        for x in range(x_bins):
            if len(char_bins[y][x]) == 0:
                if x > 0 and y > 0:
                    char_bins[y][x].append(splice_cell(char_bins[y-1][x], char_bins[y][x-1]))
                elif x > 0:
                    char_bins[y][x].append(splice_cell([], char_bins[y][x-1]))
                elif y > 0:
                    char_bins[y][x].append(splice_cell(char_bins[y-1][x], []))

    bin_to_brightness = []
    for row in char_bins:
        bin_to_brightness.append([])
        for bin in row:
            bin_to_brightness[-1].append(np.mean([char_to_brightness[c] for c in bin], axis=0))

    return char_bins, bin_to_brightness
