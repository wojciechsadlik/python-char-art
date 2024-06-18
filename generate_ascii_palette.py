from PIL import Image, ImageDraw, ImageOps, ImageEnhance
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

def generate_brightness_map(char_set, font, window_wh_size, bg_color=0, char_color=255, normalize=False):
    width, height = 0, 0
    for char in char_set:
        width = max(width, font.getbbox(char)[2])
        height = max(height, font.getbbox(char)[3])

    brightnesses = []
    for char in char_set:
        img = Image.new(mode="L", size=(width, height), color=bg_color)
        img_d = ImageDraw.Draw(img)
        img_d.text((width/2,height/2), char, font=font, fill=char_color, anchor='mm')
        res_img = img.resize(window_wh_size, Image.Resampling.BICUBIC)
        res_arr = np.array(res_img)/255
        brightnesses.append(res_arr)

    if normalize:
        brightnesses = normalize_brightness_map(brightnesses)

    return {c: b for c, b in zip(char_set, brightnesses)}

def generate_non_mono_brightness_map(char_set, font, max_width, height, bg_color=0, char_color=255, normalize=False):
    max_width_in_set = max([font.getbbox(c)[2] for c in char_set])
    width_scale = max_width / max_width_in_set
    brightnesses = []
    for char in char_set:
        _, _, char_width, char_height = font.getbbox(char)
        img = Image.new(mode="L", size=(char_width, char_height), color=bg_color)
        img_d = ImageDraw.Draw(img)
        img_d.text((char_width/2, char_height/2), char, font=font, fill=char_color, anchor='mm')
        res_img = img.resize((int(char_width * width_scale), height), Image.Resampling.BICUBIC)
        brightnesses.append(np.array(res_img)/255)

    if normalize:
        brightnesses = normalize_brightness_map(brightnesses)

    return {c: b for c, b in zip(char_set, brightnesses)}

def generate_non_mono_multi_char_brightness_map(char_set, font, width, height, bg_color=0, char_color=255, normalize=False):
    width_sorted_set = sorted([(c, font.getbbox(c)[2]) for c in char_set], key=lambda c_w: c_w[1])
    max_s, max_width = width_sorted_set.pop()
    final_set = [max_s]
    visited = set()
    while len(width_sorted_set) > 0:
        s1, w1 = width_sorted_set[-1]
        if s1 in visited:
            width_sorted_set.pop()
            continue
        else:
            visited.add(s1)

        expanded = False
        for s2, w2 in width_sorted_set:
            new_s1 = ''
            new_s2 = ''
            new_w = 0
            if w1 + w2 <= max_width:
                new_s1 = s1+s2
                new_s2 = s2+s1
                new_w = w1+w2
            if new_w > 0:
                expanded = True
                width_sorted_set.append((new_s1, new_w))
                width_sorted_set.append((new_s2, new_w))
        
        if not expanded:
            final_set.append(s1)
            width_sorted_set.pop()
    
    brightnesses = []
    for char_str in final_set:
        _, _, char_str_width, char_str_height = font.getbbox(char_str)
        img = Image.new(mode="L", size=(char_str_width, char_str_height), color=bg_color)
        img_d = ImageDraw.Draw(img)
        img_d.text((char_str_width/2, char_str_height/2), char_str, font=font, fill=char_color, anchor='mm')
        res_img = img.resize((width, height), Image.Resampling.BICUBIC)
        brightnesses.append(np.array(res_img)/255)

    if normalize:
        brightnesses = normalize_brightness_map(brightnesses)

    return {c: b for c, b in zip(final_set, brightnesses)}

def find_brightness_map(char_set, font, window_wh_size, bg_color=0, char_color=255, normalize=False):
    b_step = 0.1
    low_b = b_step
    high_b = 6
    res = []
    for b in np.arange(low_b, high_b, b_step):
        brightness_map = generate_brightness_map(char_set, font, window_wh_size, bg_color, char_color,
                                                 brightness_mod=b, normalize=normalize)
        distances = []
        brightness = list(brightness_map.values())
        for i in range(len(brightness)-1):
            i_distances = []
            for j in range(i+1, len(brightness)):
                i_distances.append(np.linalg.norm(brightness[i]-brightness[j]))
            distances.append(np.mean(sorted(i_distances)[:3]))
        res.append(np.mean(distances))
    b = low_b + res.index(max(res)) * b_step
    return generate_brightness_map(char_set, font, window_wh_size, bg_color, char_color,
                                    brightness_mod=b, normalize=normalize)

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

def generate_1_2_palette(char_palette, font, bins=(9,9), bg_color=0, char_color=255, normalize=False, search_map=False):
    if search_map:
        char_to_brightness = find_brightness_map(char_palette, font, (1,2), normalize)
    else:
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
