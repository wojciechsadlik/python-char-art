import numpy as np

default_ascii_palette = [' ', '.', ':', ';', '!', '+', 'o', '4', 'D', 'N', 'Q', '@']
default_ascii_palette_v2 = [[' ', '.', ',', '_', '_', '_', '_', '_', '_'],
                            [' ', ':', '~', ';', 'u', 'u', 'u', 'u', 'u'],
                            [' ', ':', '+', 'v', 'o', 'e', 'y', 'a', 'a'],
                            ["'", '!', '>', 'c', 'k', 'p', 'q', 'g', 'g'],
                            ['`', '!', '<', '\\', 'J', 'A', '&', '&', '&'],
                            ['"', 'f', 'F', 'K', '2', 'G', 'd', 'd', 'd'],
                            ['^', 'f', 'P', 'R', '5', '#', 'Q', 'Q', 'Q'],
                            ['^', 'f', 'P', '9', '0', 'W', 'Q', 'Q', 'Q'],
                            ['^', 'f', 'P', 'M', '$', 'W', 'Q', 'Q', '@']]

def img2ascii_arr(img_arr: np.ndarray, img_colors: int = 256, palette: list[str] = default_ascii_palette) -> list[list[str]]:
    palette_interval = img_colors / len(palette)
    ascii_arr = []
    for i in range(img_arr.shape[0]):
        ascii_arr.append([])
        for j in range(img_arr.shape[1]):
            ascii_arr[-1].append(palette[int(img_arr[i][j]//palette_interval)])
    return ascii_arr

def img2ascii_arr_v2(img_arr: np.ndarray, img_colors: int = 256, palette: list[list[str]] = default_ascii_palette_v2) -> list[list[str]]:
    palette_y_interval = img_colors / len(palette)
    palette_x_interval = img_colors / len(palette[0])
    ascii_arr = []
    for y in range(1, img_arr.shape[0], 2):
        ascii_arr.append([])
        for x in range(img_arr.shape[1]):
            ascii_arr[-1].append(palette[int(img_arr[y-1][x]//palette_y_interval)][int(img_arr[y][x]//palette_x_interval)])
    return ascii_arr