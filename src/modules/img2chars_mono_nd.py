import numpy as np
from PIL import Image
import random
from sklearn.tree import DecisionTreeClassifier


def find_match(win, symbol2brightness, randomize=False):
    max_dist = np.linalg.norm(np.ones(win.shape))
    max_sim = 0
    max_sym = ''
    close_sym = []
    close_br = []
    for char, br in symbol2brightness.items():
        sim = 1 - np.linalg.norm(win - br) / max_dist

        if (sim > max_sim):
            max_sim = sim
            max_sym = char
        if (sim > 0.75):
            close_sym.append(char)
            close_br.append(br)

    if (len(close_sym) > 1 and randomize):
        rand_char_id = random.randrange(len(close_sym))
        return close_sym[rand_char_id]
    return max_sym


def pick_cls_prediction(win, cls: DecisionTreeClassifier, randomize=False):
    if (not randomize):
        prediction = cls.predict([win.flatten()])[0]
        return prediction

    prediction = cls.predict_proba([win.flatten()])[0]
    rand = random.random()
    prob_sum = 0
    for c, prob in zip(cls.classes_, prediction):
        prob_sum += prob
        if rand <= prob_sum:
            return c


def img2char_arr_nd(img: Image.Image,
                    win_shape,
                    symbol2brightness=None,
                    cls: DecisionTreeClassifier = None,
                    randomize=False) -> list[list[str]]:
    if not symbol2brightness and not cls:
        raise Exception(
            'You need to specify symbol to value mapping or a classifier')
    img_arr = np.array(img) / 255

    char_arr = []
    for up_y in range(
            win_shape[0],
            img_arr.shape[0] + 1,
            win_shape[0]):

        char_arr.append([])
        for up_x in range(
                win_shape[1],
                img_arr.shape[1] + 1,
                win_shape[1]):
            y = up_y - win_shape[0]
            x = up_x - win_shape[1]
            win = img_arr[y:up_y, x:up_x]
            if cls is None:
                min_sym = find_match(
                    win, symbol2brightness, randomize)
            else:
                min_sym = pick_cls_prediction(
                    win, cls, randomize)
            char_arr[-1].append(min_sym)

    return char_arr
