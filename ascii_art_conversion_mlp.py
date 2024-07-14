from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from copy import deepcopy
import numpy as np
from generate_char_palette import generate_brightness_map


def train_classifier(char_set, font, size, layers, epochs, noise):
    brightness_map = generate_brightness_map(char_set, font, size, normalize=True)
    classes = list(brightness_map.keys())
    brightness_X = np.array([b.flatten() for b in brightness_map.values()])
    brightness_y = list(brightness_map.keys())

    batch_size = 50
    train_X = np.tile(brightness_X, (batch_size,1))
    train_y = brightness_y * batch_size

    cls = MLPClassifier(hidden_layer_sizes=layers)
    for _ in range(epochs):
        new_train_X = deepcopy(train_X)
        new_train_y = deepcopy(train_y)
        
        new_train_X += ((np.random.random(new_train_X.shape)-0.5) * noise)
        new_train_X = np.clip(new_train_X, a_min=0, a_max=1)
        new_train_X, new_train_y = shuffle(new_train_X, new_train_y)
        cls.partial_fit(new_train_X, new_train_y, classes)
    
    return cls, brightness_map