from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from palette.symbol2value import make_symbol2value_map


def make_batch(symbol2value, batch_size=1, noise=0.0):
    brightness_X = np.array([b.flatten() for b in symbol2value.values()])
    brightness_y = list(symbol2value.keys())

    batch_X = np.tile(brightness_X, (batch_size, 1))
    batch_X += (np.random.random(batch_X.shape) - 0.5) * noise
    batch_X = np.clip(batch_X, a_min=0, a_max=1)
    batch_y = brightness_y * batch_size
    return batch_X, batch_y


def classifier_accuracy(
        cls,
        font,
        win_shape,
        noise=0.1,
        batch_size=100):
    symbol2value = make_symbol2value_map(
        symbols=cls.classes_,
        font=font,
        val_height=win_shape[0],
        val_width=win_shape[1],
        normalize=True)

    test_X, test_y = make_batch(
        symbol2value=symbol2value,
        batch_size=batch_size,
        noise=noise)

    pred_y = cls.predict(test_X)

    return accuracy_score(test_y, pred_y)


def train_classifier(
        symbols,
        font,
        win_shape,
        noise=0.0,
        batch_size=1,
        n_estimators=50,
        max_depth=None,
        max_leaf_nodes=None,
        max_samples=0.3):
    symbol2value = make_symbol2value_map(
        symbols=symbols,
        font=font,
        val_height=win_shape[0],
        val_width=win_shape[1],
        normalize=True)

    cls = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        max_samples=max_samples)

    train_X, train_y = make_batch(
        symbol2value=symbol2value,
        batch_size=batch_size,
        noise=noise)

    cls.fit(train_X, train_y)

    return cls
