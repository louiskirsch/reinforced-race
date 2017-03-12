from typing import Iterable, Tuple

import keras
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid import ImageGrid

from memory import Experience


def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum(0)


def add_decision_boundaries(img: np.ndarray, softmax_predictions: np.ndarray) -> np.ndarray:
    rgb_img = np.stack([img, img, img], axis=2)
    width = img.shape[1]
    width_l, width_f, width_r = (width * softmax_predictions).astype(int)
    # Left is red
    rgb_img[:, :width_l, 1:] = 0
    # Center is green
    rgb_img[:, width_l:width_l + width_f, ::2] = 0
    # Right is blue
    rgb_img[:, width_l + width_f:, :2] = 0
    return rgb_img


def predict_experiences(model: keras.models.Model, experiences: Iterable[Experience]) -> np.ndarray:
    x = np.stack(exp.from_state.data for exp in experiences)
    return model.predict(x)


def images_with_probabilities(model: keras.models.Model,
                              experiences: Iterable[Experience]) -> Tuple[np.ndarray, np.ndarray]:
    predictions = predict_experiences(model, experiences)
    for experience, prediction in zip(experiences, predictions):
        img = experience.from_state.data[:, :, -1].squeeze()
        sm_prediction = softmax(prediction - np.max(prediction))
        yield img, sm_prediction


def draw_decisions(model: keras.models.Model, experiences: Iterable[Experience], count: int):
    cols = min(10, count)
    rows = math.ceil(count / cols)
    fig = plt.figure(1, (cols * 2, rows * 2))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0)
    experiences = experiences[:count]

    for i, (experience, probabilities) in enumerate(images_with_probabilities(model, experiences)):
        img = add_decision_boundaries(experience, probabilities)
        grid[i].imshow(img, origin='lower')


def save_decisions(model: keras.models.Model, experiences: Iterable[Experience]):
    for i, (experience, probabilities) in enumerate(images_with_probabilities(model, experiences)):
        img = add_decision_boundaries(experience, probabilities)
        plt.imsave('decisions/d{}.png'.format(i), img, origin='lower')