import math
from typing import Iterable, Tuple, List

import keras
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from mpl_toolkits.axes_grid import ImageGrid

from racelearning.memory import Experience


def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum(0)


def add_decision_boundaries(img: np.ndarray, softmax_predictions: np.ndarray) -> np.ndarray:
    rgb_img = np.stack([img, img, img], axis=2)
    width = img.shape[1]
    bar_height = 5
    width_l, width_f, width_r = (width * softmax_predictions).astype(int)
    # Clear top where the bar is located
    rgb_img[-bar_height:, :, :] = 1
    # Left is red
    rgb_img[-bar_height:, :width_l, 1:] = 0
    # Center is green
    rgb_img[-bar_height:, width_l:width_l + width_f, ::2] = 0
    # Right is blue
    rgb_img[-bar_height:, width_l + width_f:, :2] = 0
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


def _create_image_grid(num_elements: int) -> ImageGrid:
    cols = min(10, num_elements)
    rows = math.ceil(num_elements / cols)
    fig = plt.figure(1, (cols * 2, rows * 2))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0)
    return grid


def _min_max_normalize(a: np.ndarray) -> np.ndarray:
    a_min = np.min(a)
    a_max = np.max(a)
    return (a - a_min) / (a_max - a_min)


def draw_decisions(model: keras.models.Model, experiences: List[Experience]):
    grid = _create_image_grid(len(experiences))

    for i, (img, probabilities) in enumerate(images_with_probabilities(model, experiences)):
        img = _min_max_normalize(img)
        img = add_decision_boundaries(img, probabilities)
        grid[i].imshow(img, origin='lower')


def draw_experiences(experiences: List[Experience]):
    grid = _create_image_grid(len(experiences))

    for axes, experience in zip(grid, experiences):
        img = _min_max_normalize(experience.from_state.data[..., -1])
        axes.imshow(img, cmap='gray', origin='lower')


def save_decisions(model: keras.models.Model, experiences: Iterable[Experience]):
    for i, (img, probabilities) in enumerate(images_with_probabilities(model, experiences)):
        img = _min_max_normalize(img)
        img = add_decision_boundaries(img, probabilities)
        plt.imsave('decisions/d{}.png'.format(i), img, origin='lower')
