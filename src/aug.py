import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


def process_img(image, aug, height=4, width=4, splitter_width=0):
    images = []
    for _ in range(height * width):
        try:
            images.append(aug(dict(image=image))['image'])
        except:
            try:
                images.append(aug(image=image)['image'])
            except Exception as e:
                images.append(np.zeros((64, 64, 3), dtype=int))
        if splitter_width > 0:
            images.append(np.zeros((images[-1].shape[0], splitter_width, 3), dtype=int))
    rows = []
    if splitter_width > 0:
        width *= 2
    for row_index in range(height):
        row = np.concatenate(images[row_index * width : (row_index + 1) * width], axis=1)
        rows.append(row)
        if splitter_width > 0:
            rows.append(np.zeros((splitter_width, rows[-1].shape[1], 3), dtype=int))
    return np.concatenate(rows, axis=0)


class RandomTileAug(ImageOnlyTransform):
    def __init__(self, aug, height=4, width=4, splitter_width=1, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.aug = aug
        self.height = height
        self.width = width
        self.splitter_width = splitter_width

    def apply(self, image, **params):
        return process_img(image, self.aug, self.height, self.width, self.splitter_width)
