import cv2
import numpy as np
import torch
from torchvision.transforms import CenterCrop


class AddCircles:
    """
    Adds concentric circles on the images
    """

    def __init__(self, max_num_circles, circle_width, alpha, color):
        self.max_num_circles = max_num_circles
        self.circle_width = circle_width
        self.alpha = alpha
        self.color = color

    def __call__(self, sample):
        X, Y = sample['X'], sample['Y']
        center_x, center_y = int(X.shape[1] // 2), int(X.shape[0] // 2)

        n_circles = np.random.randint(0, self.max_num_circles)

        X_output = np.copy(X)
        X_overlay = np.copy(X)

        for i in range(n_circles):
            radius = np.random.randint(0, min(X.shape[0], X.shape[1]))
            X_overlay = cv2.circle(X_overlay, (center_x, center_y), radius, self.color,
                                 self.circle_width, lineType=cv2.LINE_AA)
        cv2.addWeighted(X_overlay, self.alpha, X_output, 1 - self.alpha, 0, X_output)

        return {'X': X_output,
                'Y': Y}


class RandomCrop:
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        X, Y = sample['X'], sample['Y']

        h, w = X.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        X = X[top: top + new_h,
             left: left + new_w]

        Y = Y[top: top + new_h,
             left: left + new_w]

        return {'X': X,
                'Y': Y}

class ToFloat:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, Y = sample['X'], sample['Y']

        X = X / 255
        Y = Y / 255

        return {'X': X,
                'Y': Y}

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        X, Y = sample['X'], sample['Y']

        if len(X.shape) == 2:
            X = X[:, :, None]
            Y = Y[:, :, None]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        X = X.transpose((2, 0, 1))
        Y = Y.transpose((2, 0, 1))
        return {'X': torch.tensor(X, dtype=torch.float32),
                'Y': torch.tensor(Y, dtype=torch.float32)}


class CenterCropPair:

    def __init__(self, size):

        if not isinstance(size, tuple):
            size = (size, size)

        self.size = size

    def __call__(self, sample):

        X, Y = sample["X"], sample["Y"]
        cropper = CenterCrop(self.size)
        X = cropper(X)
        Y = cropper(Y)

        return {"X" : X,
                "Y" : Y}