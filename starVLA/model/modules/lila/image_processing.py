"""Explicit RGB resize contract shared by dataset packing and deployment."""
import cv2
import numpy as np
from PIL import Image


def resize_rgb(image, size, resample='bicubic'):
    array = np.asarray(image.convert('RGB') if isinstance(image, Image.Image) else image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError('Expected HxWx3 uint8 RGB')
    size = tuple(map(int, size))
    if resample == 'opencv_linear':
        return Image.fromarray(cv2.resize(array, size, interpolation=cv2.INTER_LINEAR))
    if resample == 'bicubic':
        return Image.fromarray(array).resize(size, Image.Resampling.BICUBIC)
    raise ValueError(f'Unsupported RGB resampling: {resample}')
