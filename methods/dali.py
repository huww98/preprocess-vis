from pathlib import Path
import numpy as np
from nvidia.dali import pipeline_def
from nvidia.dali import fn, types
from typing import Callable, List
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from PIL import Image


class DaliPreprocess:
    def __init__(self, **kwargs):
        self.fn_kwargs = kwargs

    def _fn(self, images):
        raise NotImplementedError()

    def __call__(self, path: Path) -> np.ndarray:
        @pipeline_def
        def dali_pipeline():
            jpegs, labels = fn.readers.file(files=[str(path)])
            images = fn.decoders.image(jpegs)
            images = self._fn(images, **self.fn_kwargs)
            return images,
        pipe = dali_pipeline(batch_size=1, num_threads=1, device_id=0)
        pipe.build()
        images, = pipe.run()
        return images.at(0)

class ColorJitter(DaliPreprocess):
    _fn = staticmethod(fn.color_twist)

class GrayScale(DaliPreprocess):
    def _fn(self, images):
        return fn.hsv(images, saturation=0.)

class Solorize(DaliPreprocess):
    def _fn(self, images, threshold=128):
        threshold = int(threshold)

        inverted_images = 255 - images
        mask = images >= threshold
        neg_mask = True ^ mask
        out = mask * inverted_images + neg_mask * images
        return fn.cast(out, dtype=types.UINT8)

class GaussianBlur(DaliPreprocess):
    _fn = staticmethod(fn.gaussian_blur)

def main():
    p = ColorJitter()
    img = p('/mnt/cephfs/dataset/imagenet/val/n01440764/ILSVRC2012_val_00029930.JPEG')
    img = Image.fromarray(img)
    img.save('test.jpg')


if __name__ == '__main__':
    main()
