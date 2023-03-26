import itertools as it

import lightning as L
from lightning.app.structures import List

from dot_product import DotProductWork


def reshape(x, shape):
    rows, cols = shape
    assert rows * cols == len(x)
    return [x[start : start + cols] for start in range(0, len(x), cols)]


class MatMulFlow(L.LightningFlow):
    def __init__(self, xss, yss):
        super().__init__()
        self.dot_product_works = List(
            *[DotProductWork(xs, ys) for xs, ys in it.product(xss, yss)]
        )
        self.shape = (len(xss), len(yss))
        self.result = None

    def run(self):
        for work in self.dot_product_works:
            work.run()
        if (self.result is None) and all(
            work.result is not None for work in self.dot_product_works
        ):
            self.result = reshape(
                [work.result for work in self.dot_product_works], self.shape
            )
            for work in self.dot_product_works:
                work.stop()
