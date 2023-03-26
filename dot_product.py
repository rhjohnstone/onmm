import lightning as L


class DotProductWork(L.LightningWork):
    def __init__(self, xs, ys):
        super().__init__(parallel=True)
        self.xs = xs
        self.ys = ys
        self.result = None

    def run(self):
        if self.result is None:
            self.result = sum(x * y for x, y in zip(self.xs, self.ys))
