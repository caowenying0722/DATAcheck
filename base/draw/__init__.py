from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray


class Base2D:
    x: NDArray
    y: NDArray
    title: str
    x_label: str
    y_label: str

    fig: Figure | None = None

    def __init__(
        self,
        x: NDArray | list,
        y: NDArray | list,
        title: str = "",
        x_label: str = "x",
        y_label: str = "y",
    ):
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.fig = None

    def _draw_inner(self) -> Figure: ...

    def draw(self):
        if self.fig is None:
            fig = self._draw_inner()
            self.fig = fig
        fig = self.fig
        plt.tight_layout()
        return fig

    def show(self, show=True):
        self.draw()
        if show:
            plt.show()
        return self

    def save(self, dir: Path, dpi=300):
        fig = self.draw()

        save_path = dir / f"{self.title.replace(' ', '_')}.png"
        fig.savefig(save_path, dpi=dpi)
        print(f"Saved to {save_path}")
        plt.close(fig)

        return self
