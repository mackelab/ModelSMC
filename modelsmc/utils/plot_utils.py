import os

import matplotlib as mpl
import matplotlib.pyplot as plt

_custom_styles = ["pyloric"]
_mpl_styles = list(plt.style.available)

# Style files live in figures/ (shared with the figure-generation scripts) rather
# than being duplicated here.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
FIGURES_DIR = os.path.join(_REPO_ROOT, "figures")


def get_style(style, **kwargs):
    if style in _mpl_styles:
        return [style]
    elif style in _custom_styles:
        return [os.path.join(FIGURES_DIR, style + ".mplstyle")]
    elif style == "science":
        return ["science"]
    elif style == "science_grid":
        return ["science", {"axes.grid": True}]
    elif style is None:
        return None
    else:
        return style


class use_style:
    def __init__(self, style, kwargs=None) -> None:
        if kwargs is None:
            kwargs = {}
        super().__init__()
        self.style = get_style(style) + [kwargs]
        self.previous_style = {}

    def __enter__(self):
        self.previous_style = mpl.rcParams.copy()
        if self.style is not None:
            plt.style.use(self.style)

    def __exit__(self, *args, **kwargs):
        mpl.rcParams.update(self.previous_style)
