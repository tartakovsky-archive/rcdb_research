import numpy as np
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap


def get_default_colormap():
    colors = [hsv_to_rgb((x, 0.7, 0.95)) for x in np.linspace(0.0, 1.0, 101)]
    cmap = LinearSegmentedColormap.from_list('default_cmap', colors)
    return cmap


class Style():
    def __init__(self, tick_size=12, label_size=14, fill=False, fill_alpha=0.2,
                 show_x=True, show_y=True, percent=False, fig_size=(16, 5), dpi=150):

        self.tick_size = tick_size
        self.label_size = label_size
        self.fill = fill
        self.fill_alpha = fill_alpha
        self.show_x = show_x
        self.show_y = show_y
        self.fig_size = fig_size
        self.percent = percent
        self.dpi = dpi


colormap = get_default_colormap()
