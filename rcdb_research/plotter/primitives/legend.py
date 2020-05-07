from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase


class HandlerColormap(HandlerBase):

    def __init__(self, cmap, max_num_stripes, num_stripes, **kw):
        super().__init__(**kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
        self.max_num_stripes = max_num_stripes

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        return [
            Rectangle([xdescent + i * width / self.num_stripes, ydescent],
                      width / self.num_stripes,
                      height,
                      fc=self.cmap(i % self.max_num_stripes),
                      transform=trans)
            for i in range(self.num_stripes)
        ]
