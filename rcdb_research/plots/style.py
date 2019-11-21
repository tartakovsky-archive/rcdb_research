from matplotlib.colors import hsv_to_rgb


class Palette():
    def __init__(
        self,
        blue=hsv_to_rgb((0.56, 0.7, 0.95)),  # 48B5F2
        orange=hsv_to_rgb((0.045, 0.7, 0.95)),  # F27648
    ):
        self.blue = blue
        self.orange = orange


class ColorMap():
    def __init__(
        self,
        positive=Palette().blue,
        negative=Palette().orange,
        train_set=Palette().blue,
        test_set=Palette().orange,
    ):
        self.positive = positive
        self.negative = negative
        self.train_set = train_set
        self.test_set = test_set


class Style():
    def __init__(self, tick_size=12, label_size=14, fill=False, fill_alpha=0.2,
                 show_x=True, show_y=True, percent=False, fig_size=(16, 5),
                 dpi=150):

        self.tick_size = tick_size
        self.label_size = label_size
        self.fill = fill
        self.fill_alpha = fill_alpha
        self.show_x = show_x
        self.show_y = show_y
        self.fig_size = fig_size
        self.percent = percent
        self.dpi = dpi
