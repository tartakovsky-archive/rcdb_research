import sys
from matplotlib.colors import hsv_to_rgb

this = sys.modules['rcdb_research.plots.palette']

this.blue = hsv_to_rgb((0.56, 0.7, 0.95))
this.red = hsv_to_rgb((0.045, 0.7, 0.95))
