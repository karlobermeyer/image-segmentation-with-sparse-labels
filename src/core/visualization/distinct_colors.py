#!/usr/bin/env python3
"""
A palette of distinct colors in hex, RGB, and BGR. Import, respectively,
`distinct_colors_hex`,
`distinct_colors_rgb`, or
`distinct_colors_bgr`.

Run `./distinct_colors.py` to see a plot of the palette.

_References_
Named CSS colors available in Matplotlib
https://matplotlib.org/stable/gallery/color/named_colors.html
"""
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
#import matplotlib.cm as colormaps
#import matplotlib.colors as colorsm
plt.rcParams["figure.figsize"] = 10, 10 # Set default fig size.


# 64 Distinct Colors
distinct_colors_hex: List[str] = \
    ["#000000", "#00FF00", "#0000FF", "#FF0000", "#01FFFE",
     "#FFA6FE", "#FFDB66", "#006401", "#010067", "#95003A",
     "#007DB5", "#FF00F6", "#FFEEE8", "#774D00", "#90FB92",
     "#0076FF", "#D5FF00", "#FF937E", "#6A826C", "#FF029D",
     "#FE8900", "#7A4782", "#7E2DD2", "#85A900", "#FF0056",
     "#A42400", "#00AE7E", "#683D3B", "#BDC6FF", "#263400",
     "#BDD393", "#00B917", "#9E008E", "#001544", "#C28C9F",
     "#FF74A3", "#01D0FF", "#004754", "#E56FFE", "#788231",
     "#0E4CA1", "#91D0CB", "#BE9970", "#968AE8", "#BB8800",
     "#43002C", "#DEFF74", "#00FFC6", "#FFE502", "#620E00",
     "#008F9C", "#98FF52", "#7544B1", "#B500FF", "#00FF78",
     "#FF6E41", "#005F39", "#6B6882", "#5FAD4E", "#A75740",
     "#A5FFD2", "#FFB167", "#009BFF", "#E85EBE"]
distinct_colors_rgb: List[Tuple[int, int, int]] = []
for s in distinct_colors_hex:
    s: str = s.lstrip("#")
    rgb: Tuple[int, int, int] = tuple(int(s[i:i+2], 16) for i in (0, 2 ,4))
    distinct_colors_rgb.append(rgb)
distinct_colors_rgb: np.ndarray = np.array(distinct_colors_rgb, dtype=np.uint8)
distinct_colors_rgb.flags.writeable = False
distinct_colors_bgr: np.ndarray = np.fliplr(distinct_colors_rgb)
distinct_colors_bgr.flags.writeable = False


def main():
    """Plot the color palette."""
    print("len(distinct_colors_hex) =", len(distinct_colors_hex))

    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.0])
    for i, color_hex in enumerate(distinct_colors_hex):
        plt.plot(x, y+i*1.0, color=color_hex, linewidth=4.0)
    plt.title("Distinct Colors Palette")
    plt.ylabel("index")
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    main()
