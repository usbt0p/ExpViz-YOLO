import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import Turbo, Category10, Category20

def style_plot(p, x_label, y_label, title):
    p.background_fill_color = "#151515"
    p.border_fill_color = "#151515"
    p.outline_line_color = None
    p.xgrid.grid_line_color = "#444444"
    p.ygrid.grid_line_color = "#444444"
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.axis.axis_label_text_color = "#aaaaaa"
    p.axis.major_label_text_color = "#aaaaaa"
    p.axis.axis_label_text_font_style = "bold"
    p.title.text_color = "#ffffff"
    p.title.text_font_size = "14pt"
    return p

def get_color_map(categories):
    """
    Returns a dict mapping categories to colors.
    Switches: Category10 (<=10) -> Category20 (<=20) -> Spaced Turbo (>20).
    """
    # Ensure unique and sorted to maintain consistency
    items = sorted(list(set(categories)))
    n = len(items)
    #n = len(categories)

    if n <= 10:
        colors = Category10[10][:n]
    elif n <= 20:
        colors = Category20[20][:n]
    else:
        # Spaced sampling from Turbo's 256 colors
        indices = np.linspace(0, 255, n, dtype=int)
        colors = [Turbo[256][i] for i in indices]

    return dict(zip(categories, colors))