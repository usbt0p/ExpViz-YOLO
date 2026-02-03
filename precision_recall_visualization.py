from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
from bokeh.io import curdoc
import pandas as pd

from commons import style_plot, get_color_map


def create_pr_visualization(experiments_dataframe, output_path, top_n=None):
    """
    Creates a scatter plot of Precision vs Recall for provided experiments.

    Args:
        experiments_dataframe (pandas.DataFrame): DataFrame containing best-checkpoint metrics.
        output_path (str): Path to save the visualization HTML file.

    Returns:
        str: Path to the saved visualization HTML file.
    """

    if experiments_dataframe.empty:
        print("No data loaded for PR Viz. Aborting.")
        return

    output_file(output_path, title="Precision vs Recall Tradeoff")

    # Filter Top N
    # We use mAP50_95 for ranking "best" models to filter
    if top_n:
        experiments_dataframe = experiments_dataframe.sort_values(
            by="mAP50_95", ascending=False
        )
        top_series = experiments_dataframe["Series"].unique()[:top_n]
        experiments_dataframe = experiments_dataframe[
            experiments_dataframe["Series"].isin(top_series)
        ]
        print(f"[PR Viz] Filtering top {top_n} series")

    # Colors
    series_list = sorted(experiments_dataframe["Series"].unique())
    color_map = get_color_map(series_list)

    # Create Plot
    p = figure(
        title="Precision vs Recall Tradeoff (Best Checkpoint)",
        width_policy="max",
        height=600,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_axis_label="Recall",
        y_axis_label="Precision",
        x_range=(0, 1.05),  # Recall is 0-1
        y_range=(0, 1.05),  # Precision is 0-1
    )
    style_plot(p, "Recall", "Precision", "Precision vs Recall Tradeoff")

    legend_items = []
    all_renderers = []

    for series_name in series_list:
        subset = experiments_dataframe[
            experiments_dataframe["Series"] == series_name
        ].sort_values("SizeOrder")
        if subset.empty:
            continue

        source = ColumnDataSource(subset)
        c = color_map[series_name]

        # Draw Points (Scatter)
        # Larger scatter for visibility
        scatter = p.scatter(
            x="Recall",
            y="Precision",
            source=source,
            size=15,
            line_color="white",
            fill_color=c,
            line_width=2,
            fill_alpha=0.8,
        )

        # Connect points of the same series with a faint line to show size progression
        p.line(
            x="Recall", y="Precision", source=source, color=c, alpha=0.3, line_width=1
        )

        all_renderers.append(scatter)
        legend_items.append(LegendItem(label=series_name, renderers=[scatter]))

    # Tooltips
    hover = HoverTool(
        renderers=all_renderers,
        tooltips=[
            ("Series", "@Series"),
            ("Model Size", "@ModelSize"),
            ("mAP50", "@mAP50{0.000}"),
            ("Precision", "@Precision{0.000}"),
            ("Recall", "@Recall{0.000}"),
            ("Recall", "@Recall{0.000}"),
            ("Latency", "@InferenceTime{0.00} ms"),
            ("Size", "@Size_MB MB"),
            ("FPS", "@FPS"),
            ("Format", "@Format"),
        ],
    )
    p.add_tools(hover)

    # Legend
    legend = Legend(items=legend_items)
    legend.click_policy = "hide"
    legend.label_text_color = "#cccccc"
    legend.background_fill_color = "#202020"
    legend.border_line_color = "#444444"
    p.add_layout(legend, "right")

    out = save(p)
    print(f"Visualization saved to {out}")
    return out
