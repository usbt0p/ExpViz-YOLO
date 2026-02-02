import os
import glob
import yaml
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
from bokeh.layouts import gridplot
from bokeh.palettes import Turbo256, Category10, Category20
from bokeh.io import curdoc

# Set random seed for reproducibility of mocked data
np.random.seed(42)

# Mock Inference Time Config (Mean ms, Std ms)
# Model sizes: n (nano), s (small), m (medium), l (large), x (extra large)

# TODO the real latency comes from the result of ultralytics.benchmark and comes as a csv file most likely, 
# exact dir structure still tbd 
LATENCY_CONFIG = {
    "n": {"mean": 1.5, "std": 0.1},
    "s": {"mean": 3.2, "std": 0.2},
    "m": {"mean": 6.5, "std": 0.4},
    "l": {"mean": 11.5, "std": 0.8},
    "x": {"mean": 18.0, "std": 1.2},
}

# Mapping size to an order for sorting in line plots
SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}


def load_experiments_data(list_dir, train_dir):
    """
    Reads experiment configurations from any 'experiments*/list' folder and
    corresponding results from the sibling 'train' folder.

    Args:
        list_dir (str): Directory containing experiment configuration files. Assumes the list
            directory provided contains the `.yaml` files for each experiment.
        train_dir (str): Directory containing experiment training results. Assumes the train
            directory provided contains the subdirectories with each of the list_dir experiment names, and 
            their respective `results.csv` files.

    Returns:
        pd.DataFrame: DataFrame containing training history for all experiments.
    """
    data = []

    # Search for all YAML files in any experiments*/list folder
    # This allows for experiments_yolo26, experiments_yolo25, etc.

    # TODO change this to take the dir as an argument, assume just above .../list and .../train
    search_path = os.path.join(list_dir, "*.yaml")
    yaml_files = glob.glob(search_path)

    if not yaml_files:
        print(f"No experiment configurations found in {search_path}")
        # Fallback to specific check if glob fails or structure is different
        return pd.DataFrame()

    print(f"Found {len(yaml_files)} experiment configurations.")

    for yf in yaml_files:
        try:
            with open(yf, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error reading {yf}: {e}")
            continue

        exp_name = config.get("experiment_name")
        if not exp_name:
            continue

        # Extract experiment metadata
        model_size = config.get("model_size", "Unknown")
        is_pretrained = config.get("pretrained", False)
        model_version = config.get("model_version", "Unknown")
        freezing_strategy = config.get("freeze_layers", None)

        # Determine Series Name
        # the naming of the series is key to it's managing: we'll plot all series with unique names
        rect_mode = config.get("rect_mode", False)
        variant = "Pretr." if is_pretrained else "Scratch"
        if freezing_strategy:
            variant += f"   ({freezing_strategy})"
        series_name = f"{model_version.capitalize()} {variant}"
        if rect_mode:
            series_name += " (Rect)"

        results_path = os.path.join(train_dir, exp_name, "results.csv")

        if not os.path.exists(results_path):
            print(f"Results not found for experiment: {exp_name} at {results_path}")
            continue

        try:
            # Read CSV
            df = pd.read_csv(results_path)
            df.columns = df.columns.str.strip()

            col_map = {
                "map50": "metrics/mAP50(B)",
                "map50_95": "metrics/mAP50-95(B)",
                "precision": "metrics/precision(B)",
                "recall": "metrics/recall(B)",
            }

            if not all(c in df.columns for c in col_map.values()):
                continue

            best_idx = df[col_map["map50_95"]].idxmax()
            best_row = df.loc[best_idx]

            # TODO the latency config should be given from the arguments in a csv
            lat_conf = LATENCY_CONFIG.get(model_size, {"mean": 10.0, "std": 1.0})
            inference_time = np.random.normal(lat_conf["mean"], lat_conf["std"])
            inference_std = lat_conf["std"]

            data_entry = {
                "Experiment": exp_name,
                "Series": series_name,
                "ModelSize": model_size,
                "SizeOrder": SIZE_ORDER.get(model_size, 99),
                "InferenceTime": inference_time,
                "InferenceStd": inference_std,
                "mAP50": best_row[col_map["map50"]],
                "mAP50_95": best_row[col_map["map50_95"]],
                "Precision": best_row[col_map["precision"]],
                "Recall": best_row[col_map["recall"]],
                "Epoch": best_row["epoch"],
            }
            data.append(data_entry)

        except Exception as e:
            print(f"Error processing {exp_name}: {e}")

    return pd.DataFrame(data)


def style_plot(p, x_label, y_label, title):
    """
    Applies a dark theme style similar to the provided image.
    """
    # Dark background
    p.background_fill_color = "#151515"
    p.border_fill_color = "#151515"
    p.outline_line_color = None

    # Grid lines
    p.xgrid.grid_line_color = "#444444"
    p.ygrid.grid_line_color = "#444444"

    # Axis labels and ticks
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = y_label
    p.axis.axis_label_text_color = "#aaaaaa"
    p.axis.major_label_text_color = "#aaaaaa"
    p.axis.axis_label_text_font_style = "bold"

    # Title
    p.title.text_color = "#ffffff"
    p.title.text_font_size = "14pt"

    return p


def create_visualization(experiments_dataframe, output_path):
    """
    Creates a visualization of metrics for a given DataFrame, metric VS latency in pareto charts.

    Args:
        experiments_dataframe (pandas.DataFrame): DataFrame containing training history for all experiments.
        output_file (str): Path to save the visualization HTML file.
    
    Returns:
        str: Path to the saved visualization HTML file.
    """
    
    if experiments_dataframe.empty:
        print("No data loaded. Aborting.")
        return

    output_file(output_path, title="Metrics Visualization")

    metrics = [
        {"col": "mAP50", "title": "mAP 50", "y_axis": "mAP@50"},
        {"col": "mAP50_95", "title": "mAP 50-95", "y_axis": "mAP@50:95"},
        {"col": "Precision", "title": "Precision", "y_axis": "Precision"},
        {"col": "Recall", "title": "Recall", "y_axis": "Recall"},
    ]

    # the naming of the series is key to it's managing
    series_list = sorted(experiments_dataframe["Series"].unique())
    # TODO see if this is correct
    l = len(series_list)
    if l <= 10:
        colors = Category10[10]
    elif l <= 20:
        colors = Category20[20]
    else:
        colors = Turbo256[: len(series_list)]
    color_map = {
        series: colors[i % len(colors)] for i, series in enumerate(series_list)
    }

    plots = []

    for m in metrics:
        p = figure(
            title=f"{m['title']} vs Latency",
            width_policy="max",
            height=400,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        style_plot(p, "Latency (ms) [Mocked]", m["y_axis"], m["title"])

        legend_items = []
        all_renderers = []  # Collect renderers for HoverTool

        for series_name in series_list:
            subset = experiments_dataframe[
                experiments_dataframe["Series"] == series_name
                ].sort_values("SizeOrder")

            if subset.empty:
                continue

            source = ColumnDataSource(subset)
            c = color_map[series_name]

            # Draw Line
            line = p.line(
                x="InferenceTime",
                y=m["col"],
                source=source,
                line_width=2,
                color=c,
                alpha=0.8,
            )

            # Draw Points (Scatter)
            # Outer circle for effect
            scatter_outline = p.scatter(
                x="InferenceTime",
                y=m["col"],
                source=source,
                size=12,
                line_color="white",
                fill_color=None,
                line_width=1.5,
            )
            # Inner circle
            scatter_fill = p.scatter(
                x="InferenceTime",
                y=m["col"],
                source=source,
                size=8,
                color=c,
                line_color=None,
            )

            all_renderers.append(scatter_fill)
            legend_items.append(
                LegendItem(label=series_name, renderers=[line, scatter_fill])
            )

        # Add Tooltips - ensure we target ALL renderers
        hover = HoverTool(
            renderers=all_renderers,
            tooltips=[
                ("Series", "@Series"),
                ("Experiment", "@Experiment"),
                ("Model Size", "@ModelSize"),
                ("Latency", "@InferenceTime{0.00} ms (+/- @InferenceStd{0.00})"),
                ("mAP50", "@mAP50{0.000}"),
                ("mAP50-95", "@mAP50_95{0.000}"),
                ("Precision", "@Precision{0.000}"),
                ("Recall", "@Recall{0.000}"),
                ("Epoch", "@Epoch"),
            ],
        )
        p.add_tools(hover)

        legend = Legend(items=legend_items)
        legend.click_policy = "hide"
        legend.label_text_color = "#cccccc"
        legend.background_fill_color = "#202020"
        legend.border_line_color = "#444444"
        p.add_layout(legend, "right")

        plots.append(p)

    # Layout: Grid 2x2
    # [mAP50, mAP50-95]
    # [Precision, Recall]
    grid = gridplot(
        [[plots[0], plots[1]], [plots[2], plots[3]]], sizing_mode="stretch_width"
    )

    out = save(grid)
    print(f"Visualization saved to {out}")
    return out
