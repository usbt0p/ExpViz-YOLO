import os
import glob
import yaml
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
from bokeh.layouts import gridplot
from bokeh.palettes import Turbo256, Category10
from bokeh.io import curdoc

# Set random seed for reproducibility of mocked data
np.random.seed(42)

# --- Configuration ---
# Paths assuming script is run from project root
base_dir = os.path.dirname(os.path.abspath(__file__))
EXP_LIST_DIR = os.path.join(base_dir, "experiments_yolo26/list")
EXP_TRAIN_DIR = os.path.join(base_dir, "experiments_yolo26/train")
OUTPUT_FILE = "metrics_visualization.html"

# Mock Inference Time Config (Mean ms, Std ms)
# Model sizes: n (nano), s (small), m (medium), l (large), x (extra large)
# These are rough estimates relative to each other for visualization purposes
LATENCY_CONFIG = {
    "n": {"mean": 1.5, "std": 0.1},
    "s": {"mean": 3.2, "std": 0.2},
    "m": {"mean": 6.5, "std": 0.4},
    "l": {"mean": 11.5, "std": 0.8},
    "x": {"mean": 18.0, "std": 1.2},
}

# Mapping size to an order for sorting in line plots
SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}


def load_experiments_data():
    """
    Reads experiment configurations from the list folder and corresponding results
    from the train folder. Mocks latency data.
    """
    data = []

    # Get all YAML config files
    yaml_files = glob.glob(os.path.join(EXP_LIST_DIR, "*.yaml"))

    if not yaml_files:
        print(f"No experiment configurations found in {EXP_LIST_DIR}")
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
        model_size = config.get("model_size", "n")
        is_pretrained = config.get("pretrained", False)

        # Define Series Name for grouping (e.g., "YOLO26 Scratch", "YOLO26 Pretrained")
        # We assume 'rect_mode' might be another variant, adding it if needed.
        rect_mode = config.get("rect_mode", False)
        variant = "Pretrained" if is_pretrained else "Scratch"
        series_name = f"YOLO26 {variant}"
        if rect_mode:
            series_name += " (Rect)"

        # Locate results.csv
        results_path = os.path.join(EXP_TRAIN_DIR, exp_name, "results.csv")
        if not os.path.exists(results_path):
            print(f"Results not found for experiment: {exp_name}")
            continue

        try:
            # Read CSV
            df = pd.read_csv(results_path)
            # Clean column names (remove potential leading/trailing spaces)
            df.columns = df.columns.str.strip()

            # Identify metric columns
            # Expected: metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
            col_map = {
                "map50": "metrics/mAP50(B)",
                "map50_95": "metrics/mAP50-95(B)",
                "precision": "metrics/precision(B)",
                "recall": "metrics/recall(B)",
            }

            # verify columns exist
            if not all(c in df.columns for c in col_map.values()):
                print(
                    f"Missing expected columns in {exp_name}. Available: {df.columns.tolist()}"
                )
                continue

            # Select the Best Epoch based on mAP50-95
            best_idx = df[col_map["map50_95"]].idxmax()
            best_row = df.loc[best_idx]

            # Mock Inference Time
            # Generate a random value based on the configuration mean/std
            lat_conf = LATENCY_CONFIG.get(model_size, {"mean": 10.0, "std": 1.0})
            inference_time = np.random.normal(lat_conf["mean"], lat_conf["std"])
            inference_std = lat_conf["std"]

            # Append to data list
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


def create_visualization():
    df = load_experiments_data()

    if df.empty:
        print("No data loaded. Aborting.")
        return

    # Prepare output
    output_file(OUTPUT_FILE, title="Metrics Visualization")

    # Metrics to visualize
    metrics = [
        {"col": "mAP50", "title": "mAP 50", "y_axis": "mAP 50"},
        {"col": "mAP50_95", "title": "mAP 50-95", "y_axis": "mAP 50-95"},
        {"col": "Precision", "title": "Precision", "y_axis": "Precision"},
        {"col": "Recall", "title": "Recall", "y_axis": "Recall"},
    ]

    # Get unique series for consistent coloring
    series_list = sorted(df["Series"].unique())
    # Use a palette. Category10_10 has 10 colors.
    colors = Category10[10] if len(series_list) <= 10 else Turbo256[: len(series_list)]
    color_map = {
        series: colors[i % len(colors)] for i, series in enumerate(series_list)
    }

    plots = []

    for m in metrics:
        # Create figure
        p = figure(
            title=f"{m['title']} vs Latency",
            width_policy="max",
            height=400,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        style_plot(p, "Latency (ms) [Mocked]", m["y_axis"], m["title"])

        # Store renderers to create legend later
        legend_items = []

        # Plot each series
        for series_name in series_list:
            subset = df[df["Series"] == series_name].sort_values("SizeOrder")

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

            # Draw Points (Circle) for tooltips and visibility
            # Use distinct outer circle for 'pop' effect like in image
            circle_outline = p.circle(
                x="InferenceTime",
                y=m["col"],
                source=source,
                size=12,
                color="white",
                fill_alpha=0,
            )
            circle_fill = p.circle(
                x="InferenceTime", y=m["col"], source=source, size=8, color=c
            )

            legend_items.append(
                LegendItem(label=series_name, renderers=[line, circle_fill])
            )

        # Add Tooltips
        # Shows results of ALL other metrics as requested
        hover = HoverTool(
            renderers=[circle_fill],
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

        # Improve Legend
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

    save(grid)
    print(f"Visualization saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    create_visualization()
