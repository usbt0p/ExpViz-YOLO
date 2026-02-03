import os
import glob
import yaml
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
from bokeh.layouts import gridplot
from bokeh.io import curdoc
import warnings

from commons import get_color_map, style_plot

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

    # Search for all YAML files in any experiments*/list folder
    # This allows for experiments_yolo26, experiments_yolo25, etc.

    search_path = os.path.join(list_dir, "*.yaml")
    yaml_files = glob.glob(search_path)

    if not yaml_files:
        print(f"No experiment configurations found in {search_path}")
        return pd.DataFrame()

    # TODO separate this into a function
    # Load Inference Times from CSV
    # Assumes combined_results.csv is in the parent of the train_dir (i.e. experiments/combined_results.csv)
    experiments_root = os.path.dirname(train_dir)
    csv_path = os.path.join(experiments_root, "combined_results.csv")
    latency_map = {}

    if os.path.exists(csv_path):
        try:
            print(f"Loading inference times from {csv_path}")
            latency_df = pd.read_csv(csv_path)
            # Ensure column names are clean
            latency_df.columns = latency_df.columns.str.strip()
            # Create a map: model -> Inference_ms_im
            if (
                "model" in latency_df.columns
                and "Inference_ms_im" in latency_df.columns
            ):
                latency_map = latency_df.set_index("model")["Inference_ms_im"].to_dict()
            else:
                print(
                    f"Warning: Expected columns 'model' and 'Inference_ms_im' not found in {csv_path}"
                )
                print(f"Available columns: {latency_df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading inference time CSV: {e}")
    else:
        print(f"Warning: Inference time CSV not found at {csv_path}. Using fallbacks.")

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

            # Get Latency: Try exact match, or fallback to size-based default
            # precise matching logic might be needed if exp_name doesn't match model name exactly
            if exp_name in latency_map:
                inference_time = latency_map[exp_name]
            else:
                warnings.warn(f"Latency not found for {exp_name}, skipping.")
                inference_time = None

            # We don't have std in the CSV, so setting to 0 or None
            inference_std = 0.0

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
    # TODO sort by maAP50
    experiments_dataframe = experiments_dataframe.sort_values(by="mAP50", ascending=False)
    series_list = experiments_dataframe["Series"].unique()
    color_map = get_color_map(series_list)

    plots = []

    for m in metrics:
        p = figure(
            title=f"{m['title']} vs Latency",
            width_policy="max",
            height=400,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        style_plot(p, "Latency (ms)", m["y_axis"], m["title"])

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

            all_renderers.append(line)
            all_renderers.append(scatter_fill)
            all_renderers.append(scatter_outline)
            legend_items.append(
                LegendItem(label=series_name, renderers=[line, scatter_fill, scatter_outline])
            )

        # Add Tooltips - ensure we target ALL renderers
        hover = HoverTool(
            renderers=all_renderers,
            tooltips=[
                ("Series", "@Series"),
                ("Experiment", "@Experiment"),
                ("Model Size", "@ModelSize"),
                ("Latency", "@InferenceTime{0.00} ms"),
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
