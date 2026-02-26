import os
import glob
import yaml
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Legend,
    LegendItem,
    RadioGroup,
    CustomJS,
)
from bokeh.layouts import gridplot, row, column
from bokeh.io import curdoc
import warnings

from commons import get_color_map, style_plot

# Mapping size to an order for sorting in line plots
SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}


def load_inference_metadata(experiments_root, csv_name="combined_results.csv"):
    """
    Loads inference metadata (latency, size, fps, format) from combined_results.csv.
    """
    csv_path = os.path.join(experiments_root, csv_name)
    latency_map = {}
    size_map = {}
    fps_map = {}
    format_map = {}

    if os.path.exists(csv_path):
        try:
            print(f"Loading inference times from {csv_path}")
            latency_df = pd.read_csv(csv_path)
            # Ensure column names are clean
            latency_df.columns = latency_df.columns.str.strip()

            # Create maps for additional fields
            if "model" in latency_df.columns:
                latency_map = (
                    latency_df.set_index("model")["Inference_ms_im"].to_dict()
                    if "Inference_ms_im" in latency_df.columns
                    else {}
                )
                size_map = (
                    latency_df.set_index("model")["Size_MB"].to_dict()
                    if "Size_MB" in latency_df.columns
                    else {}
                )
                fps_map = (
                    latency_df.set_index("model")["FPS"].to_dict()
                    if "FPS" in latency_df.columns
                    else {}
                )
                format_map = (
                    latency_df.set_index("model")["Format"].to_dict()
                    if "Format" in latency_df.columns
                    else {}
                )
            else:
                warnings.warn(f"'model' column not found in {csv_path}")
        except Exception as e:
            warnings.warn(f"Error reading inference time CSV: {e}")
    else:
        warnings.warn(f"Inference time CSV not found at {csv_path}.")

    return latency_map, size_map, fps_map, format_map


def load_experiments_data(list_dir, train_dir, results_csv="combined_results.csv"):
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

    # Load Inference Times from CSV
    # Assumes combined_results.csv is in the parent of the train_dir (i.e. experiments/combined_results.csv)
    experiments_root = os.path.dirname(train_dir)
    latency_map, size_map, fps_map, format_map = load_inference_metadata(
        experiments_root, results_csv
    )

    print(f"Found {len(yaml_files)} experiment configurations.")

    for yf in yaml_files:
        try:
            with open(yf, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            warnings.warn(f"Error reading {yf}: {e}")
            continue

        exp_name = config.get("experiment_name")
        if not exp_name:
            continue

        # Extract experiment metadata
        model_size = config.get("model_size", "Unknown")
        if model_size == "Unknown":
            warnings.warn(f"Unknown model size for experiment: {exp_name}")
        is_pretrained = config.get("pretrained", False)
        model_version = config.get("model_version", "Unknown")
        freezing_strategy = config.get("freeze_layers", None)

        # this has to support all naming differences and ensure there's no collisions in order
        # for the legend to work properly

        variant = "Pretr." if is_pretrained else "Scratch"
        if freezing_strategy:
            variant += f" {freezing_strategy}"
        series_name = f"{model_version.capitalize()} {variant}"

        rect_mode = config.get("rect_mode", False)
        if rect_mode:
            series_name += " rect"

        # TODO: make this more robust, like reading from the yaml instead of parsing the name
        # there is an old and a new naming style
        # for ex: yolo26m_freeze-backbone_rectFalse_size512
        # old one is: yolo26s_pretrained_full_rectFalse_size800
        # we need to get the additional differentiators like size800 that the new one added
        diff = exp_name.split("rect")[1].split("_")[1]
        series_name += f" {diff.replace("size", "sz")}"

        results_path = os.path.join(train_dir, exp_name, "results.csv")

        if not os.path.exists(results_path):
            warnings.warn(
                f"Results not found for experiment: {exp_name} at {results_path}"
            )
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
            inference_time = latency_map.get(exp_name)
            if inference_time is None:
                warnings.warn(f"Latency not found for {exp_name}, skipping.")
                inference_time = None

            # Get other metadata
            size_mb = size_map.get(exp_name, "N/A")
            fps = fps_map.get(exp_name, None)
            fmt = format_map.get(exp_name, "N/A")

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
                "Size_MB": size_mb,
                "FPS": fps,
                "Format": fmt,
            }
            data.append(data_entry)

        except Exception as e:
            warnings.warn(f"Error processing {exp_name}: {e}")

    return pd.DataFrame(data)


def create_grid_plots(df, metrics, series_list, color_map, x_col, x_label):
    plots = []
    renderer_map = {}

    for m in metrics:
        p = figure(
            title=f"{m['title']} vs {x_label.split(' ')[0]}",
            width_policy="max",
            height=400,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        style_plot(p, x_label, m["y_axis"], m["title"])

        all_renderers = []
        for series_name in series_list:
            subset = df[df["Series"] == series_name].sort_values("SizeOrder")

            if subset.empty:
                continue

            source = ColumnDataSource(subset)
            c = color_map[series_name]

            # Draw Line
            line = p.line(
                x=x_col,
                y=m["col"],
                source=source,
                line_width=2,
                color=c,
                alpha=0.8,
            )

            # Draw Points (Scatter)
            # Outer circle for effect
            scatter_outline = p.scatter(
                x=x_col,
                y=m["col"],
                source=source,
                size=12,
                line_color="white",
                fill_color=None,
                line_width=1.5,
            )
            # Inner circle
            scatter_fill = p.scatter(
                x=x_col,
                y=m["col"],
                source=source,
                size=8,
                color=c,
                line_color=None,
            )

            all_renderers.append(line)
            all_renderers.append(scatter_fill)
            all_renderers.append(scatter_outline)

            # Collect for global legend
            if series_name not in renderer_map:
                renderer_map[series_name] = []
            renderer_map[series_name].extend([line, scatter_fill, scatter_outline])

        # Add Tooltips - transparent background
        hover = HoverTool(
            renderers=all_renderers,
            tooltips="""
            <div style="background-color: rgba(32, 32, 32, 0.7); padding: 10px; border: 1px solid #444; border-radius: 5px;">
                <div style="color: #fff; font-weight: bold; margin-bottom: 5px;">@Series (@ModelSize)</div>
                <div style="color: #aaa; font-size: 0.9em;">
                    Latency: <span style="color: #eee;">@InferenceTime{0.00} ms</span><br>
                    mAP50: <span style="color: #eee;">@mAP50{0.000}</span><br>
                    mAP50_95: <span style="color: #eee;">@mAP50_95{0.000}</span><br>
                    Precision: <span style="color: #eee;">@Precision{0.000}</span><br>
                    Recall: <span style="color: #eee;">@Recall{0.000}</span><br>
                    FPS: <span style="color: #eee;">@FPS</span> | Format: <span style="color: #eee;">@Format</span>
                </div>
            </div>
            """,
        )
        p.add_tools(hover)
        plots.append(p)

    return (
        gridplot(
            [[plots[0], plots[1]], [plots[2], plots[3]]], sizing_mode="stretch_width"
        ),
        renderer_map,
    )


def create_metrics_visualization(experiments_dataframe, output_path, top_n=None):
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
        with open(output_path, "w", encoding="utf-8") as err_html:
            err_html.write(
                "<html><body style='display:flex;justify-content:center;align-items:center;height:100vh;font-family:'Georgia', 'Times New Roman', serif;color:#666;'><div><h2>No data loaded</h2><p>The metrics dataframe is empty.</p></div></body></html>"
            )
        return

    output_file(output_path, title="Metrics Visualization")

    metrics = [
        {"col": "mAP50", "title": "mAP50", "y_axis": "mAP50"},
        {"col": "mAP50_95", "title": "mAP50_95", "y_axis": "mAP50_95"},
        {"col": "Precision", "title": "Precision", "y_axis": "Precision"},
        {"col": "Recall", "title": "Recall", "y_axis": "Recall"},
    ]

    # Sort and Filter Top N
    experiments_dataframe = experiments_dataframe.sort_values(
        by="mAP50_95", ascending=False
    )

    if top_n:
        top_series = experiments_dataframe["Series"].unique()[:top_n]
        experiments_dataframe = experiments_dataframe[
            experiments_dataframe["Series"].isin(top_series)
        ]
        print(f"Filtering top {top_n} series: {top_series}")

    series_list = experiments_dataframe["Series"].unique()
    color_map = get_color_map(series_list)

    grid_fps, map_fps = create_grid_plots(
        experiments_dataframe,
        metrics,
        series_list,
        color_map,
        x_col="FPS",
        x_label="FPS",
    )

    grid_lat, map_lat = create_grid_plots(
        experiments_dataframe,
        metrics,
        series_list,
        color_map,
        x_col="InferenceTime",
        x_label="Latency (ms)",
    )
    grid_fps.visible = False

    # Construct Global Legend
    height = 50 + len(series_list) * 20
    p_legend = figure(
        width=250,
        height=max(800, height),
        toolbar_location=None,
        outline_line_color=None,
        x_range=(0, 1),
        y_range=(0, 1),
    )
    p_legend.background_fill_color = "#151515"
    p_legend.border_fill_color = "#151515"
    p_legend.xaxis.visible = False
    p_legend.yaxis.visible = False
    p_legend.grid.visible = False

    legend_items = []
    for series_name in series_list:
        c = color_map[series_name]
        d_line = p_legend.line(x=[2], y=[2], color=c, line_width=2)
        d_scatter = p_legend.scatter(x=[2], y=[2], color=c, size=8)

        real_renderers = map_fps.get(series_name, []) + map_lat.get(series_name, [])
        legend_items.append(
            LegendItem(
                label=series_name, renderers=real_renderers + [d_line, d_scatter]
            )
        )

    n_cols = 1 if len(series_list) <= 30 else 2
    legend = Legend(items=legend_items, ncols=n_cols)
    legend.click_policy = "hide"
    legend.label_text_color = "#cccccc"
    legend.label_text_font_size = "8pt"
    legend.spacing = 1
    legend.background_fill_color = "#202020"
    legend.border_line_color = "#444444"
    legend.location = "top_left"
    p_legend.add_layout(legend)

    # Toggle Control
    radio_group = RadioGroup(
        labels=["FPS", "Latency"],
        active=1,
        inline=True,
    )

    # JavaScript Callback
    callback = CustomJS(
        args=dict(g_fps=grid_fps, g_lat=grid_lat),
        code="""
        g_fps.visible = false;
        g_lat.visible = false;
        
        if (cb_obj.active == 0) {
            g_fps.visible = true;
        } else {
            g_lat.visible = true;
        }
    """,
    )
    radio_group.js_on_change("active", callback)

    # Layout
    grids_col = column(grid_fps, grid_lat, sizing_mode="stretch_width")
    layout = column(
        row(radio_group, sizing_mode="fixed", width=450),
        row(grids_col, p_legend, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    # Final styling
    curdoc().theme = "dark_minimal"
    out = save(layout)

    print(f"Visualization saved to {out}")
    return out
