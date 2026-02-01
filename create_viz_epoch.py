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
    Column,
    Row,
)
from bokeh.layouts import gridplot, column, row
from bokeh.palettes import Turbo256, Category10
from bokeh.io import curdoc

# Import styles and config from existing script
# We will duplicate load_experiments_data behavior but adjusted for full history (similar to what I tried before)
# Since the original load_experiments_data only returns one row per experiment, I need to implement a full history loader here.
# I will copy the helper functions to avoid breaking the original script.

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = "training_metrics_visualization.html"

LATENCY_CONFIG = {
    "n": {"mean": 1.5, "std": 0.1},
    "s": {"mean": 3.2, "std": 0.2},
    "m": {"mean": 6.5, "std": 0.4},
    "l": {"mean": 11.5, "std": 0.8},
    "x": {"mean": 18.0, "std": 1.2},
}

SIZE_ORDER = {"n": 0, "s": 1, "m": 2, "l": 3, "x": 4}


def load_training_history():
    """
    Reads full training history for all experiments.
    """
    data_frames = []
    search_path = os.path.join(base_dir, "experiments*", "list", "*.yaml")
    yaml_files = glob.glob(search_path)

    if not yaml_files:
        print(f"No experiment configurations found in {search_path}")
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

        model_size = config.get("model_size", "n")
        is_pretrained = config.get("pretrained", False)
        rect_mode = config.get("rect_mode", False)

        # Determine path
        exp_root = os.path.dirname(os.path.dirname(yf))
        train_dir = os.path.join(exp_root, "train")
        results_path = os.path.join(train_dir, exp_name, "results.csv")

        if not os.path.exists(results_path):
            continue

        try:
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

            # Series Label for plot
            # e.g. "26n Scratch"
            variant = "Pret." if is_pretrained else "Scratch"
            clean_name = f"YOLO26{model_size} {variant}"
            if rect_mode:
                clean_name += " Rect"

            lat_conf = LATENCY_CONFIG.get(model_size, {"mean": 10.0, "std": 1.0})
            inference_time = np.random.normal(lat_conf["mean"], lat_conf["std"])

            df_exp = df.copy()
            df_exp["Experiment"] = exp_name
            df_exp["Label"] = clean_name
            df_exp["InferenceTime"] = inference_time

            df_exp.rename(
                columns={
                    col_map["map50"]: "mAP50",
                    col_map["map50_95"]: "mAP50_95",
                    col_map["precision"]: "Precision",
                    col_map["recall"]: "Recall",
                    "epoch": "Epoch",
                },
                inplace=True,
            )

            keep_cols = [
                "Experiment",
                "Label",
                "InferenceTime",
                "mAP50",
                "mAP50_95",
                "Precision",
                "Recall",
                "Epoch",
            ]
            data_frames.append(df_exp[keep_cols])

        except Exception as e:
            print(f"Error processing {exp_name}: {e}")

    if not data_frames:
        return pd.DataFrame()

    return pd.concat(data_frames, ignore_index=True)


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


def create_grid(df, axis_type="linear"):
    """
    Creates a 2x2 grid of plots.
    """
    metrics = [
        {"col": "mAP50", "title": f"mAP 50 ({axis_type})", "y_axis": "mAP 50"},
        {"col": "mAP50_95", "title": f"mAP 50-95 ({axis_type})", "y_axis": "mAP 50-95"},
        {
            "col": "Precision",
            "title": f"Precision ({axis_type})",
            "y_axis": "Precision",
        },
        {"col": "Recall", "title": f"Recall ({axis_type})", "y_axis": "Recall"},
    ]

    plots = []

    # Colors
    exp_list = sorted(df["Experiment"].unique())
    colors = Category10[10] if len(exp_list) <= 10 else Turbo256[: len(exp_list)]
    color_map = {exp: colors[i % len(colors)] for i, exp in enumerate(exp_list)}

    for m in metrics:
        p = figure(
            title=m["title"],
            width_policy="max",
            height=400,
            tools="pan,box_zoom,wheel_zoom,reset,save",
            active_scroll="wheel_zoom",
            y_axis_type=axis_type,
        )
        style_plot(p, "Epoch", m["y_axis"], m["title"])

        legend_items = []
        all_renderers = []

        for exp_name in exp_list:
            subset = df[df["Experiment"] == exp_name].sort_values("Epoch")
            if subset.empty:
                continue

            source = ColumnDataSource(subset)
            c = color_map[exp_name]
            label = subset.iloc[0]["Label"]

            # Line only, no points to avoid clutter
            line = p.line(
                x="Epoch", y=m["col"], source=source, line_width=2, color=c, alpha=0.9
            )

            # Invisible scatter for tooltip targeting (makes hitting the line easier)
            scatter = p.scatter(
                x="Epoch",
                y=m["col"],
                source=source,
                size=10,
                alpha=0,
                hover_alpha=0.5,
                color=c,
            )

            all_renderers.append(scatter)
            legend_items.append(LegendItem(label=label, renderers=[line]))

        # Tooltip: Minimalist to avoid "Giant Tooltip"
        # mode='mouse' ensures only the hovered point is shown
        hover = HoverTool(
            renderers=all_renderers,
            tooltips=[
                ("Model", "@Label"),
                ("Values", f"Epoch: @Epoch, {m['y_axis']}: @{m['col']}{{0.000}}"),
            ],
            mode="mouse",
        )
        p.add_tools(hover)

        legend = Legend(items=legend_items)
        legend.click_policy = "hide"
        legend.label_text_color = "#cccccc"
        legend.background_fill_color = "#202020"
        legend.border_line_color = "#444444"
        p.add_layout(legend, "right")

        plots.append(p)

    return gridplot(
        [[plots[0], plots[1]], [plots[2], plots[3]]], sizing_mode="stretch_width"
    )


def create_viz_epoch():
    df = load_training_history()
    if df.empty:
        print("No data loaded.")
        return

    output_file(OUTPUT_FILE, title="Training Curves Visualization")

    # Create two separate grids: Linear and Log
    grid_linear = create_grid(df, "linear")
    grid_log = create_grid(df, "log")

    # Log grid is hidden initially
    grid_log.visible = False

    # Toggle Control
    radio_group = RadioGroup(
        labels=["Linear Scale", "Log Scale"], active=0, inline=True
    )

    # JavaScript Callback to toggle visibility
    callback = CustomJS(
        args=dict(linear=grid_linear, log=grid_log),
        code="""
        if (cb_obj.active == 0) {
            linear.visible = true;
            log.visible = false;
        } else {
            linear.visible = false;
            log.visible = true;
        }
    """,
    )
    radio_group.js_on_change("active", callback)

    # Layout
    layout = column(
        row(radio_group, sizing_mode="fixed", width=300),
        grid_linear,
        grid_log,
        sizing_mode="stretch_width",
    )

    save(layout)
    print(f"Visualization saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    create_viz_epoch()
