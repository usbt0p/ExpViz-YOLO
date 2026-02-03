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
from bokeh.io import curdoc

from commons import style_plot, get_color_map


def load_training_history(exp_list_dir, exp_train_dir):
    """
    Reads full training history for all experiments.

    Args:
        exp_list_dir (str): Directory containing experiment configuration files. Assumes the list
            directory provided contains the `.yaml` files for each experiment.
        exp_train_dir (str): Directory containing experiment training results. Assumes the train
            directory provided contains the subdirectories with each of the exp_list_dir experiment names, and
            their respective `results.csv` files.

    Returns:
        pd.DataFrame: DataFrame containing training history for all experiments.
    """
    data_frames = []
    search_path = os.path.join(exp_list_dir, "*.yaml")
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
        model_version = config.get("model_version", "Unknown")
        freezing_strategy = config.get("freeze_layers", None)

        results_path = os.path.join(exp_train_dir, exp_name, "results.csv")

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
                "val_box": "val/box_loss",
                "val_cls": "val/cls_loss",
                "val_dfl": "val/dfl_loss",
                "train_box": "train/box_loss",
            }

            if not all(c in df.columns for c in col_map.values()):
                continue

            # Label Generation
            variant = "Pret." if is_pretrained else "Scratch"
            if freezing_strategy:
                variant += f" ({freezing_strategy})"
            clean_name = f"{model_version.capitalize()}{model_size} {variant}"
            if rect_mode:
                clean_name += " Rect"

            df_exp = df.copy()
            df_exp["Experiment"] = exp_name
            df_exp["Label"] = clean_name

            df_exp.rename(
                columns={
                    col_map["map50"]: "mAP50",
                    col_map["map50_95"]: "mAP50_95",
                    col_map["precision"]: "Precision",
                    col_map["recall"]: "Recall",
                    col_map["val_box"]: "ValBoxLoss",
                    col_map["val_cls"]: "ValClsLoss",
                    col_map["val_dfl"]: "ValDflLoss",
                    col_map["train_box"]: "TrainBoxLoss",
                    "epoch": "Epoch",
                },
                inplace=True,
            )

            # Calculate Error Metrics (1 - Metric) for Inverted Log View
            df_exp["Error_mAP50"] = 1 - df_exp["mAP50"]
            df_exp["Error_mAP50_95"] = 1 - df_exp["mAP50_95"]
            df_exp["Error_Precision"] = 1 - df_exp["Precision"]
            df_exp["Error_Recall"] = 1 - df_exp["Recall"]

            # Clip error at extremely small epsilon for log scale safety
            epsilon = 1e-6
            for col in [
                "Error_mAP50",
                "Error_mAP50_95",
                "Error_Precision",
                "Error_Recall",
            ]:
                df_exp[col] = df_exp[col].clip(lower=epsilon)

            keep_cols = [
                "Experiment",
                "Label",
                "mAP50",
                "mAP50_95",
                "Precision",
                "Recall",
                "Error_mAP50",
                "Error_mAP50_95",
                "Error_Precision",
                "Error_Recall",
                "ValBoxLoss",
                "ValClsLoss",
                "ValDflLoss",
                "TrainBoxLoss",
                "Epoch",
            ]
            data_frames.append(df_exp[keep_cols])

        except Exception as e:
            print(f"Error processing {exp_name}: {e}")

    if not data_frames:
        return pd.DataFrame()

    return pd.concat(data_frames, ignore_index=True)

def create_grid(df, metrics, axis_type="linear"):
    """
    Creates a 2x2 grid of plots for a given list of metrics.

    Args:
        df (pandas.DataFrame): DataFrame containing training history for all experiments.
        metrics (list of dict): List of dictionaries containing metric information.
        axis_type (str, optional): Type of axis to use. Defaults to "linear".

    Returns:
        bokeh.layouts.gridplot.GridPlot: A 2x2 grid of plots.
    """
    plots = []

    # Colors
    exp_list = sorted(df["Experiment"].unique())
    #colors = Category10[10] if len(exp_list) <= 10 else Turbo[256]
    #color_map = {exp: colors[i % len(colors)] for i, exp in enumerate(exp_list)}
    color_map = get_color_map(exp_list)

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

            # Line only
            line = p.line(
                x="Epoch", y=m["col"], source=source, line_width=2, color=c, alpha=0.9
            )

            # Invisible scatter for tooltip targeting (makes it easier to target the tooltip)
            scatter = p.scatter(
                x="Epoch",
                y=m["col"],
                source=source,
                size=3,
                alpha=0,
                hover_alpha=0.5,
                color=c,
            )

            all_renderers.append(scatter)
            legend_items.append(LegendItem(label=label, renderers=[line]))

        # Tooltip
        hover = HoverTool(
            renderers=all_renderers,
            tooltips=[
                ("Model", "@Label"),
                (
                    "Values",
                    f"Epoch: @Epoch, {m['tooltip_label']}: @{m['col']}{{0.0000}}",
                ),
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


def create_viz_epoch(training_history_df, output_path):
    """
    Creates a visualization of training curves for a given DataFrame. Supports linear,
    log and log-error (1-metric) views.

    Args:
        training_history_df (pandas.DataFrame): DataFrame containing training history for all experiments.
        output_file (str): Path to save the visualization HTML file.
    Returns:
        str: Path to the saved visualization HTML file.
    """

    if training_history_df.empty:
        print("No data loaded.")
        return

    output_file(output_path, title="Training Curves Visualization")

    # 1. Linear Metric View
    metrics_linear = [
        {
            "col": "mAP50",
            "title": "mAP 50 (Linear)",
            "y_axis": "mAP 50",
            "tooltip_label": "mAP 50",
        },
        {
            "col": "mAP50_95",
            "title": "mAP 50-95 (Linear)",
            "y_axis": "mAP 50-95",
            "tooltip_label": "mAP 50-95",
        },
        {
            "col": "Precision",
            "title": "Precision (Linear)",
            "y_axis": "Precision",
            "tooltip_label": "Precision",
        },
        {
            "col": "Recall",
            "title": "Recall (Linear)",
            "y_axis": "Recall",
            "tooltip_label": "Recall",
        },
    ]
    grid_linear = create_grid(training_history_df, metrics_linear, "linear")

    # 2. Log Loss View (Replaces Log Metric)
    metrics_log_loss = [
        {
            "col": "ValBoxLoss",
            "title": "Val Box Loss (Log)",
            "y_axis": "Loss",
            "tooltip_label": "Val Box",
        },
        {
            "col": "ValClsLoss",
            "title": "Val Class Loss (Log)",
            "y_axis": "Loss",
            "tooltip_label": "Val Class",
        },
        {
            "col": "ValDflLoss",
            "title": "Val DFL Loss (Log)",
            "y_axis": "Loss",
            "tooltip_label": "Val DFL",
        },
        {
            "col": "TrainBoxLoss",
            "title": "Train Box Loss (Log)",
            "y_axis": "Loss",
            "tooltip_label": "Train Box",
        },
    ]
    grid_log_loss = create_grid(training_history_df, metrics_log_loss, "log")
    grid_log_loss.visible = False

    # 3. Log Error View (Inverted)
    metrics_log_error = [
        {
            "col": "Error_mAP50",
            "title": "Error (1-mAP50) (Log)",
            "y_axis": "Error Rate",
            "tooltip_label": "Error",
        },
        {
            "col": "Error_mAP50_95",
            "title": "Error (1-mAP50-95) (Log)",
            "y_axis": "Error Rate",
            "tooltip_label": "Error",
        },
        {
            "col": "Error_Precision",
            "title": "Error (1-Precision) (Log)",
            "y_axis": "Error Rate",
            "tooltip_label": "Error",
        },
        {
            "col": "Error_Recall",
            "title": "Error (1-Recall) (Log)",
            "y_axis": "Error Rate",
            "tooltip_label": "Error",
        },
    ]
    grid_log_error = create_grid(training_history_df, metrics_log_error, "log")
    grid_log_error.visible = False

    # Toggle Control
    radio_group = RadioGroup(
        labels=["Linear Metric", "Log Loss", "Log Error (Inverted)"],
        active=0,
        inline=True,
    )

    # JavaScript Callback
    callback = CustomJS(
        args=dict(g1=grid_linear, g2=grid_log_loss, g3=grid_log_error),
        code="""
        g1.visible = false;
        g2.visible = false;
        g3.visible = false;
        
        if (cb_obj.active == 0) {
            g1.visible = true;
        } else if (cb_obj.active == 1) {
            g2.visible = true;
        } else {
            g3.visible = true;
        }
    """,
    )
    radio_group.js_on_change("active", callback)

    # Layout
    layout = column(
        row(radio_group, sizing_mode="fixed", width=450),
        grid_linear,
        grid_log_loss,
        grid_log_error,
        sizing_mode="stretch_width",
    )

    out = save(layout, output_path)
    print(f"Visualization saved to {out}")
    return out
