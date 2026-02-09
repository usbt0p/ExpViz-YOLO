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
            
            # this has to support all naming differences and ensure there's no collisions in order
            # for the legend to work properly

            df_exp = df.copy()
            df_exp["Experiment"] = exp_name
            df_exp["Label"] = exp_name.replace("_", " "
                ).replace("pretrained", "pret.").replace("backbone", "back."
                ).replace("rectFalse", "").replace("rectTrue", "rect").replace("  ", " ")

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
    # colors = Category10[10] if len(exp_list) <= 10 else Turbo[256]
    # color_map = {exp: colors[i % len(colors)] for i, exp in enumerate(exp_list)}
    color_map = get_color_map(exp_list)

    renderer_map = {}

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

            # Invisible scatter for tooltip targeting
            scatter = p.scatter(
                x="Epoch",
                y=m["col"],
                source=source,
                size=2,
                alpha=0,
                hover_alpha=0.5,
                color=c,
            )

            all_renderers.append(scatter)

            # Collect renderers for global legend
            if label not in renderer_map:
                renderer_map[label] = []
            renderer_map[label].append(line)

        # Tooltip
        hover = HoverTool(
            renderers=all_renderers,
            tooltips="""
            <div style="background-color: rgba(32, 32, 32, 0.7); padding: 10px; border: 1px solid #444; border-radius: 5px;">
                <div style="color: #fff; font-weight: bold; margin-bottom: 5px;">@Label</div>
                <div style="color: #aaa; font-size: 0.9em;">
                    Epoch: <span style="color: #eee;">@Epoch</span><br>
                    {tooltip_label}: <span style="color: #eee;">@{col_name}{{0.0000}}</span>
                </div>
            </div>
            """.format(
                tooltip_label=m["tooltip_label"], col_name=m["col"]
            ),
            mode="mouse",
        )
        p.add_tools(hover)
        plots.append(p)

    grid = gridplot(
        [[plots[0], plots[1]], [plots[2], plots[3]]], sizing_mode="stretch_width"
    )
    return grid, renderer_map


def create_viz_epoch(training_history_df, output_path, top_n=None):
    """
    Creates a visualization of training curves for a given DataFrame. Supports linear,
    log and log-error (1-metric) views.

    Args:
        training_history_df (pandas.DataFrame): DataFrame containing training history for all experiments.
        output_file (str): Path to save the visualization HTML file.
        top_n (int, optional): Number of top experiments to show based on mAP50_95.
    Returns:
        str: Path to the saved visualization HTML file.
    """

    if training_history_df.empty:
        print("No data loaded. Aborting.")
        with open(output_path, "w", encoding='utf-8') as err_html:
            err_html.write("<html><body style='display:flex;justify-content:center;align-items:center;height:100vh;font-family:'Georgia', 'Times New Roman', serif;color:#666;'><div><h2>No data loaded</h2><p>The training dataframe is empty.</p></div></body></html>")
        return

    # Filter Top N
    if top_n:
        # Find max mAP50_95 per experiment to rank them
        ranking = (
            training_history_df.groupby("Experiment")["mAP50_95"]
            .max()
            .sort_values(ascending=False)
        )
        top_exps = ranking.index[:top_n].tolist()
        training_history_df = training_history_df[
            training_history_df["Experiment"].isin(top_exps)
        ]
        print(f"[Training Viz] Filtering top {top_n} experiments")

    output_file(output_path, title="Training Curves Visualization")

    # 1. Linear Metric View
    metrics_linear = [
        # ... existing metrics ...
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
    grid_linear, map_linear = create_grid(training_history_df, metrics_linear, "linear")

    # 2. Log Loss View
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
    grid_log_loss, map_log_loss = create_grid(
        training_history_df, metrics_log_loss, "log"
    )
    grid_log_loss.visible = False

    # 3. Log Error View
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
    grid_log_error, map_log_error = create_grid(
        training_history_df, metrics_log_error, "log"
    )
    grid_log_error.visible = False

    # Merge Renderer Maps for Global Toggle
    exp_list = sorted(training_history_df["Experiment"].unique())
    color_map = get_color_map(exp_list)
    label_to_color = {
        training_history_df[training_history_df["Experiment"] == exp].iloc[0][
            "Label"
        ]: color_map[exp]
        for exp in exp_list
        if not training_history_df[training_history_df["Experiment"] == exp].empty
    }

    all_labels = sorted(
        list(
            set(map_linear.keys())
            | set(map_log_loss.keys())
            | set(map_log_error.keys())
        )
    )

    # Create Unified Legend Plot
    # We create a dummy plot to host the legend on the right of the entire grid
    height = 150 + len(all_labels) * 20
    p_legend = figure(
        width=250,
        height=height,
        toolbar_location=None,
        outline_line_color=None,
        x_range=(0, 1),
        y_range=(0, 1),  # Keep dummy glyphs out of view
    )
    p_legend.background_fill_color = "#151515"
    p_legend.border_fill_color = "#151515"
    p_legend.xaxis.visible = False
    p_legend.yaxis.visible = False
    p_legend.grid.visible = False

    legend_items = []
    for label in all_labels:
        # Add a dummy glyph to p_legend so the legend specimen has something to draw in its own plot context
        color = label_to_color.get(label, "#ffffff")
        dummy = p_legend.line(
            x=[2], y=[2], color=color, line_width=2
        )  # x=2 is out of (0,1) range

        renderers = (
            map_linear.get(label, [])
            + map_log_loss.get(label, [])
            + map_log_error.get(label, [])
            + [dummy]
        )
        legend_items.append(LegendItem(label=label, renderers=renderers))

    legend = Legend(items=legend_items, ncols=1)
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
    # Group the 3 grids in a single column and place the legend next to it
    grids_col = column(
        grid_linear, grid_log_loss, grid_log_error, sizing_mode="stretch_width"
    )

    layout = column(
        row(radio_group, sizing_mode="fixed", width=450),
        row(grids_col, p_legend, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )

    # Final styling: background color of the whole page
    curdoc().theme = "dark_minimal"

    out = save(layout, output_path)

    print(f"Visualization saved to {out}")
    return out
