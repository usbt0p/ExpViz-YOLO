from pathlib import Path
import os
import pandas as pd

from training_visualization import load_training_history, create_viz_epoch
from metrics_visualization import load_experiments_data, create_metrics_visualization
from precision_recall_visualization import create_pr_visualization

import webbrowser
from http.server import SimpleHTTPRequestHandler
import socketserver


def generate_index(output_dir: Path):
    """Generates a simple index.html linking to the visualizations."""

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Auria Vision Experiments</title>
        <style>
            body {
                font-family: "Georgia", "Times New Roman", serif;
                background-color: #ffffff;
                color: #333;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                text-align: center;
            }
            .container {
                max-width: 600px;
                width: 100%;
                padding: 40px;
                border: 1px solid #ddd;
                box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            }
            h1 {
                font-weight: normal;
                border-bottom: 1px solid #ddd;
                padding-bottom: 20px;
                margin-bottom: 40px;
                letter-spacing: 1px;
                text-transform: uppercase;
                font-size: 1.8rem;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                margin: 30px 0;
            }
            a {
                text-decoration: none;
                font-size: 1.5rem;
                color: #000;
                border-bottom: 1px solid transparent;
                transition: border-bottom 0.3s ease;
            }
            a:hover {
                border-bottom: 1px solid #000;
            }
            p {
                margin-top: 8px;
                color: #777;
                font-size: 0.9rem;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Experiment Visualizations</h1>
            <ul>
                <li>
                    <a href="training_visualization.html">Training Curves</a>
                    <p>Metrics vs Epochs (mAP, Precision, Recall, Loss)</p>
                </li>
                <li>
                    <a href="metrics_visualization.html">Performance Metrics</a>
                    <p>Pareto Frontier (Metrics vs Latency)</p>
                </li>
                <li>
                    <a href="pr_visualization.html">Precision-Recall Tradeoff</a>
                    <p>Scatter Plot (Precision vs Recall for Best Checkpoint)</p>
                </li>
            </ul>
        </div>
    </body>
    </html>
    """

    with open(output_dir / "index.html", "w") as f:
        f.write(html_content)
    print(f"Index created at {output_dir / 'index.html'}")


def serve_results(output_dir: Path, port=8000):
    """Serves the results directory via HTTP."""
    os.chdir(output_dir)

    Handler = SimpleHTTPRequestHandler

    # Allow address reuse to avoid "Address already in use" errors during quick restarts
    socketserver.TCPServer.allow_reuse_address = True

    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"\nServing results at {url}")
        print("Press Ctrl+C to stop.")

        # Open in browser automatically
        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.")
            httpd.server_close()

if __name__ == "__main__":
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # fp32 was benchmarked separately per experiment group; all other precision
    # variants were aggregated into a single CSV covering every group.

    # use this structure when you have *distinct experiments with one CSV each*, but want
    # them all visualized together in the same plots (e.g. to compare them side by side)

    # note: for the last benchs, True_True_800 would mean halfp=True, int8=True, size=800
    # some configs like that failed so they don't have to be plotted (logical, you cant have int8 and halfp at the same time)
    EXP_GROUPS = [
        {
            "list_dir": base_dir / "compressed_yolo26_imgsz_exps" / "list",
            "train_dir": base_dir / "compressed_yolo26_imgsz_exps" / "train",
        },
        {
            "list_dir": base_dir / "compressed_yolo26_p2_p6_exps" / "list",
            "train_dir": base_dir / "compressed_yolo26_p2_p6_exps" / "train",
        },
    ]

    # These CSVs already contain results from all experiment groups.
    # use these when you have several experiments that share the same 
    # benchmark results in *one CSV* file 
    SHARED_BENCH_VARIANTS = [
        {"label": "fp32_640",       "csv": base_dir / "PUTOS_BENCHS/putos_benchs_False_False_640"       / "combined_results.csv"},
        {"label": "fp32_800",       "csv": base_dir / "PUTOS_BENCHS/putos_benchs_False_False_800"       / "combined_results.csv"},
        {"label": "halfp_640",      "csv": base_dir / "PUTOS_BENCHS/putos_benchs_True_False_640"       / "combined_results.csv"},
        {"label": "halfp_800",      "csv": base_dir / "PUTOS_BENCHS/putos_benchs_True_False_800"       / "combined_results.csv"},
        # int8 experiments failed due to a problem with versions
        # {"label": "int8_640",       "csv": base_dir / "PUTOS_BENCHS/putos_benchs_False_True_640"        / "combined_results.csv"},
        # {"label": "int8_800",       "csv": base_dir / "PUTOS_BENCHS/putos_benchs_False_True_800"        / "combined_results.csv"},
    ]

    OUTPUT_DIR = base_dir / "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TOP_N = 150

    print("Generating Training Visualization...")
    dfs_train = [load_training_history(g["list_dir"], g["train_dir"]) for g in EXP_GROUPS]
    df_train = pd.concat(dfs_train, ignore_index=True)
    create_viz_epoch(df_train, OUTPUT_DIR / "training_visualization.html", top_n=TOP_N)

    print("Generating Metrics / PR Visualizations...")
    dfs_metrics = (
        # shared variants: one CSV already covers all groups
        [load_experiments_data(g["list_dir"], g["train_dir"], v["csv"], bench_label=v["label"]) for g in EXP_GROUPS for v in SHARED_BENCH_VARIANTS]
    )
    df_metrics = pd.concat(dfs_metrics, ignore_index=True)
    print(df_metrics)

    create_metrics_visualization(
        df_metrics, OUTPUT_DIR / "metrics_visualization.html", top_n=TOP_N
    )
    create_pr_visualization(
        df_metrics, OUTPUT_DIR / "pr_visualization.html", top_n=TOP_N
    )

    print("Generating Index...")
    generate_index(OUTPUT_DIR)

    serve_results(OUTPUT_DIR)
