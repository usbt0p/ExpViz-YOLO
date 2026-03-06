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

    # Each entry points to one group of experiments. The benchmark CSV is
    # looked up relative to the parent of train_dir (see load_experiments_data).
    exps = [
        {
            "list_dir": base_dir / "compressed_yolo26_imgsz_exps" / "list",
            "train_dir": base_dir / "compressed_yolo26_imgsz_exps" / "train",
            "bench_csv": "combined_results.csv",
        },
        {
            "list_dir": base_dir / "compressed_yolo26_p2_p6_exps" / "experiments_yolo26_p2_segundo_intento" / "list",
            "train_dir": base_dir / "compressed_yolo26_p2_p6_exps" / "experiments_yolo26_p2_segundo_intento" / "train",
            "bench_csv": "combined_results.csv",
        },
    ]

    OUTPUT_DIR = base_dir / "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load each group separately and combine so all experiments appear in a single plot
    dfs_train, dfs_metrics = [], []
    for exp in exps:
        dfs_train.append(load_training_history(exp["list_dir"], exp["train_dir"]))
        dfs_metrics.append(load_experiments_data(exp["list_dir"], exp["train_dir"], exp["bench_csv"]))

    df_train = pd.concat(dfs_train, ignore_index=True)
    df_metrics = pd.concat(dfs_metrics, ignore_index=True)

    # Filter Top N models to avoid clutter (e.g. 15)
    TOP_N = 150

    print("Generating Training Visualization...")
    create_viz_epoch(df_train, OUTPUT_DIR / "training_visualization.html", top_n=TOP_N)

    print("Generating Metrics Visualization...")
    print(df_metrics)
    create_metrics_visualization(
        df_metrics, OUTPUT_DIR / "metrics_visualization.html", top_n=TOP_N
    )

    print("Generating Precision-Recall Visualization...")
    create_pr_visualization(
        df_metrics, OUTPUT_DIR / "pr_visualization.html", top_n=TOP_N
    )

    print("Generating Index...")
    generate_index(OUTPUT_DIR)

    serve_results(OUTPUT_DIR)
