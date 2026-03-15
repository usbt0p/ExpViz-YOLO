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
                background-color: #f7f9fa;
                color: #333;
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                padding: 40px 20px;
            }
            .container {
                max-width: 700px;
                width: 100%;
                padding: 40px;
                border: 1px solid #ddd;
                box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            }
            h1 {
                font-weight: 600;
                border-bottom: 2px solid #f0f0f0;
                padding-bottom: 20px;
                margin-top: 0;
                margin-bottom: 30px;
                font-size: 2rem;
                color: #111;
                text-align: center;
            }
            .info-box {
                background-color: #f8fbff;
                border-left: 4px solid #0056b3;
                padding: 20px;
                margin-bottom: 40px;
                border-radius: 0 4px 4px 0;
            }
            .info-box h2 {
                margin-top: 0;
                font-size: 1.2rem;
                color: #0056b3;
                margin-bottom: 12px;
            }
            .info-box ul {
                padding-left: 20px;
                margin-bottom: 15px;
            }
            .info-box p, .info-box li {
                font-size: 0.95rem;
                color: #444;
                line-height: 1.6;
                margin-bottom: 8px;
            }
            .info-box p:last-child, .info-box ul:last-child {
                margin-bottom: 0;
            }
            .info-box code {
                background: #edf2f7;
                padding: 2px 4px;
                border-radius: 3px;
                font-size: 0.9rem;
            }
            .links-list {
                list-style-type: none;
                padding: 0;
                margin: 0;
            }
            .links-list > li {
                margin-bottom: 30px;
            }
            .links-list > li:last-child {
                margin-bottom: 0;
            }
            .links-list a {
                text-decoration: none;
                font-size: 1.4rem;
                font-weight: 500;
                color: #111;
                border-bottom: 2px solid transparent;
                transition: color 0.2s ease, border-bottom-color 0.2s ease;
                display: inline-block;
                margin-bottom: 8px;
            }
            .links-list a:hover {
                color: #0056b3;
                border-bottom-color: #0056b3;
            }
            .desc {
                margin: 0;
                color: #666;
                font-size: 0.95rem;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Experiment Visualizations</h1>
            
            <div class="info-box">
                <h2>Experiment Details & Features</h2>
                <p>These interactive dashboards display results iteratively collected for the latest <strong>Ultralytics YOLOv26</strong> objects detection model under various configurations. The graphs support zooming, panning, saving outputs, and trace isolation (by clicking on legend items to toggle visibility).</p>
                <ul>
                    <li><strong>Architectures:</strong> regular YOLO26, plus P2 and P6 variants</li>
                    <li><strong>Training Image Sizes:</strong> 512, 640, and 800</li>
                    <li><strong>Inference Precisions:</strong> FP32 and Half-Precision</li>
                    <li><strong>Inference Image Sizes:</strong> 640 and 800</li>
                    <li><strong>Pretraining Strategies:</strong> Fully trainable parameters, and pre-trained with frozen backbone.</li>
                    <li><strong>Model Sizes:</strong> <code>n</code>, <code>s</code>, and <code>m</code></li>
                </ul>
            </div>

            <ul class="links-list">
                <li>
                    <a href="training_visualization.html">Training Curves</a>
                    <div class="desc">
                        Explore metric evolution over epochs (mAP, Precision, Recall, Loss). The X-axis toggle supports linear, log-loss, and log-error scales (where log error is log(1-metric)).
                    </div>
                </li>
                <li>
                    <a href="metrics_visualization.html">Performance Metrics</a>
                    <div class="desc">
                        Compare Pareto frontiers evaluating metrics against model speed. Tested on different precisions and network sizes. The X-axis toggle supports either Latency (ms) or FPS.
                    </div>
                </li>
                <li>
                    <a href="pr_visualization.html">Precision-Recall Tradeoff</a>
                    <div class="desc">
                        A scatter plot visualizing Precision against Recall, evaluated taking the best checkpoint for each model permutation.
                    </div>
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
