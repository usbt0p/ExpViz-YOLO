from pathlib import Path
import os

from training_visualization import load_training_history, create_viz_epoch
from metrics_visualization import load_experiments_data, create_visualization


def serve_results():
    # TODO
    ...

if __name__ == "__main__":
    # Paths assuming script is run from project root
    base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    EXP_LIST_DIR = base_dir / "experiments_yolo26/list"
    EXP_TRAIN_DIR = base_dir / "experiments_yolo26/train"
    OUTPUT_DIR = base_dir / "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_training_history(EXP_LIST_DIR, EXP_TRAIN_DIR)
    create_viz_epoch(df, OUTPUT_DIR / "training_visualization.html")

    # new output for this one
    df = load_experiments_data(EXP_LIST_DIR, EXP_TRAIN_DIR)
    create_visualization(df, OUTPUT_DIR / "metrics_visualization.html")

    # TODO serve each one under a simple index.html
    serve_results()
