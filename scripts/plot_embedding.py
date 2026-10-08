"""Generate the original UMAP ID versus synthetic OOD figure."""
from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot import generate_and_plot_id_vs_ood

if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    from configs.config import get_params
    generate_and_plot_id_vs_ood(get_params())
