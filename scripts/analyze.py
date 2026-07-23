"""Build the comparison table + figures from results/. CPU only, no GPU needed.

    python scripts/analyze.py

Outputs comparison_table.csv, ablation.png, privacy_utility.png,
byzantine.png, learning_curves.png under FEDDAPT_ROOT/figures/.
"""
from fedapt.config import load_config
from fedapt import analysis


def main():
    analysis.run_analysis(load_config())


if __name__ == "__main__":
    main()
