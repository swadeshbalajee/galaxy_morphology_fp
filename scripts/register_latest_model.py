from __future__ import annotations

import argparse

from src.registry.register_best_model import register_best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    print(register_best_model(run_id=args.run_id))
