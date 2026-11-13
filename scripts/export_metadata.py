"""Export canonical network metadata from the validated scientific constants."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EDGES = ((1, 2), (1, 8), (2, 3), (2, 9), (3, 4), (3, 10),
         (4, 5), (4, 11), (5, 6), (5, 12), (6, 7), (6, 13),
         (7, 14), (8, 9), (9, 10), (10, 11), (11, 12),
         (12, 13), (13, 14))
NODES = (1, 2, 3, 4, 5, 6, 7, 14, 13, 12, 11, 10, 9, 8)
PAIRS = tuple((NODES[i], NODES[j]) for i in range(14) for j in range(i + 1, 14))


def main() -> None:
    target = ROOT / "data/metadata"
    target.mkdir(parents=True, exist_ok=True)
    with (target / "physical_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sensor", "node_u", "node_v"))
        writer.writerows((f"S{i}", u, v) for i, (u, v) in enumerate(EDGES, 1))
    with (target / "terminal_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pair_index_zero_based", "identification_pair", "validation_pair", "node_u", "node_v"))
        writer.writerows((i, f"{u}-{v}", f"{min(u, v)}-{max(u, v)}", u, v)
                         for i, (u, v) in enumerate(PAIRS))
    (target / "network.json").write_text(json.dumps({
        "terminal_nodes": 14,
        "sensing_elements": 19,
        "terminal_pairs": 91,
        "grounded_node": 14,
        "clockwise_node_order": NODES,
        "pair_order_rule": "upper triangle of clockwise_node_order; validation labels normalize node IDs ascending without changing index order",
        "source": "src/scientific/experiment_I5_checkpointed_convergence.py",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote metadata to {target}")


if __name__ == "__main__":
    main()
