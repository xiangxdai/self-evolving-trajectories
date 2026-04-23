import argparse
import json
import random
from pathlib import Path


# Scheme 4 only: Bi-directional Anchored Smoothing.
DEFAULT_N = 9
DEFAULT_NUM_TRAIN = 100_000
DEFAULT_NUM_TEST = 1_000
DEFAULT_SEED = 42
DEFAULT_POS_CONST = [3, 1, 4, 1, 5, 9, 2, 6, 5]


def parse_pos_const(raw: str, n: int) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"len(pos_const) must equal n: got {len(vals)} vs n={n}")
    return [v % 10 for v in vals]


def default_pos_const(n: int) -> list[int]:
    if n <= len(DEFAULT_POS_CONST):
        return DEFAULT_POS_CONST[:n]
    out: list[int] = []
    while len(out) < n:
        out.extend(DEFAULT_POS_CONST)
    return out[:n]


def generate_samples(
    num_samples: int,
    n: int,
    pos_const: list[int],
    rng: random.Random,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _ in range(num_samples):
        plain = [rng.randint(0, 9) for _ in range(n)]
        cipher = [0] * n

        # Anchors.
        cipher[0] = plain[0]
        cipher[n - 1] = plain[n - 1]

        # Interior dependency.
        for i in range(1, n - 1):
            cipher[i] = (plain[i - 1] + plain[i] + plain[i + 1] + pos_const[i]) % 10

        rows.append(
            {
                "input": "".join(str(x) for x in cipher),
                "output": "".join(str(x) for x in plain),
            }
        )
    return rows


def verify_one(sample: dict[str, str], pos_const: list[int]) -> bool:
    plain = [int(x) for x in sample["output"]]
    n = len(plain)
    recon = [0] * n
    recon[0] = plain[0]
    recon[n - 1] = plain[n - 1]
    for i in range(1, n - 1):
        recon[i] = (plain[i - 1] + plain[i] + plain[i + 1] + pos_const[i]) % 10
    return "".join(str(x) for x in recon) == sample["input"]


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Bi-directional Anchored Smoothing cipher dataset."
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Sequence length")
    parser.add_argument("--num-train", type=int, default=DEFAULT_NUM_TRAIN, help="Train size")
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST, help="Test size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output prefix (default: cipher{n}_bidirectional_anchored_smoothing)",
    )
    parser.add_argument(
        "--pos-const",
        type=str,
        default=None,
        help="Comma-separated constants K (length must equal n). "
        f"Default for n={DEFAULT_N}: {DEFAULT_POS_CONST}",
    )
    args = parser.parse_args()

    if args.n <= 2:
        raise ValueError("n must be > 2 for bi-directional interior dependency")

    if args.pos_const is None:
        pos_const = default_pos_const(args.n)
    else:
        pos_const = parse_pos_const(args.pos_const, args.n)

    rng = random.Random(args.seed)
    train_rows = generate_samples(args.num_train, args.n, pos_const, rng)
    test_rows = generate_samples(args.num_test, args.n, pos_const, rng)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.name or f"cipher{args.n}_bidirectional_anchored_smoothing"
    train_path = out_dir / f"{prefix}_train.jsonl"
    test_path = out_dir / f"{prefix}_test.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    print("Generated Bi-directional Anchored Smoothing dataset.")
    print(f"n={args.n}, seed={args.seed}, pos_const={pos_const}")
    print(f"train -> {train_path}")
    print(f"test  -> {test_path}")

    if test_rows:
        ok = verify_one(test_rows[0], pos_const)
        print(f"first-sample verification: {'SUCCESS' if ok else 'FAILED'}")
        print(f"example cipher={test_rows[0]['input']}")
        print(f"example plain ={test_rows[0]['output']}")


if __name__ == "__main__":
    main()
