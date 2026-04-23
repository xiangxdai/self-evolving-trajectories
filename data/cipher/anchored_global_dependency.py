import argparse
import json
import math
import random
from pathlib import Path


# Main paper-style default: Anchored Global Dependency (Cipher-17).
DEFAULT_N = 17
DEFAULT_K_OFFSET = 5
DEFAULT_POS_CONST = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2]
DEFAULT_NUM_TRAIN = 100_000
DEFAULT_NUM_TEST = 1_000
DEFAULT_SEED = 42


def parse_pos_const(raw: str, n: int) -> list[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"len(pos_const) must equal n: got {len(vals)} vs n={n}")
    return [v % 10 for v in vals]


def solve_order_0_based(n: int, k_offset: int) -> list[int]:
    """
    Return anchored decode chain for:
      c[0] = p[0]
      c[i] = (p[i] + p[(i+k) % n] + K[i]) % 10, i>=1
    """
    order: list[int] = []
    cur = 0
    for _ in range(n - 1):
        cur = (cur - k_offset) % n
        order.append(cur)
    return order


def generate_samples_anchored_global(
    num_samples: int,
    n: int,
    k_offset: int,
    pos_const: list[int],
    rng: random.Random,
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for _ in range(num_samples):
        plain = [rng.randint(0, 9) for _ in range(n)]
        cipher = [0] * n
        cipher[0] = plain[0]
        for i in range(1, n):
            j = (i + k_offset) % n
            cipher[i] = (plain[i] + plain[j] + pos_const[i]) % 10
        samples.append(
            {
                "input": "".join(str(x) for x in cipher),
                "output": "".join(str(x) for x in plain),
            }
        )
    return samples


def verify_one_sample(
    sample: dict[str, str],
    n: int,
    k_offset: int,
    pos_const: list[int],
) -> tuple[bool, list[int], list[int]]:
    c = [int(x) for x in sample["input"]]
    p_true = [int(x) for x in sample["output"]]
    p_solved: list[int | None] = [None] * n
    p_solved[0] = c[0]

    for i in solve_order_0_based(n, k_offset):
        j = (i + k_offset) % n
        assert p_solved[j] is not None
        p_solved[i] = (c[i] - p_solved[j] - pos_const[i]) % 10

    solved_int = [int(x) for x in p_solved]
    return solved_int == p_true, p_true, solved_int


def write_jsonl(path: Path, samples: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Anchored Global Dependency cipher dataset (Cipher-17 by default)."
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Sequence length")
    parser.add_argument("--k-offset", type=int, default=DEFAULT_K_OFFSET, help="Jump offset k")
    parser.add_argument(
        "--pos-const",
        type=str,
        default=",".join(str(x) for x in DEFAULT_POS_CONST),
        help="Comma-separated constants K (length must equal n)",
    )
    parser.add_argument("--num-train", type=int, default=DEFAULT_NUM_TRAIN, help="Train size")
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST, help="Test size")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Output prefix (default: cipher{n}_anchored_global_mod10)",
    )
    args = parser.parse_args()

    if args.n <= 1:
        raise ValueError("n must be > 1")
    if not (1 <= args.k_offset < args.n):
        raise ValueError("k_offset must satisfy 1 <= k_offset < n")

    pos_const = parse_pos_const(args.pos_const, args.n)

    if math.gcd(args.n, args.k_offset) != 1:
        print(
            f"[Warning] gcd(n={args.n}, k_offset={args.k_offset}) != 1. "
            "Dependency chain may not visit all positions."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.name or f"cipher{args.n}_anchored_global_mod10"
    train_path = out_dir / f"{prefix}_train.jsonl"
    test_path = out_dir / f"{prefix}_test.jsonl"

    rng = random.Random(args.seed)
    train_samples = generate_samples_anchored_global(
        num_samples=args.num_train,
        n=args.n,
        k_offset=args.k_offset,
        pos_const=pos_const,
        rng=rng,
    )
    test_samples = generate_samples_anchored_global(
        num_samples=args.num_test,
        n=args.n,
        k_offset=args.k_offset,
        pos_const=pos_const,
        rng=rng,
    )

    write_jsonl(train_path, train_samples)
    write_jsonl(test_path, test_samples)

    order_1_based = [i + 1 for i in solve_order_0_based(args.n, args.k_offset)]
    print("Generated Anchored Global Dependency dataset.")
    print(f"n={args.n}, k_offset={args.k_offset}, seed={args.seed}")
    print(f"pos_const={pos_const}")
    print(f"solve_order (1-based, excluding anchor p1): {order_1_based}")
    print(f"train -> {train_path}")
    print(f"test  -> {test_path}")

    if test_samples:
        ok, p_true, p_solved = verify_one_sample(
            sample=test_samples[0],
            n=args.n,
            k_offset=args.k_offset,
            pos_const=pos_const,
        )
        print("\nVerification on first test sample:")
        print(f"cipher = {test_samples[0]['input']}")
        print(f"plain  = {test_samples[0]['output']}")
        print(f"solved = {''.join(str(x) for x in p_solved)}")
        print(f"status = {'SUCCESS' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
