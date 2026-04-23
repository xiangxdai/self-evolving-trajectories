# Beyond Left-to-Right: Unifying Autoregressive and Masked Diffusion Generation via Self-Evolving Trajectories

This repository accompanies our trajectory-learning framework for globally constrained reasoning.
It releases three paper stages - **Tom-CAT**, **Dep-DOG**, and **Ser-FOX** - together with standard
masked-diffusion and autoregressive baselines.

**Release note:** The manuscript PDF is not included in the current public repository snapshot.

> **Important:** `MDM/` is the released vanilla masked-diffusion baseline.
> It is **not** the same thing as `Dep-DOG/`.

---

## What is in this repository?

- `Tom-CAT/`: Stage 1 Tom-CAT implementation
- `Dep-DOG/`: Stage 2 Dep-DOG implementation
- `Ser-FOX/`: Stage 3 Ser-FOX implementation
- `MDM/`: standard masked diffusion baseline
- `AR/`: standard autoregressive baseline

---

## Installation

We provide two **Conda explicit lockfiles** for Linux:

- `spec-file.txt`: recommended default runtime environment
- `spec-file-h100.txt`: H100-oriented alternative environment

Because these are explicit lockfiles rather than a hand-written `environment.yml`,
create them with `conda create --file ...`.

### Recommended setup

```bash
conda create -n settraj --file spec-file.txt
conda activate settraj
```

### H100 setup

```bash
conda create -n settraj_h100 --file spec-file-h100.txt
conda activate settraj_h100
```

### Verify the environment

```bash
python - <<'PY'
import importlib

mods = ["torch", "torchvision", "torchaudio", "transformers", "datasets"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"[ok] {m}")
    except Exception as e:
        print(f"[missing] {m}: {e}")

import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

If you do not specifically need the H100 lockfile, start with `spec-file.txt`.

---

## Quickstart

The most complete runnable example in the current repo is the retained **Dep-DOG CD4 pipeline**.

### 1) Prepare a JSONL dataset

Each line should look like:

```json
{"input": "...", "output": "..."}
```

### 2) Pack it into the repository format

Countdown-style example:

```bash
python data/prepare_data_generic_cd.py \
  --data_path data/cd4_train.jsonl \
  --out_dir data/cd/cd4/k1 \
  --input_key input \
  --output_key output
```

This writes:

- `train.bin`
- `val.bin`
- `meta.pkl`

The retained CD4 pipeline also expects a JSONL test file in the same dataset directory.
For example:

```bash
cp data/cd4_test.jsonl data/cd/cd4/k1/cd4_test.jsonl
```

Adjust the source path to match your local setup.

### 3) Run one complete Dep-DOG example

```bash
bash Dep-DOG/depdog_pipeline_cd4padlast.sh
```

This script is a full multi-round example rather than a smoke test.
Checkpoints are written under `out/depdog_train/`, and generated reordered
datasets such as `train_gen_round2.bin` and `train_gen_round3.bin` are written
back into the dataset directory.

---

## Dep-DOG in one paragraph

Dep-DOG is a **round-based** protocol rather than a single training pass.

- **Round 1** trains on canonical `train.bin` with random masking.
- **Later rounds** warm-start from the previous checkpoint, generate reordered training files such as
  `train_gen_round2.bin` or `train_gen_round3.bin`, and train on teacher-forced cut states induced
  by the learned easy-to-hard order.
- The trainer supports mixing three streams: current reordered data (`D_curr`), historical reordered
  data (`D_prev`), and the canonical anchor (`D_canon`) through `mix_ratios`.

In the retained CD4 example pipeline:

- Round 2 uses `mix_ratios=1.0,0.0,0.0`
- Round 3 uses `mix_ratios=0.7,0.2,0.1`

For a first run, start with the CD4 pipeline above.

---

## Data format and preprocessing

The repository currently exposes two generic preprocessing entrypoints for fixed-length JSONL reasoning datasets:

- `data/prepare_data_generic.py`: general fixed-length preprocessing
- `data/prepare_data_generic_cd.py`: countdown-oriented variant with different defaults

It also includes:

- `data/minimal_planning/generate_dataset.py`: generator for the minimal planning benchmark
- `data/cipher/README.md`: cipher-focused dataset generation notes and scripts

Other paper tasks such as Sudoku, SAT, Countdown, and Cipher are expected to be prepared as JSONL datasets and then packed with the generic preprocessing scripts.

### Additional dataset resources

Datasets for several tasks used in this project can be accessed from this Google Drive package:

- [Google Drive dataset package](https://drive.google.com/file/d/1b0OIlYL76rVVuNYIfIb-L_Ptdg6k_y0c/view?usp=sharing)

For Cipher-and-Plain related material, please also see:

- [`data/cipher/`](data/cipher/README.md)
- [`data/minimal_planning/`](data/minimal_planning/README.md)

For additional Sudoku dataset information, please see:

- [zeyuzhangzyz repositories](https://github.com/zeyuzhangzyz?tab=repositories)

### Packing rule

The preprocessors use a **character-level vocabulary** and serialize each example as:

```text
[quiz padded + <SEP>][response padded + <EOS>]
```

Common special tokens are:

- `<PAD>`
- `<SEP>`
- `<EOS>`
- `<MASK>`
- `$`

The resulting metadata in `meta.pkl` is reused by training and evaluation entrypoints that need to decode or reconstruct packed examples.

### Important convention

Dataset arguments are paths **relative to `data/`**.

Example:

```text
--dataset cd/cd4/k1
```

refers to:

```text
data/cd/cd4/k1
```

---

## Repository layout

This section lists the primary file entrypoints. Conceptual distinctions between
the methods are summarized in **Method map** below.

### Paper stages

#### Tom-CAT (Stage 1)

- Model: `Tom-CAT/tomcat_model.py`
- Train: `Tom-CAT/tomcat_train.py`
- Eval: `Tom-CAT/tomcat_eval.py`
- Ablations:
  - `Tom-CAT/ablations/two_seg_teacherless/`
  - `Tom-CAT/ablations/three_seg_prefix_visible/`

#### Dep-DOG (Stage 2)

- Core model: `Dep-DOG/depdog_model.py`
- Unified train: `Dep-DOG/depdog_train.py`
- Eval: `Dep-DOG/depdog_eval.py`
- Pipeline example: `Dep-DOG/depdog_pipeline_cd4padlast.sh`

#### Ser-FOX (Stage 3)

- Model: `Ser-FOX/serfox_model.py`
- Train: `Ser-FOX/serfox_train.py`
- Eval: `Ser-FOX/serfox_eval.py`
- Additional eval scripts:
  - `Ser-FOX/serfox_test_confidence_guided.py`
  - `Ser-FOX/serfox_test_serialized_ar.py`

### Released baselines

#### Standard MDM baseline

- Model: `MDM/mdm_model.py`
- Train: `MDM/mdm_train.py`
- Eval: `MDM/mdm_eval.py`

#### Standard AR baseline

- Model: `AR/ar_model.py`
- Train: `AR/ar_train.py`
- Eval: `AR/ar_eval.py`

---

## Method map

This section focuses on the role of each method rather than its file locations.

### Tom-CAT

Tom-CAT is the paper's Stage 1 method. It realizes diffusion-style masked denoising inside a causal Transformer through a teacherless write space. The default practical realization in this repo is **Tom-CAT-Placeholder**. The stricter **Tom-CAT-SelfMask** variant and the main ablations live under `Tom-CAT/ablations/`.

### Dep-DOG

Dep-DOG is the paper's Stage 2 trajectory-learning module. It learns better easy-to-hard reveal orders and aligns training to the resulting monotone trajectory. It is part of the paper-stage pipeline and should not be confused with the separate `MDM/` baseline release.

### Ser-FOX

Ser-FOX is the paper's Stage 3 method. It converts learned trajectory preferences into ordinary autoregressive supervision by serializing index-value pairs into a standard 1D causal training sequence.

### MDM

`MDM/` is the released standard masked-diffusion baseline.

### AR

`AR/` is the released standard causal autoregressive baseline.

---

## Representative entrypoints

Once a packed dataset exists under `data/<dataset>`, the main entrypoints are:

```bash
# Tom-CAT
python Tom-CAT/tomcat_train.py --help
python Tom-CAT/tomcat_eval.py --help

# Dep-DOG
python Dep-DOG/depdog_train.py --help
python Dep-DOG/depdog_eval.py --help

# Standard MDM
python MDM/mdm_train.py --help
python MDM/mdm_eval.py --help

# Standard AR
python AR/ar_train.py --help
python AR/ar_eval.py --help

# Ser-FOX
python Ser-FOX/serfox_train.py --help
python Ser-FOX/serfox_eval.py --help
```

For a complete runnable example, prefer the retained Dep-DOG CD4 pipeline over historical one-off command lines.

---

## Historical naming notes

The repository keeps paper-stage names at the directory level, but some internal filenames and checkpoint layouts come from older experiment code.

In particular:

- Tom-CAT code uses `tomcat_*` filenames
- Dep-DOG code uses `depdog_*` filenames
- Ser-FOX code uses `serfox_*` filenames
- Some older checkpoints still live under legacy output roots such as `out/MDMx0/...`, `out/AR/...`, or `out/AR2train...`

The canonical open-source entrypoints remain the ones listed above.

---

## Acknowledgments

We gratefully acknowledge the public resources released by
[HKUNLP/diffusion-vs-ar](https://github.com/HKUNLP/diffusion-vs-ar?tab=readme-ov-file),
which helped inform related reasoning and planning benchmark work.

Related reference:

```bibtex
@article{ye2024beyond,
  title={Beyond Autoregression: Discrete Diffusion for Complex Reasoning and Planning},
  author={Ye, Jiacheng and Gao, Jiahui and Gong, Shansan and Zheng, Lin and Jiang, Xin and Li, Zhenguo and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2410.14157},
  year={2024}
}
```

---

## Citation

If you use this repository, please cite the forthcoming public paper record once
it is available. A BibTeX entry will be added here when the citation details are
finalized.
