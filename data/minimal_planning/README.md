# Minimal Planning

This directory contains the dataset generation script for the
**Minimal Planning Task**.

The input consists of a set of shuffled edges together with a start node and a
goal node, and the model is required to output the correct path as an ordered
sequence of edges. This synthetic setup is used as a controlled benchmark for
evaluating multi-step planning beyond immediate greedy choices.

## Data Preparation

To create the dataset, run:

```bash
python data/minimal_planning/generate_dataset.py
```

This script does not currently use command-line arguments. The main
configurations are defined directly in
`data/minimal_planning/generate_dataset.py`:

```python
STREAM = 2
NODE_PER_STREAM = 14
NUM_TRAIN = 1000000
NUM_TEST = 1000
SEED = 1
```

With the default settings, the script writes:

* `path_train-2-14.jsonl`
* `path_test-2-14.jsonl`

Note: these output files are currently written relative to the working
directory from which the script is executed. If you run the command from the
repository root, the generated files will appear at the repository root.

## Next Step

The generated files already use the repository's standard JSONL schema:

```json
{"input": "...", "output": "..."}
```

You can then pack the training split with `data/prepare_data_generic.py` for
the generic AR, Tom-CAT, or MDM entrypoints, while keeping
`path_test-2-14.jsonl` as the held-out evaluation file.
