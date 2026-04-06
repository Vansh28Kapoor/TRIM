# POMDP Parameter Scripts

This folder contains runners for precomputing POMDP parameters used by TRIM-Agg.

Model override behavior:

- Pass model flags directly to these scripts.
- The scripts parse those flags, export `TARGET_MODEL` / `DRAFT_MODEL` / `PRM_MODEL`, and then source `scripts/launch_servers.sh`.
- This guarantees the same model names are used for both server launch and Python client arguments.
- These scripts also accept `--` overrides for dataset/threshold/split options (examples below).

## `run_observation_params.sh`

Launches vLLM servers via `scripts/launch_servers.sh`, then runs:

1. `pomdp_params/get_observation_function.py`

### Usage

```bash
bash scripts/pomdp/run_observation_params.sh
```

Override PRM by script flag:

```bash
bash scripts/pomdp/run_observation_params.sh --prm-model-name Qwen/Qwen2.5-Math-PRM-7B
```

Override observation params by script flags:

```bash
bash scripts/pomdp/run_observation_params.sh --obs-benchmark omnimath --bin-size 0.05 --rerun
```

## `run_transition_params.sh`

Launches vLLM servers via `scripts/launch_servers.sh`, then runs:

1. `pomdp_params/get_transition_function.py`

### Usage

```bash
bash scripts/pomdp/run_transition_params.sh
```

Override target/draft/PRM models by script flags:

```bash
bash scripts/pomdp/run_transition_params.sh \
	--target-model-name Qwen/Qwen2.5-7B-Instruct \
	--draft-model-name Qwen/Qwen2.5-1.5B-Instruct \
	--prm-model-name Qwen/Qwen2.5-Math-PRM-7B
```

Override transition params by script flags:

```bash
bash scripts/pomdp/run_transition_params.sh \
	--transition-benchmark aime \
	--transition-split train \
	--thr 0.35 \
	--max-steps 30 \
	--terminal-predictor-dir pomdp_data \
	--rerun
```

## `run_pomdp_params.sh`

Launches vLLM servers via `scripts/launch_servers.sh`, then runs in sequence:

1. `pomdp_params/get_observation_function.py`
2. `pomdp_params/get_transition_function.py`

### Usage

```bash
bash scripts/pomdp/run_pomdp_params.sh
```

Override models directly by script flags:

```bash
bash scripts/pomdp/run_pomdp_params.sh --target-model-name Qwen/Qwen2.5-7B-Instruct
```

Override both observation and transition params by script flags:

```bash
bash scripts/pomdp/run_pomdp_params.sh \
	--obs-benchmark omnimath \
	--bin-size 0.05 \
	--transition-benchmark aime \
	--transition-split train \
	--thr 0.35 \
	--max-steps 30 \
	--terminal-predictor-dir pomdp_data \
	--rerun
```

### Optional environment overrides

```bash
OBS_BENCHMARK=omnimath \
BIN_SIZE=0.05 \
TRANSITION_BENCHMARK=aime \
TRANSITION_SPLIT=train \
THR=0.35 \
MAX_STEPS=30 \
TERMINAL_PREDICTOR_DIR=pomdp_data \
RERUN=0 \
bash scripts/pomdp/run_pomdp_params.sh
```

Notes:

- Both snake_case and kebab-case model flags are accepted, e.g. `--target_model_name` and `--target-model-name`.
- Transition benchmark/split flags support aliases: `--transition-benchmark`/`--train-benchmark` and `--transition-split`/`--train-split`.
- Unknown flags now fail fast with an error to avoid accidentally applying one script's arg to the wrong Python command.
