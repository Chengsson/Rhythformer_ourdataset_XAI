# RhythmFormer OurDataset FaceXFormer preprocessing

This repository preserves the code and operating record used to convert the
OurDataset camera videos into face-aligned, RhythmFormer-compatible inputs.
The raw dataset, generated videos, checkpoints, caches, and runtime logs are
intentionally not tracked by Git.

## Current result

The production run completed successfully on 2026-07-31 at 21:59 CST:

- 618 discovered sessions
- 607 validated output sessions (98.22%)
- 11 intentionally skipped sessions
- 0 failed sessions
- 125 GB under `/mnt/adata/OurDataset_RhythmFormer_FaceX_128`
- postflight exit code `0`

The eight sessions for demonstration subject `original_subject = 798`
(`subject491` through `subject498`) all passed. See
[docs/preprocessing_status_2026-08-01.md](docs/preprocessing_status_2026-08-01.md)
for the complete audit result and skipped-session reasons.

## Repository layout

```text
configs/
  5OurDataset_FaceX128_RHYTHMFORMER.yaml
preprocessing/
  facexformer_preprocess.py
  launch_facexformer_4gpu.py
  environment.yml
  README.md
  tests/
  operations/
docs/
  preprocessing_status_2026-08-01.md
```

`preprocessing/README.md` contains the implementation, synchronization,
geometry, quality-gate, environment, and recovery details. The scripts under
`preprocessing/operations/` are the exact attempt-03 control scripts, adjusted
only to use the code in this repository and environment-variable overrides.

## Fixed dependencies

- FaceXFormer: `https://github.com/Kartik-3004/facexformer.git`
- FaceXFormer commit: `10fe8291f8a64e2ca1daf938e3e0007bd860303b`
- FaceXFormer checkpoint: Hugging Face repository
  `kartiknarayan/facexformer`, file `ckpts/model.pt`
- Checkpoint SHA-256:
  `327a755849ba64d336fb96589ff87b27e84a12be1ecf8bcfaa503d66f803286d`
- Preserved preprocessing implementation commit:
  `fa117605e302da947f11cbebec5f594f560f402d`
- RhythmFormer FaceX128 config commit:
  `ab063be0bf119e28c4ea724fdd3916873f16ca8a`

The upstream FaceXFormer source and model checkpoint are dependencies, not
vendored files in this repository.

## Environment

```bash
conda env create --file preprocessing/environment.yml
conda activate facexformer-preprocess
export PYTHONNOUSERSITE=1
```

Clone FaceXFormer and download its checkpoint as documented in
`preprocessing/README.md` before running the pipeline.

## Run the four-GPU pipeline

The production defaults match the nordlinglab server. Override them when using
another host:

```bash
export INPUT_ROOT="/mnt/adata/Non-contact Video Archive/Video"
export OUTPUT_ROOT="/mnt/adata/OurDataset_RhythmFormer_FaceX_128"
export FACEXFORMER_ROOT="/home/nordlinglab/Louis/facex_preprocess/facexformer"
export FACEXFORMER_CHECKPOINT="$FACEXFORMER_ROOT/ckpts/model.pt"
export PYTHON_BIN="/home/nordlinglab/miniconda3/envs/facexformer-preprocess/bin/python"

bash preprocessing/operations/run_facex_full_v3.sh
```

The launcher uses GPUs `0,1,2,3`, FFV1 output, a minimum temporal RGB
correlation of 0.99, and a minimum detection rate of 0.80. It resumes already
validated sessions by default.

Run the postflight checks after the launcher writes `launcher_v3.exit`:

```bash
bash preprocessing/operations/facex_postflight_v3.sh
```

The optional completion notifier requires `NTFY_TOPIC` in the environment; no
notification topic or credential is stored in this repository.

## Tests

```bash
python -m unittest discover -s preprocessing/tests -v
python -m py_compile \
  preprocessing/facexformer_preprocess.py \
  preprocessing/launch_facexformer_4gpu.py
```

## Output contract

Every validated session contains:

```text
subjectNNN/
  vid.avi              # FFV1, 128x128, source FPS
  ground_truth.txt     # aligned PPG signal
  .facex_quality.json  # detection/interpolation/integrity metrics
```

The output root also contains `session_mapping.csv` and `quality_report.csv`.
These are generated artifacts and remain excluded from Git.

## Data policy

Do not commit raw MP4 files, generated AVI files, alignment data, model
checkpoints, caches, contact sheets, or runtime logs. The `.gitignore` enforces
the common forms of these artifacts. Source data remains under `/mnt/adata`.
