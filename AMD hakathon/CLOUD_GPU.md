# Run on a cloud NVIDIA GPU

For AMD GPUs, use `AMD_ROCM.md` instead.

Use an NVIDIA GPU instance on Google Colab, Kaggle, RunPod, Lambda, or another
Linux cloud provider. An NVIDIA T4 with 16 GB VRAM is sufficient for the default
3B model using 4-bit QLoRA. An L4, A10, A100, or H100 will train faster.

## Important limitation of the selected dataset

The Hugging Face repository
`claritystorm/fda-faers-drug-adverse-events` contains a public-domain sample of
only 1,000 rows from the **demographics (`demo`) table**.

It does not contain the `drug`, `reac`, or `outc` tables. Therefore, it cannot
teach:

- drug-to-event assessment
- MedDRA term suggestions
- seriousness criteria
- labeling status
- causality

The included builder uses this dataset only to teach the model to abstain when
medical evidence is missing. For the complete assistant, obtain or build joined
FAERS data containing `demo`, `drug`, `reac`, and `outc`, then add qualified
reviewer annotations for labeling and causality.

## Colab / cloud commands

Upload or clone this project into the cloud instance, then run:

```bash
nvidia-smi
python -m pip install -U pip
pip install -r requirements-nvidia.txt

python src/build_huggingface_dataset.py \
  --dataset claritystorm/fda-faers-drug-adverse-events

python src/train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file data/processed/train.jsonl \
  --validation-file data/processed/validation.jsonl \
  --output-dir outputs/medical-review-lora \
  --load-in-4bit \
  --epochs 2
```

Smoke-test the pipeline before a longer run:

```bash
python src/train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir outputs/smoke-test \
  --load-in-4bit \
  --max-train-samples 50 \
  --epochs 1
```

Test inference:

```bash
python src/infer.py \
  --adapter outputs/medical-review-lora \
  --case-file examples/case.json
```

Download or persist the entire `outputs/medical-review-lora` directory. It
contains the LoRA adapter, not a standalone copy of the base model.
