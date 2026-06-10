# Run on an AMD ROCm cloud GPU

Use a Linux AMD GPU instance with ROCm and a ROCm-compatible PyTorch build.
AMD Instinct MI210, MI250, MI300X, and similar cloud accelerators are suitable.

PyTorch intentionally exposes ROCm devices through the `torch.cuda` API. Seeing
`torch.cuda.is_available() == True` on AMD is expected.

## Install

Start from an AMD ROCm PyTorch image supplied by your cloud provider, then run:

```bash
python -m pip install -U pip
pip install -r requirements.txt
```

Do not install `requirements-nvidia.txt`, and do not use `--load-in-4bit`.
The project's 4-bit path uses NVIDIA-oriented `bitsandbytes`.

Verify ROCm:

```bash
rocminfo | head
python -c "import torch; print('GPU:', torch.cuda.is_available()); print('HIP:', torch.version.hip); print('Device:', torch.cuda.get_device_name(0))"
```

`HIP` must print a version and `GPU` must print `True`.

## Prepare the Hugging Face sample

```bash
python src/build_huggingface_dataset.py \
  --dataset claritystorm/fda-faers-drug-adverse-events
```

This public sample contains demographics only. It creates safe abstention
examples, not useful labels for MedDRA, seriousness, labeling, or causality.

## Train with BF16 LoRA

```bash
python src/train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train-file data/processed/train.jsonl \
  --validation-file data/processed/validation.jsonl \
  --output-dir outputs/medical-review-amd-lora \
  --epochs 2
```

First run a smoke test:

```bash
python src/train.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --output-dir outputs/amd-smoke-test \
  --max-train-samples 50 \
  --epochs 1
```

If the GPU runs out of memory, use:

```bash
python src/train.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/medical-review-amd-lora \
  --epochs 2
```

Monitor utilization with `watch -n 1 rocm-smi`.

## Inference

```bash
python src/infer.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter outputs/medical-review-amd-lora \
  --case-file examples/case.json
```
