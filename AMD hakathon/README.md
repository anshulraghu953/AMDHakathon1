# AI-led Medical Review Assistant (local fine-tuning starter)

This project fine-tunes a small open-weight instruction model to produce a
structured **draft** review of an adverse-event case:

- assess reported seriousness and criteria
- suggest MedDRA Preferred Term (PT) names
- record labeling status when label evidence is supplied
- support, but never replace, expert causality assessment

It is designed for research and reviewer assistance. It must not make autonomous
medical or regulatory decisions.

## Important data limitations

FAERS reports are useful weak supervision for seriousness and reported MedDRA PT
names. They do **not** establish causality, and a report with multiple products
and reactions does not link a particular product to a particular reaction.

Numeric MedDRA codes and the full terminology require authorized MedDRA access.
Do not redistribute MedDRA files. Put your licensed English ASCII distribution
under `data/meddra/` if you want to add a deterministic terminology lookup.

Labeling status and causality should be annotated by trained reviewers using
approved local procedures. The model should return `not_assessed` or
`unassessable` when evidence is missing.

## Data sources

1. **FAERS / openFDA drug-event API** - weak labels for seriousness and reported
   reaction PT names:
   https://open.fda.gov/apis/drug/event/
2. **FDA quarterly FAERS files** - larger bulk alternative:
   https://www.fda.gov/drugs/drug-approvals-and-databases/fda-adverse-event-reporting-system-faers-database
3. **DailyMed SPL labels** - source evidence for US labeling assessment:
   https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm
4. **MedDRA subscription/access** - required for official terminology/code use:
   https://www.meddra.org/subscription/process

## Setup

Python 3.10+ is recommended. For AMD GPUs, use Linux with a PyTorch build
supported by your ROCm environment, then install the remaining packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The exact supported ROCm/PyTorch combination depends on your cloud image and
GPU. See `AMD_ROCM.md` for AMD cloud commands. CPU training is very slow.

## Build a starter dataset

Download public FAERS cases from openFDA and convert them to chat-style JSONL:

```powershell
python src/download_openfda.py --max-records 2000
python src/build_dataset.py
```

This creates:

- `data/raw/openfda_events.jsonl`
- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`

For a medically meaningful model, replace or supplement weak labels with
reviewer-approved examples that follow `data/expert_annotations.example.jsonl`.
Never put identifiable patient information into training data.

To use the selected Hugging Face demographics sample directly:

```powershell
python src/build_huggingface_dataset.py
```

That sample does not include drugs, reactions, or outcomes, so it only produces
safe abstention examples. See `CLOUD_GPU.md` for cloud GPU commands and the
dataset limitation.

## Fine-tune with LoRA

```powershell
python src/train.py `
  --model Qwen/Qwen2.5-3B-Instruct `
  --train-file data/processed/train.jsonl `
  --validation-file data/processed/validation.jsonl `
  --output-dir outputs/medical-review-lora
```

For a first smoke test, add `--max-train-samples 50 --epochs 1`.

## Run locally

```powershell
python src/infer.py `
  --adapter outputs/medical-review-lora `
  --case-file examples/case.json
```

The output is a draft requiring reviewer confirmation. Keep temperature at zero
for reproducibility and validate every response against the JSON schema.
