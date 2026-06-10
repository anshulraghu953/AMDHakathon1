import argparse
import json
from pathlib import Path

from datasets import load_dataset

from common import SYSTEM_PROMPT


DATASET_ID = "claritystorm/fda-faers-drug-adverse-events"


def clean(value):
    return None if value is None else value


def convert(row):
    case = {
        "report_id": str(row["primaryid"]),
        "case_id": str(row["caseid"]),
        "patient": {
            "age_years": clean(row.get("age_years")),
            "sex": clean(row.get("sex_label")),
            "weight_kg": clean(row.get("wt_kg")),
        },
        "reporter_type": clean(row.get("reporter_type")),
        "report_type": clean(row.get("report_type")),
        "reporter_country": clean(row.get("reporter_country")),
        "occurrence_country": clean(row.get("occr_country")),
        "fda_receipt_date": clean(row.get("fda_dt")),
        "source_quarter": clean(row.get("_quarter")),
        "drugs": [],
        "reported_reactions": [],
        "seriousness_evidence": None,
        "label_evidence": None,
    }
    target = {
        "seriousness": "unknown",
        "seriousness_criteria": [],
        "meddra_suggestions": [],
        "labeling_status": "not_assessed",
        "causality": "unassessable",
        "missing_information": [
            "suspect and concomitant drugs",
            "reported adverse reactions",
            "serious outcome data",
            "case narrative and event chronology",
            "approved product label evidence",
        ],
        "reviewer_notes": (
            "The linked Hugging Face sample contains demographics only. "
            "No medical assessment can be made from this record."
        ),
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Prepare a draft review for this case:\n"
                + json.dumps(case, ensure_ascii=True),
            },
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=True)},
        ],
        "metadata": {
            "source": DATASET_ID,
            "report_id": str(row["primaryid"]),
            "warning": "demographics-only sample",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = load_dataset(args.dataset, split="train")
    required = {"primaryid", "caseid", "age_years", "sex_label"}
    missing = required.difference(source.column_names)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    converted = source.map(convert, remove_columns=source.column_names)
    split = converted.train_test_split(
        test_size=args.validation_fraction, seed=args.seed
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split["train"].to_json(f"{args.output_dir}/train.jsonl")
    split["test"].to_json(f"{args.output_dir}/validation.jsonl")
    print(
        "WARNING: this Hugging Face dataset is a demographics-only sample. "
        "It can teach abstention behavior, not medical assessment."
    )


if __name__ == "__main__":
    main()
