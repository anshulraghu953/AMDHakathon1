import argparse
import json
import random

from common import SYSTEM_PROMPT, read_jsonl, write_jsonl


CRITERIA_FIELDS = {
    "seriousnessdeath": "death",
    "seriousnesslifethreatening": "life_threatening",
    "seriousnesshospitalization": "hospitalization",
    "seriousnessdisabling": "disability",
    "seriousnesscongenitalanomali": "congenital_anomaly",
    "seriousnessother": "other_medically_important_condition",
}


def compact_case(report):
    patient = report.get("patient", {})
    drugs = []
    for drug in patient.get("drug", []):
        drugs.append(
            {
                "name": drug.get("medicinalproduct"),
                "characterization": drug.get("drugcharacterization"),
                "indication": drug.get("drugindication"),
                "route": drug.get("drugadministrationroute"),
            }
        )
    return {
        "report_id": report.get("safetyreportid"),
        "patient": {
            "age": patient.get("patientonsetage"),
            "age_unit": patient.get("patientonsetageunit"),
            "sex": patient.get("patientsex"),
        },
        "drugs": drugs,
        "reported_reactions": [
            reaction.get("reactionmeddrapt")
            for reaction in patient.get("reaction", [])
            if reaction.get("reactionmeddrapt")
        ],
        "reported_outcome": patient.get("patientdeath"),
        "report_serious": report.get("serious"),
    }


def weak_target(report):
    reactions = report.get("patient", {}).get("reaction", [])
    criteria = [
        name for field, name in CRITERIA_FIELDS.items() if report.get(field) == "1"
    ]
    serious_value = report.get("serious")
    seriousness = (
        "serious" if serious_value == "1" else "non_serious" if serious_value == "2" else "unknown"
    )
    suggestions = []
    for reaction in reactions:
        term = reaction.get("reactionmeddrapt")
        if term:
            suggestions.append(
                {"reported_term": term, "pt_name": term, "pt_code": None}
            )
    return {
        "seriousness": seriousness,
        "seriousness_criteria": criteria,
        "meddra_suggestions": suggestions,
        "labeling_status": "not_assessed",
        "causality": "unassessable",
        "missing_information": [
            "case narrative and event chronology",
            "dechallenge/rechallenge information",
            "relevant medical history and concomitant therapy",
            "approved product label evidence",
        ],
        "reviewer_notes": "Weakly supervised FAERS example; human review required.",
    }


def convert(report):
    case = compact_case(report)
    target = weak_target(report)
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
            "source": "openFDA FAERS weak supervision",
            "report_id": case["report_id"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/openfda_events.jsonl")
    parser.add_argument("--train-output", default="data/processed/train.jsonl")
    parser.add_argument("--validation-output", default="data/processed/validation.jsonl")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = [convert(record) for record in read_jsonl(args.input)]
    random.Random(args.seed).shuffle(records)
    split = max(1, int(len(records) * args.validation_fraction))
    write_jsonl(args.validation_output, records[:split])
    write_jsonl(args.train_output, records[split:])
    print(f"Wrote {len(records) - split} train and {split} validation examples")


if __name__ == "__main__":
    main()

