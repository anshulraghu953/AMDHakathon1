import argparse
import json

import torch
from jsonschema import validate
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import OUTPUT_SCHEMA, SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--case-file", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    with open(args.case_file, encoding="utf-8") as handle:
        case = json.load(handle)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Prepare a draft review for this case:\n"
            + json.dumps(case, ensure_ascii=True),
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(
        inputs, max_new_tokens=700, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(output[0, inputs.shape[-1] :], skip_special_tokens=True)
    result = json.loads(text)
    validate(result, OUTPUT_SCHEMA)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
