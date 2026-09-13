from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.vllm_utils import VLLMServer

MODEL_ID = "allenai/OLMo-2-0425-1B"

PROMPT_FILES = {
    "question_only":
        "cs336_alignment/prompts/question_only.prompt",

    "r1_zero":
        "cs336_alignment/prompts/r1_zero.prompt",

    "r1_zero_three_shot":
        "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt",
}

def load_jsonl(path: str) -> list[dict]:
    examples = []

    with open(path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples

def extract_ground_truth(answer: str) -> str:
    return answer.split("####")[-1].strip()

def load_prompt_template(prompt_type: str) -> str:
    path = PROMPT_FILES[prompt_type]

    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(
        examples: list[dict],
        prompt_type: str,
) -> list[str]:

    template = load_prompt_template(prompt_type)

    return [
        template.format(question=example["question"])
        for example in examples
    ]

def build_sampling_params(
        prompt_type: str,
        seed: int,
) -> dict:

    sampling_params = {
        "temperature" : 1.0,
        "top_p" : 1.0,
        "max_tokens" : 512,
        "n" : 1,
        "seed" : seed,
    }

    if prompt_type != "question_only":
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True

    return sampling_params

def evaluate_prompt_type(
        server: VLLMServer,
        examples: list[dict],
        prompt_type: str,
        output_path: str,
        batch_size: int | None,
        seed: int,
) -> None:

    prompts = build_prompt(
        examples,
        prompt_type,
    )

    sampling_params = build_sampling_params(
        prompt_type,
        seed,
    )

    completions = server.generate_completions(
        prompt=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )

    if prompt_type == "question_only":
        reward_fn = question_only_reward_fn
    else:
        reward_fn = r1_zero_reward_fn

    category_counts = Counter()
    records = []

    for example, prompt, completion in zip(
        examples,
        prompts,
        completions,
        strict=True,
    ):

        ground_truth = extract_ground_truth(
            example["answer"]
        )

        rewards = reward_fn(
            completion.text,
            ground_truth,
        )

        format_reward = rewards["format_reward"]
        answer_reward = rewards["answer_reward"]

        if format_reward == 1 and answer_reward == 1:
            category = "correct"

        elif format_reward == 1 and answer_reward == 0:
            category = "formatted_but_wrong"

        else:
            category = "unformatted_and_wrong"

        category_counts[category] += 1

        records.append(
            {
                "question": example["question"],
                "ground_truth": ground_truth,
                "prompt": prompt,
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "num_response_tokens": len(completion.token_ids),
                "reward": rewards["reward"],
                "format_reward": format_reward,
                "answer_reward": answer_reward,
                "category": category,
            }
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(records)

    print()
    print("=" * 60)
    print("Prompt type:", prompt_type)
    print("Total examples:", total)

    for category, count in category_counts.items():
        print(
            f"{category}: "
            f"{count} "
            f"({count / total:.2%})"
        )

    accuracy = (
        sum(
            record["answer_reward"]
            for record in records
        )
        / total
    )

    format_rate = (
        sum(
            record["format_reward"]
            for record in records
        )
        / total
    )

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Format rate: {format_rate:.2%}")
    print("Saved to:", output_file)
    print("=" * 60)

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
    )

    parser.add_argument(
        "--data-path",
        default="data/gsm8k/test.jsonl",
    )

    parser.add_argument(
        "--prompt-type",
        choices=[
            "question_only",
            "r1_zero",
            "r1_zero_three_shot",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/prompting",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )   

    args = parser.parse_args()

    examples = load_jsonl(args.data_path)

    if args.limit is not None:
        examples = examples[:args.limit]

    if args.prompt_type == "all":
        prompt_types = [
            "question_only",
            "r1_zero",
            "r1_zero_three_shot",
        ]
    else:
        prompt_types = [args.prompt_type]

    server = VLLMServer(
        model_id=args.model_id,
        gpu=args.gpu,
        seed=args.seed,
    )

    server.start()

    try:
        for prompt_type in prompt_types:

            output_path = Path(args.output_dir) / f"{prompt_type}.jsonl"

            evaluate_prompt_type(
                server,
                examples,
                prompt_type,
                output_path,
                args.batch_size,
                args.seed,
            )

    finally:
        server.stop()


if __name__ == "__main__":
    main()



