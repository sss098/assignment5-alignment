from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import wandb

from cs336_alignment.checkpoint import (
    get_model_and_tokenizer,
)
from cs336_alignment.drgrpo_grader import (
    r1_zero_reward_fn,
)
from cs336_alignment.grpo import (
    compute_rollout_rewards,
    grpo_train_step,
)
from cs336_alignment.vllm_utils import (
    VLLMServer,
)


MODEL_ID = "allenai/OLMo-2-0425-1B"

R1_ZERO_PROMPT_PATH = (
    "cs336_alignment/prompts/r1_zero.prompt"
)


def load_jsonl(
    path: str,
) -> list[dict]:

    examples = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                examples.append(
                    json.loads(line)
                )

    return examples


def extract_ground_truth(
    answer: str,
) -> str:

    return (
        answer
        .split("####")[-1]
        .strip()
    )


def load_prompt_template(
    path: str,
) -> str:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return f.read()


def build_prompts_and_ground_truths(
    examples: list[dict],
    prompt_template: str,
) -> tuple[list[str], list[str]]:

    prompts = []

    ground_truths = []

    for example in examples:

        prompt = prompt_template.format(
            question=example["question"]
        )

        ground_truth = extract_ground_truth(
            example["answer"]
        )

        prompts.append(prompt)

        ground_truths.append(
            ground_truth
        )

    return (
        prompts,
        ground_truths,
    )


def repeat_each(
    items: list[str],
    repeats: int,
) -> list[str]:

    return [
        item
        for item in items
        for _ in range(repeats)
    ]


def build_sampling_params(
    seed: int,
    max_tokens: int,
) -> dict:

    return {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": seed,

        "stop": [
            "</answer>"
        ],

        "include_stop_str_in_output": True,
    }


@torch.no_grad()
def evaluate(
    server: VLLMServer,
    examples: list[dict],
    prompt_template: str,
    batch_size: int,
    max_tokens: int,
    seed: int,
) -> dict[str, float]:

    prompts, ground_truths = (
        build_prompts_and_ground_truths(
            examples,
            prompt_template,
        )
    )

    sampling_params = (
        build_sampling_params(
            seed=seed,
            max_tokens=max_tokens,
        )
    )

    completions = (
        server.generate_completions(
            prompts=prompts,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )
    )

    responses = [
        completion.text
        for completion in completions
    ]

    raw_rewards, reward_metadata = (
        compute_rollout_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=
                ground_truths,
        )
    )

    average_response_length = (
        sum(
            len(completion.token_ids)
            for completion in completions
        )
        / len(completions)
    )

    metrics = {
        "val/reward": (
            raw_rewards.mean().item()
        ),
        "val/format_reward": (
            reward_metadata[
                "mean_format_reward"
            ]
        ),
        "val/average_response_length":
            average_response_length,
    }

    return metrics


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
    )

    parser.add_argument(
        "--train-path",
        default="data/gsm8k/train.jsonl",
    )

    parser.add_argument(
        "--val-path",
        default="data/gsm8k/test.jsonl",
    )

    parser.add_argument(
        "--prompt-path",
        default=R1_ZERO_PROMPT_PATH,
    )

    parser.add_argument(
        "--n-train-examples",
        type=int,
        default=6400,
    )

    parser.add_argument(
        "--n-val-examples",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--num-rollout-steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--log-rollouts-every",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--vllm-batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/grpo",
    )

    parser.add_argument(
        "--wandb-project",
        default="cs336-assignment5-grpo",
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Basic checks.
    # ---------------------------------------------

    if (
        args.rollout_batch_size
        % args.group_size
        != 0
    ):
        raise ValueError(
            "rollout_batch_size must be "
            "divisible by group_size."
        )

    prompts_per_step = (
        args.rollout_batch_size
        // args.group_size
    )

    required_examples = (
        prompts_per_step
        * args.num_rollout_steps
    )

    if required_examples > args.n_train_examples:
        raise ValueError(
            "Not enough training examples for "
            "the requested number of rollout steps."
        )

    # ---------------------------------------------
    # Random seeds.
    # ---------------------------------------------

    random.seed(args.seed)

    torch.manual_seed(args.seed)

    torch.cuda.manual_seed_all(
        args.seed
    )

    rng = random.Random(
        args.seed
    )

    # ---------------------------------------------
    # Load data.
    # ---------------------------------------------

    train_examples = load_jsonl(
        args.train_path
    )

    val_examples = load_jsonl(
        args.val_path
    )

    rng.shuffle(train_examples)

    train_examples = (
        train_examples[
            :args.n_train_examples
        ]
    )

    val_examples = (
        val_examples[
            :args.n_val_examples
        ]
    )

    prompt_template = (
        load_prompt_template(
            args.prompt_path
        )
    )

    # ---------------------------------------------
    # GPU 0: trainable PyTorch policy.
    # ---------------------------------------------

    policy, tokenizer = (
        get_model_and_tokenizer(
            args.model_id,
            device="cuda:0",
        )
    )

    policy.train()

    policy.config.use_cache = False

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    # ---------------------------------------------
    # GPU 1: vLLM rollout policy.
    # ---------------------------------------------

    server = VLLMServer(
        model_id=args.model_id,
        gpu=1,
        seed=args.seed,
    )

    server.start()

    try:

        # Create NCCL connection between
        # training model and vLLM.
        server.init_weight_sync(
            policy_device="cuda:0"
        )

        # -----------------------------------------
        # Logging.
        # -----------------------------------------

        wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )

        output_dir = Path(
            args.output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rollout_log_path = (
            output_dir
            / f"rollouts_seed_{args.seed}.jsonl"
        )

        # =========================================
        # Main GRPO loop.
        # =========================================

        for step in range(
            args.num_rollout_steps
        ):

            print()
            print(
                f"========== Step {step} =========="
            )

            # -------------------------------------
            # Select this step's questions.
            # -------------------------------------

            start = (
                step * prompts_per_step
            )

            end = (
                start + prompts_per_step
            )

            batch_examples = (
                train_examples[start:end]
            )

            prompts, ground_truths = (
                build_prompts_and_ground_truths(
                    batch_examples,
                    prompt_template,
                )
            )

            # -------------------------------------
            # Each question gets G rollouts.
            # -------------------------------------

            repeated_prompts = (
                repeat_each(
                    prompts,
                    args.group_size,
                )
            )

            repeated_ground_truths = (
                repeat_each(
                    ground_truths,
                    args.group_size,
                )
            )

            # -------------------------------------
            # Make vLLM exactly match the
            # current training policy.
            # -------------------------------------

            server.sync_policy_weights(
                policy
            )

            # -------------------------------------
            # Generate rollouts on GPU 1.
            # -------------------------------------

            sampling_params = (
                build_sampling_params(
                    seed=args.seed + step,
                    max_tokens=
                        args.max_tokens,
                )
            )

            completions = (
                server.generate_completions(
                    prompts=
                        repeated_prompts,
                    sampling_params=
                        sampling_params,
                    batch_size=
                        args.vllm_batch_size,
                )
            )

            rollout_responses = [
                completion.text
                for completion in completions
            ]

            # -------------------------------------
            # GRPO update on GPU 0.
            # -------------------------------------

            loss, train_metadata = (
                grpo_train_step(
                    model=policy,
                    tokenizer=tokenizer,
                    optimizer=optimizer,

                    gradient_accumulation_steps=
                        args.gradient_accumulation_steps,

                    max_grad_norm=
                        args.max_grad_norm,

                    reward_fn=
                        r1_zero_reward_fn,

                    repeated_prompts=
                        repeated_prompts,

                    rollout_responses=
                        rollout_responses,

                    repeated_ground_truths=
                        repeated_ground_truths,

                    group_size=
                        args.group_size,

                    baseline="mean",

                    advantage_normalizer=
                        "std",

                    importance_reweighting_method=
                        "none",

                    loss_normalization=
                        "sequence",
                )
            )

            train_average_response_length = (
                sum(
                    len(completion.token_ids)
                    for completion
                    in completions
                )
                / len(completions)
            )

            metrics = {
                "train/loss":
                    loss.item(),

                "train/gradient_norm":
                    train_metadata[
                        "gradient_norm"
                    ],

                "train/token_entropy":
                    train_metadata[
                        "token_entropy"
                    ],

                "train/reward":
                    train_metadata[
                        "mean_reward"
                    ],

                "train/format_reward":
                    train_metadata[
                        "mean_format_reward"
                    ],

                "train/average_response_length":
                    train_average_response_length,
            }

            # -------------------------------------
            # Validation.
            #
            # Policy was just updated, so sync
            # again before evaluating.
            # -------------------------------------

            if (
                step % args.eval_every == 0
            ):

                server.sync_policy_weights(
                    policy
                )

                val_metrics = evaluate(
                    server=server,
                    examples=val_examples,
                    prompt_template=
                        prompt_template,
                    batch_size=
                        args.vllm_batch_size,
                    max_tokens=
                        args.max_tokens,
                    seed=
                        args.seed
                        + 100_000
                        + step,
                )

                metrics.update(
                    val_metrics
                )

            # -------------------------------------
            # Save qualitative rollouts.
            # -------------------------------------

            if (
                step
                % args.log_rollouts_every
                == 0
            ):

                with open(
                    rollout_log_path,
                    "a",
                    encoding="utf-8",
                ) as f:

                    for (
                        prompt,
                        response,
                        ground_truth,
                    ) in zip(
                        repeated_prompts,
                        rollout_responses,
                        repeated_ground_truths,
                        strict=True,
                    ):

                        record = {
                            "step": step,
                            "prompt": prompt,
                            "response":
                                response,
                            "ground_truth":
                                ground_truth,
                        }

                        f.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            # -------------------------------------
            # Log.
            # -------------------------------------

            wandb.log(
                metrics,
                step=step,
            )

            print(
                json.dumps(
                    metrics,
                    indent=2,
                )
            )

    finally:

        server.stop()

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()