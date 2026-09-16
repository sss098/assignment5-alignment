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


# ============================================================
# Data helpers
# ============================================================


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
    """
    Example:

    [A, B], repeats=3

    ->
    [A, A, A, B, B, B]
    """

    return [
        item
        for item in items
        for _ in range(repeats)
    ]


# ============================================================
# Sampling
# ============================================================


def build_sampling_params(
    seed: int,
    max_tokens: int,
    n: int = 1,
) -> dict:
    return {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,

        "stop": [
            "</answer>"
        ],

        "include_stop_str_in_output": True,
    }


def generate_grouped_rollouts(
    server: VLLMServer,
    prompts: list[str],
    group_size: int,
    max_tokens: int,
    seed: int,
) -> list:
    """
    Generate group_size independent samples for every prompt.

    IMPORTANT:

    We deliberately use batch_size=1 here.

    Why?

    The current vLLM helper returns a flat list of completions.
    By sending one prompt per request and asking for n=group_size,
    we guarantee that the output ordering is:

        prompt 0 sample 0
        prompt 0 sample 1
        ...
        prompt 0 sample G-1

        prompt 1 sample 0
        ...
        prompt 1 sample G-1

    Therefore every consecutive group_size responses form exactly
    one GRPO group.
    """

    sampling_params = build_sampling_params(
        seed=seed,
        max_tokens=max_tokens,
        n=group_size,
    )

    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,

        # Deliberate:
        # one prompt per vLLM request.
        batch_size=1,
    )

    expected_num_completions = (
        len(prompts) * group_size
    )

    if len(completions) != expected_num_completions:
        raise RuntimeError(
            "Unexpected number of rollouts: "
            f"expected {expected_num_completions}, "
            f"got {len(completions)}."
        )

    return completions


# ============================================================
# Diagnostics
# ============================================================


def compute_group_diagnostics(
    rollout_responses: list[str],
    raw_rewards: torch.Tensor,
    group_size: int,
) -> dict[str, float]:
    """
    Diagnostics that are especially important for GRPO.

    A group is informative when its rewards are not all identical.

    Example:

        [0, 0, 1, 0, 0, 1, 0, 0]

    is informative.

    But:

        [0, 0, 0, 0, 0, 0, 0, 0]

    and:

        [1, 1, 1, 1, 1, 1, 1, 1]

    are not informative for mean-baseline GRPO because all
    advantages become zero.
    """

    if (
        len(rollout_responses)
        % group_size
        != 0
    ):
        raise ValueError(
            "Number of rollout responses must be "
            "divisible by group_size."
        )

    grouped_responses = [
        rollout_responses[
            start:start + group_size
        ]
        for start in range(
            0,
            len(rollout_responses),
            group_size,
        )
    ]

    unique_counts = [
        len(set(group))
        for group in grouped_responses
    ]

    mean_unique_responses = (
        sum(unique_counts)
        / len(unique_counts)
    )

    min_unique_responses = min(
        unique_counts
    )

    max_unique_responses = max(
        unique_counts
    )

    grouped_rewards = raw_rewards.reshape(
        -1,
        group_size,
    )

    group_min = grouped_rewards.min(
        dim=1
    ).values

    group_max = grouped_rewards.max(
        dim=1
    ).values

    informative_mask = (
        group_min != group_max
    )

    informative_group_rate = (
        informative_mask
        .float()
        .mean()
        .item()
    )

    group_reward_sums = (
        grouped_rewards.sum(
            dim=1
        )
    )

    all_zero_group_rate = (
        (
            group_reward_sums == 0
        )
        .float()
        .mean()
        .item()
    )

    all_one_group_rate = (
        (
            group_reward_sums
            == group_size
        )
        .float()
        .mean()
        .item()
    )

    return {
        "train/unique_responses_per_group":
            mean_unique_responses,

        "train/min_unique_responses_per_group":
            float(min_unique_responses),

        "train/max_unique_responses_per_group":
            float(max_unique_responses),

        "train/informative_group_rate":
            informative_group_rate,

        "train/all_zero_group_rate":
            all_zero_group_rate,

        "train/all_one_group_rate":
            all_one_group_rate,
    }


# ============================================================
# Evaluation
# ============================================================


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
            n=1,
        )
    )

    completions = (
        server.generate_completions(
            prompts=prompts,
            sampling_params=
                sampling_params,
            batch_size=batch_size,
        )
    )

    responses = [
        completion.text
        for completion in completions
    ]

    raw_rewards, reward_metadata = (
        compute_rollout_rewards(
            reward_fn=
                r1_zero_reward_fn,
            rollout_responses=
                responses,
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
        "val/reward":
            raw_rewards.mean().item(),

        "val/answer_reward":
            reward_metadata[
                "mean_answer_reward"
            ],

        "val/format_reward":
            reward_metadata[
                "mean_format_reward"
            ],

        "val/average_response_length":
            average_response_length,
    }

    return metrics


# ============================================================
# Logging helpers
# ============================================================


def append_jsonl(
    path: Path,
    record: dict,
) -> None:

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def save_rollout_examples(
    path: Path,
    step: int,
    prompts: list[str],
    rollout_responses: list[str],
    ground_truths: list[str],
    group_size: int,
) -> None:
    """
    Save every rollout together with reward information,
    group index and sample index.
    """

    repeated_ground_truths = repeat_each(
        ground_truths,
        group_size,
    )

    expected_size = (
        len(prompts) * group_size
    )

    if len(rollout_responses) != expected_size:
        raise RuntimeError(
            "Rollout ordering/size error."
        )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        for rollout_index, (
            response,
            ground_truth,
        ) in enumerate(
            zip(
                rollout_responses,
                repeated_ground_truths,
                strict=True,
            )
        ):

            group_index = (
                rollout_index
                // group_size
            )

            sample_index = (
                rollout_index
                % group_size
            )

            rewards = (
                r1_zero_reward_fn(
                    response,
                    ground_truth,
                )
            )

            record = {
                "step": step,

                "group_index":
                    group_index,

                "sample_index":
                    sample_index,

                "prompt":
                    prompts[group_index],

                "response":
                    response,

                "ground_truth":
                    ground_truth,

                "reward":
                    rewards["reward"],

                "answer_reward":
                    rewards[
                        "answer_reward"
                    ],

                "format_reward":
                    rewards[
                        "format_reward"
                    ],
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Main
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
    )

    parser.add_argument(
        "--train-path",
        default=
            "data/gsm8k/train.jsonl",
    )

    parser.add_argument(
        "--val-path",
        default=
            "data/gsm8k/test.jsonl",
    )

    parser.add_argument(
        "--prompt-path",
        default=
            R1_ZERO_PROMPT_PATH,
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
        help=(
            "Batch size used for validation. "
            "Training uses one prompt per request "
            "to guarantee GRPO group ordering."
        ),
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
        default=
            "cs336-assignment5-grpo",
    )

    args = parser.parse_args()

    # ========================================================
    # Basic argument checks
    # ========================================================

    if args.group_size <= 1:
        raise ValueError(
            "group_size must be > 1 "
            "for GRPO."
        )

    if (
        args.rollout_batch_size
        % args.group_size
        != 0
    ):
        raise ValueError(
            "rollout_batch_size must "
            "be divisible by group_size."
        )

    prompts_per_step = (
        args.rollout_batch_size
        // args.group_size
    )

    required_examples = (
        prompts_per_step
        * args.num_rollout_steps
    )

    if (
        required_examples
        > args.n_train_examples
    ):
        raise ValueError(
            "Not enough training examples. "
            f"Need {required_examples}, "
            f"but n_train_examples="
            f"{args.n_train_examples}."
        )

    # ========================================================
    # Seeds
    # ========================================================

    random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    rng = random.Random(
        args.seed
    )

    # ========================================================
    # Load data
    # ========================================================

    train_examples = load_jsonl(
        args.train_path
    )

    val_examples = load_jsonl(
        args.val_path
    )

    rng.shuffle(
        train_examples
    )

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

    # ========================================================
    # GPU 0: trainable PyTorch policy
    # ========================================================

    print(
        "Loading trainable policy "
        "on GPU 0..."
    )

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

    # ========================================================
    # GPU 1: vLLM rollout model
    # ========================================================

    print(
        "Starting vLLM rollout "
        "server on GPU 1..."
    )

    server = VLLMServer(
        model_id=args.model_id,
        gpu=1,
        seed=args.seed,
    )

    server.start()

    try:

        # ====================================================
        # NCCL weight synchronization
        # ====================================================

        print(
            "Initializing NCCL "
            "weight synchronization..."
        )

        server.init_weight_sync(
            policy_device="cuda:0"
        )

        # ====================================================
        # Logging setup
        # ====================================================

        wandb.init(
            project=
                args.wandb_project,
            config=
                vars(args),
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
            / (
                f"rollouts_seed_"
                f"{args.seed}.jsonl"
            )
        )

        metrics_log_path = (
            output_dir
            / (
                f"metrics_seed_"
                f"{args.seed}.jsonl"
            )
        )

        config_path = (
            output_dir
            / (
                f"config_seed_"
                f"{args.seed}.json"
            )
        )

        # Start a fresh run rather than
        # mixing records with an old run.
        rollout_log_path.write_text(
            "",
            encoding="utf-8",
        )

        metrics_log_path.write_text(
            "",
            encoding="utf-8",
        )

        with open(
            config_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                vars(args),
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ====================================================
        # Main GRPO training loop
        # ====================================================

        for step in range(
            args.num_rollout_steps
        ):

            print()
            print(
                "=" * 60
            )
            print(
                f"Step {step}"
            )
            print(
                "=" * 60
            )

            # ------------------------------------------------
            # Select this step's unique GSM8K questions
            # ------------------------------------------------

            start = (
                step
                * prompts_per_step
            )

            end = (
                start
                + prompts_per_step
            )

            batch_examples = (
                train_examples[
                    start:end
                ]
            )

            prompts, ground_truths = (
                build_prompts_and_ground_truths(
                    batch_examples,
                    prompt_template,
                )
            )

            # ------------------------------------------------
            # The training tensors need prompt and ground
            # truth repeated G times because we will have G
            # rollout responses per question.
            #
            # But these repeated prompts are NOT sent to vLLM.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Synchronize current PyTorch policy -> vLLM
            # ------------------------------------------------

            server.sync_policy_weights(
                policy
            )

            # ------------------------------------------------
            # Correct GRPO sampling:
            #
            # one unique prompt
            #      ↓
            # n = group_size independent rollouts
            #
            # NOT:
            #
            # group_size duplicate prompts × n=1
            # ------------------------------------------------

            completions = (
                generate_grouped_rollouts(
                    server=server,
                    prompts=prompts,
                    group_size=
                        args.group_size,
                    max_tokens=
                        args.max_tokens,
                    seed=
                        args.seed
                        + step,
                )
            )

            rollout_responses = [
                completion.text
                for completion
                in completions
            ]

            # ------------------------------------------------
            # Compute rewards BEFORE training only for
            # diagnostics.
            #
            # grpo_train_step will compute rewards again.
            # That is slightly redundant but very useful for
            # debugging and insignificant compared with model
            # inference/training cost.
            # ------------------------------------------------

            raw_rewards, (
                pretrain_reward_metadata
            ) = compute_rollout_rewards(
                reward_fn=
                    r1_zero_reward_fn,
                rollout_responses=
                    rollout_responses,
                repeated_ground_truths=
                    repeated_ground_truths,
            )

            group_diagnostics = (
                compute_group_diagnostics(
                    rollout_responses=
                        rollout_responses,
                    raw_rewards=
                        raw_rewards,
                    group_size=
                        args.group_size,
                )
            )

            print(
                "Before update:"
            )

            print(
                "  reward:",
                (
                    pretrain_reward_metadata[
                        "mean_reward"
                    ]
                ),
            )

            print(
                "  unique responses/group:",
                (
                    group_diagnostics[
                        "train/"
                        "unique_responses_per_group"
                    ]
                ),
            )

            print(
                "  informative group rate:",
                (
                    group_diagnostics[
                        "train/"
                        "informative_group_rate"
                    ]
                ),
            )

            # A useful warning.
            if (
                group_diagnostics[
                    "train/"
                    "informative_group_rate"
                ]
                == 0.0
            ):
                print(
                    "WARNING: no informative "
                    "GRPO groups in this step. "
                    "All groups have identical "
                    "rewards, so advantages will "
                    "be zero."
                )

            # ------------------------------------------------
            # GRPO update on GPU 0
            # ------------------------------------------------

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

                    baseline=
                        "mean",

                    advantage_normalizer=
                        "std",

                    importance_reweighting_method=
                        "none",

                    loss_normalization=
                        "sequence",
                )
            )

            # ------------------------------------------------
            # Training metrics
            # ------------------------------------------------

            train_average_response_length = (
                sum(
                    len(
                        completion.token_ids
                    )
                    for completion
                    in completions
                )
                / len(completions)
            )

            metrics = {
                "step":
                    step,

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

                "train/answer_reward":
                    train_metadata[
                        "mean_answer_reward"
                    ],

                "train/format_reward":
                    train_metadata[
                        "mean_format_reward"
                    ],

                "train/reward_std":
                    raw_rewards.std().item(),

                "train/average_response_length":
                    train_average_response_length,
            }

            metrics.update(
                group_diagnostics
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if (
                step
                % args.eval_every
                == 0
            ):

                # Policy has just changed.
                # vLLM still contains the pre-update
                # policy, so synchronize again.
                server.sync_policy_weights(
                    policy
                )

                val_metrics = evaluate(
                    server=server,
                    examples=
                        val_examples,
                    prompt_template=
                        prompt_template,
                    batch_size=
                        args.vllm_batch_size,
                    max_tokens=
                        args.max_tokens,

                    # FIXED validation seed.
                    #
                    # We deliberately do NOT add step
                    # here. This makes validation
                    # comparisons less noisy.
                    seed=
                        args.seed
                        + 100_000,
                )

                metrics.update(
                    val_metrics
                )

            # ------------------------------------------------
            # Save qualitative rollout samples
            # ------------------------------------------------

            if (
                step
                % args.log_rollouts_every
                == 0
            ):

                save_rollout_examples(
                    path=
                        rollout_log_path,
                    step=
                        step,
                    prompts=
                        prompts,
                    rollout_responses=
                        rollout_responses,
                    ground_truths=
                        ground_truths,
                    group_size=
                        args.group_size,
                )

            # ------------------------------------------------
            # Save metrics as plain JSONL.
            #
            # This makes later analysis much easier than
            # depending only on W&B's binary offline files.
            # ------------------------------------------------

            append_jsonl(
                metrics_log_path,
                metrics,
            )

            # W&B logging.
            wandb_metrics = {
                key: value
                for key, value
                in metrics.items()
                if key != "step"
            }

            wandb.log(
                wandb_metrics,
                step=step,
            )

            print()
            print(
                json.dumps(
                    metrics,
                    indent=2,
                    ensure_ascii=False,
                )
            )

    finally:

        server.stop()

        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()