from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from typing import Callable
from typing import Literal

def token_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:

    if len(prompt_strs) != len(output_strs):
        raise ValueError(
            f"prompt_strs and output_strs must have the same length, "
            f"but got {len(prompt_strs)} and {len(output_strs)}"
        )

    all_token_ids = []
    all_response_masks = []

    for prompt, output in zip(prompt_strs, output_strs):

        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        output_ids = tokenizer.encode(
            output,
            add_special_tokens=False,
        )

        token_ids = prompt_ids + output_ids

        response_mask = (
            [False] * len(prompt_ids)
            + [True] * len(output_ids)
        )

        all_token_ids.append(token_ids)
        all_response_masks.append(response_mask)

    max_length = max(
        len(token_ids)
        for token_ids in all_token_ids
    )

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        raise ValueError(
            "tokenizer must have a pad_token_type_id, "
            "but got None"
        )

    padded_token_ids = []
    padded_response_masks = []

    for token_ids, response_mask in zip(
        all_token_ids,
        all_response_masks,
    ):
        padding_length = max_length - len(token_ids)

        padded_token_ids.append(
            token_ids
            + [pad_token_id] * padding_length
        )

        padded_response_masks.append(
            response_mask
            + [False] * padding_length
        )

    token_ids_tensor = torch.tensor(
        padded_token_ids,
        dtype=torch.long
    )

    response_masks_tensor = torch.tensor(
        padded_response_masks,
        dtype=torch.bool
    )

    input_ids = token_ids_tensor[:, :-1]

    labels = token_ids_tensor[:, 1:]

    response_mask = response_masks_tensor[:, 1:]

    return{
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }

def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits

    all_log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        probs = all_log_probs.exp()
        token_entropy = -(
            probs * all_log_probs
        ).sum(dim=-1)

        result["token_entropy"] = token_entropy

    return result

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal[
        "mean",
        "none",
    ] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal[
        "std",
        "none",
        "mean",
    ] = "std",
) -> tuple[
    torch.Tensor,
    dict[str, float],
]:

    if group_size <= 0:
        raise ValueError(
            "group_size must be positive."
        )

    if (
        raw_rewards.numel()
        % group_size
        != 0
    ):
        raise ValueError(
            "Number of rewards must be "
            "divisible by group_size."
        )

    # ---------------------------------------------
    # Shape:
    #
    # (rollout_batch_size,)
    #
    # ->
    #
    # (num_prompts, group_size)
    # ---------------------------------------------

    grouped_rewards = (
        raw_rewards.reshape(
            -1,
            group_size,
        )
    )

    # ---------------------------------------------
    # Statistics for each GRPO group.
    # ---------------------------------------------

    group_means = (
        grouped_rewards.mean(
            dim=1,
            keepdim=True,
        )
    )

    group_stds = (
        grouped_rewards.std(
            dim=1,
            keepdim=True,
        )
    )

    # =============================================
    # 1. Baseline
    # =============================================

    if baseline == "mean":

        grouped_advantages = (
            grouped_rewards
            - group_means
        )

    elif baseline == "none":

        grouped_advantages = (
            grouped_rewards.clone()
        )

    else:

        raise ValueError(
            f"Unsupported baseline: "
            f"{baseline}"
        )

    # =============================================
    # 2. Advantage normalization
    # =============================================

    if (
        advantage_normalizer
        == "std"
    ):

        grouped_advantages = (
            grouped_advantages
            / (
                group_stds
                + advantage_eps
            )
        )

    elif (
        advantage_normalizer
        == "none"
    ):

        # No normalization.
        pass

    elif (
        advantage_normalizer
        == "mean"
    ):

        grouped_advantages = (
            grouped_advantages
            / (
                group_means
                + advantage_eps
            )
        )

    else:

        raise ValueError(
            "Unsupported "
            "advantage_normalizer: "
            f"{advantage_normalizer}"
        )

    # ---------------------------------------------
    # Return to flat rollout order.
    # ---------------------------------------------

    advantages = (
        grouped_advantages.reshape(-1)
    )

    metadata = {
        "reward_mean":
            raw_rewards.mean().item(),

        "reward_std":
            raw_rewards.std().item(),

        "advantage_mean":
            advantages.mean().item(),

        "advantage_std":
            advantages.std().item(),

        "group_reward_mean":
            group_means.mean().item(),
    }

    return (
        advantages,
        metadata,
    )

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal[
        "none",
        "noclip",
        "grpo",
        "gspo",
    ] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:

    # -------------------------------------------------
    # Advantage:
    #
    # (B,)
    #
    # ->
    #
    # (B, 1)
    #
    # so it can broadcast across sequence length.
    # -------------------------------------------------

    if raw_rewards_or_advantages.ndim == 1:
        advantages = (
            raw_rewards_or_advantages
            .unsqueeze(-1)
        )
    elif raw_rewards_or_advantages.ndim == 2:
        advantages = (
            raw_rewards_or_advantages
        )
    else:
        raise ValueError(
            "raw_rewards_or_advantages must have "
            "shape (B,) or (B, 1)."
        )

    # =================================================
    # 1. On-policy / naive off-policy
    # =================================================

    if importance_reweighting_method == "none":

        per_token_policy_gradient_loss = (
            -advantages
            * policy_log_probs
        )

        return (
            per_token_policy_gradient_loss,
            {},
        )

    # =================================================
    # Every importance-reweighted method needs π_old.
    # =================================================

    if old_log_probs is None:
        raise ValueError(
            "old_log_probs is required when "
            "importance_reweighting_method "
            "is not 'none'."
        )

    if (
        old_log_probs.shape
        != policy_log_probs.shape
    ):
        raise ValueError(
            "old_log_probs and policy_log_probs "
            "must have the same shape."
        )

    # π_old is fixed.
    old_log_probs = (
        old_log_probs.detach()
    )

    # -------------------------------------------------
    # log(πθ / πold)
    #
    # =
    #
    # log πθ - log πold
    # -------------------------------------------------

    log_ratio = (
        policy_log_probs
        - old_log_probs
    )

    # =================================================
    # 2. Token-level importance reweighting,
    #    no clipping.
    # =================================================

    if importance_reweighting_method == "noclip":

        ratio = torch.exp(
            log_ratio
        )

        objective = (
            advantages
            * ratio
        )

        per_token_policy_gradient_loss = (
            -objective
        )

        metadata = {
            "importance_ratio_mean":
                ratio.detach().mean(),
        }

        return (
            per_token_policy_gradient_loss,
            metadata,
        )

    # =================================================
    # 3. PPO / GRPO token-level clipping
    # =================================================

    if importance_reweighting_method == "grpo":

        if cliprange is None:
            raise ValueError(
                "cliprange is required for "
                "importance_reweighting_method='grpo'."
            )

        ratio = torch.exp(
            log_ratio
        )

        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - cliprange,
            max=1.0 + cliprange,
        )

        unclipped_objective = (
            advantages
            * ratio
        )

        clipped_objective = (
            advantages
            * clipped_ratio
        )

        objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )

        per_token_policy_gradient_loss = (
            -objective
        )

        # This indicates positions where the clipped
        # branch actually determines the PPO objective.
        is_clipped = (
            clipped_objective
            < unclipped_objective
        )

        metadata = {
            "importance_ratio_mean":
                ratio.detach().mean(),

            "clip_fraction":
                (
                    is_clipped
                    .float()
                    .mean()
                    .detach()
                ),
        }

        return (
            per_token_policy_gradient_loss,
            metadata,
        )

    # =================================================
    # 4. GSPO:
    #    sequence-level geometric-mean importance ratio
    # =================================================

    if importance_reweighting_method == "gspo":

        if cliprange is None:
            raise ValueError(
                "cliprange is required for "
                "importance_reweighting_method='gspo'."
            )

        if response_mask is None:
            raise ValueError(
                "response_mask is required for "
                "importance_reweighting_method='gspo'."
            )

        if (
            response_mask.shape
            != policy_log_probs.shape
        ):
            raise ValueError(
                "response_mask and policy_log_probs "
                "must have the same shape."
            )

        mask = response_mask.to(
            dtype=policy_log_probs.dtype
        )

        response_lengths = (
            mask.sum(
                dim=-1,
                keepdim=True,
            )
        )

        if torch.any(
            response_lengths == 0
        ):
            raise ValueError(
                "GSPO requires at least one "
                "response token per sequence."
            )

        # ---------------------------------------------
        # log geometric mean:
        #
        # log s
        # =
        # 1/L * sum_t log(πθ / πold)
        # ---------------------------------------------

        sequence_log_ratio = (
            (
                log_ratio
                * mask
            ).sum(
                dim=-1,
                keepdim=True,
            )
            / response_lengths
        )

        # ---------------------------------------------
        # s = exp(mean log ratio)
        #
        # shape:
        # (B, 1)
        # ---------------------------------------------

        sequence_ratio = torch.exp(
            sequence_log_ratio
        )

        clipped_sequence_ratio = (
            torch.clamp(
                sequence_ratio,
                min=1.0 - cliprange,
                max=1.0 + cliprange,
            )
        )

        unclipped_objective = (
            advantages
            * sequence_ratio
        )

        clipped_objective = (
            advantages
            * clipped_sequence_ratio
        )

        sequence_objective = (
            torch.minimum(
                unclipped_objective,
                clipped_objective,
            )
        )

        # ---------------------------------------------
        # aggregate_loss_across_microbatch()
        # expects a (B, L) tensor.
        #
        # GSPO has one sequence-level objective,
        # so repeat it across token positions.
        #
        # The response mask in the aggregation step
        # will discard prompt/padding positions.
        # ---------------------------------------------

        per_token_policy_gradient_loss = (
            -sequence_objective.expand_as(
                policy_log_probs
            )
        )

        is_clipped = (
            clipped_objective
            < unclipped_objective
        )

        metadata = {
            "sequence_importance_ratio_mean":
                (
                    sequence_ratio
                    .detach()
                    .mean()
                ),

            "clip_fraction":
                (
                    is_clipped
                    .float()
                    .mean()
                    .detach()
                ),
        }

        return (
            per_token_policy_gradient_loss,
            metadata,
        )

    raise ValueError(
        "Unsupported importance_reweighting_method: "
        f"{importance_reweighting_method}"
    )

def aggregate_loss_across_microbatch(
        per_token_policy_gradient_loss: torch.Tensor,
        mask: torch.Tensor,
        loss_normalization: str = "sequence",
        normalization_constant: int | None = None,
) -> torch.Tensor:

    mask = mask.to(
        dtype=per_token_policy_gradient_loss.dtype
    )

    masked_loss = (
        per_token_policy_gradient_loss * mask
    )

    if loss_normalization == "sequence":

        num_response_tokens = mask.sum(
            dim=1
        )

        if torch.any(num_response_tokens == 0):
            raise ValueError(
                "Some sequences have no response tokens, "
                "which would lead to division by zero."
            )

        per_sequence_loss = (
            masked_loss.sum(dim=1)
            / num_response_tokens
        )

        loss = per_sequence_loss.mean()

        return loss

    if loss_normalization == "constant":

        if normalization_constant is None:
            raise ValueError(
                "normalization_constant must be provided "
                "when loss_normalization='constant'."
            )

        loss = masked_loss.sum() / normalization_constant

        return loss

    raise NotImplementedError(
        f"Unsupported loss_normalization: {loss_normalization}"
    )

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn,
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: str = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: str = "std",
    importance_reweighting_method: str = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: str = "sequence",
    normalization_constant: int | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor | float],
]:

    batch_size = len(rollout_responses)

    if not (
        len(repeated_prompts)
        == len(rollout_responses)
        == len(repeated_ground_truths)
    ):
        raise ValueError(
            "repeated_prompts, rollout_responses, "
            "and repeated_ground_truths must have "
            "the same length."
        )

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive."
        )

    if gradient_accumulation_steps > batch_size:
        raise ValueError(
            "gradient_accumulation_steps cannot exceed "
            "the batch size."
        )

    # -------------------------------------------------
    # 1. Tokenize the full rollout batch.
    # -------------------------------------------------

    tokenized = token_prompt_and_output(
        prompt_strs=repeated_prompts,
        output_strs=rollout_responses,
        tokenizer=tokenizer,
    )

    # -------------------------------------------------
    # 2. Compute rewards for all rollouts.
    # -------------------------------------------------

    raw_rewards, reward_metadata = (
        compute_rollout_rewards(
            reward_fn=reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=
                repeated_ground_truths,
        )
    )

    # -------------------------------------------------
    # 3. Convert rewards into advantages.
    # -------------------------------------------------

    advantages, advantage_metadata = (
        compute_group_normalized_rewards(
            raw_rewards=raw_rewards,
            group_size=group_size,
            baseline=baseline,
            advantage_eps=advantage_eps,
            advantage_normalizer=
                advantage_normalizer,
        )
    )

    # -------------------------------------------------
    # 4. Find the model device.
    # -------------------------------------------------

    device = next(
        model.parameters()
    ).device

    # -------------------------------------------------
    # 5. Clear old gradients before accumulation.
    # -------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    # Used only for logging.
    total_loss = torch.zeros(
        (),
        device=device,
    )

    entropy_sum = torch.zeros(
        (),
        device=device,
    )

    entropy_count = torch.zeros(
        (),
        device=device,
    )

    # -------------------------------------------------
    # 6. Work out the microbatch sizes.
    #
    # Example:
    # batch_size = 10
    # steps = 3
    #
    # sizes = [4, 3, 3]
    # -------------------------------------------------

    base_microbatch_size = (
        batch_size
        // gradient_accumulation_steps
    )

    remainder = (
        batch_size
        % gradient_accumulation_steps
    )

    start = 0

    # -------------------------------------------------
    # 7. Forward + backward each microbatch.
    # -------------------------------------------------

    for microbatch_index in range(
        gradient_accumulation_steps
    ):

        microbatch_size = (
            base_microbatch_size
            + (
                1
                if microbatch_index < remainder
                else 0
            )
        )

        end = start + microbatch_size

        # ---------------------------------------------
        # Slice this microbatch and move to the
        # model's device.
        # ---------------------------------------------

        input_ids = (
            tokenized["input_ids"][start:end]
            .to(device)
        )

        labels = (
            tokenized["labels"][start:end]
            .to(device)
        )

        response_mask = (
            tokenized["response_mask"][start:end]
            .to(device)
        )

        microbatch_advantages = (
            advantages[start:end]
            .to(device)
        )

        # ---------------------------------------------
        # Forward through the current policy.
        # ---------------------------------------------

        log_prob_output = (
            get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
        )

        policy_log_probs = (
            log_prob_output["log_probs"]
        )

        token_entropy = (
            log_prob_output["token_entropy"]
        )

        # ---------------------------------------------
        # For future off-policy variants.
        # Standard on-policy has old_log_probs=None.
        # ---------------------------------------------

        if old_log_probs is None:
            microbatch_old_log_probs = None
        else:
            microbatch_old_log_probs = (
                old_log_probs[start:end]
                .to(device)
            )

        # ---------------------------------------------
        # Compute per-token policy-gradient loss.
        # ---------------------------------------------

        (
            per_token_policy_gradient_loss,
            policy_metadata,
        ) = compute_policy_gradient_loss(
            raw_rewards_or_advantages=
                microbatch_advantages,
            policy_log_probs=
                policy_log_probs,
            importance_reweighting_method=
                importance_reweighting_method,
            old_log_probs=
                microbatch_old_log_probs,
            cliprange=cliprange,
            response_mask=response_mask,
        )

        # ---------------------------------------------
        # Aggregate response-token losses into a
        # scalar microbatch loss.
        # ---------------------------------------------

        microbatch_loss = (
            aggregate_loss_across_microbatch(
                per_token_policy_gradient_loss=
                    per_token_policy_gradient_loss,
                mask=response_mask,
                loss_normalization=
                    loss_normalization,
                normalization_constant=
                    normalization_constant,
            )
        )

        # ---------------------------------------------
        # Sequence normalization computes an average
        # over sequences within this microbatch.
        #
        # Reweight so all sequences in the FULL
        # rollout batch are equally weighted.
        # ---------------------------------------------

        if loss_normalization == "sequence":

            accumulation_weight = (
                microbatch_size
                / batch_size
            )

            adjusted_loss = (
                microbatch_loss
                * accumulation_weight
            )

        else:
            # Constant normalization already divides
            # by the global fixed normalization
            # constant, so microbatch contributions
            # should simply be summed.
            adjusted_loss = (
                microbatch_loss
            )

        # ---------------------------------------------
        # Accumulate gradients.
        # ---------------------------------------------

        adjusted_loss.backward()

        # Loss is detached because it is only logged.
        total_loss = (
            total_loss
            + adjusted_loss.detach()
        )

        # ---------------------------------------------
        # Log entropy over response tokens only.
        # ---------------------------------------------

        entropy_mask = (
            response_mask.to(
                dtype=token_entropy.dtype
            )
        )

        entropy_sum = (
            entropy_sum
            + (
                token_entropy.detach()
                * entropy_mask
            ).sum()
        )

        entropy_count = (
            entropy_count
            + entropy_mask.sum()
        )

        start = end

    # -------------------------------------------------
    # 8. Gradient clipping.
    # -------------------------------------------------

    if max_grad_norm is not None:

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
        )

    else:

        grad_norms = [
            parameter.grad.detach().norm(2)
            for parameter in model.parameters()
            if parameter.grad is not None
        ]

        if len(grad_norms) == 0:
            grad_norm = torch.tensor(
                0.0,
                device=device,
            )
        else:
            grad_norm = torch.stack(
                grad_norms
            ).norm(2)

    # -------------------------------------------------
    # 9. Update policy parameters.
    # -------------------------------------------------

    optimizer.step()

    # -------------------------------------------------
    # 10. Clear grads so this step leaves the model
    #     ready for the next rollout batch.
    # -------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    # -------------------------------------------------
    # 11. Metadata for logging.
    # -------------------------------------------------

    metadata = {}

    metadata.update(
        reward_metadata
    )

    metadata.update(
        advantage_metadata
    )

    metadata["gradient_norm"] = (
        grad_norm.detach().item()
    )

    if entropy_count.item() > 0:
        metadata["token_entropy"] = (
            entropy_sum
            / entropy_count
        ).item()
    else:
        metadata["token_entropy"] = 0.0

    return (
        total_loss,
        metadata,
    )