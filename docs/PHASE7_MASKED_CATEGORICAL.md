# Phase 7 masked categorical policy

Date: 2026-09-23

## Contract

The initial shared actor is a feed-forward multilayer perceptron with the
configured two 64-unit tanh hidden layers. It emits nine logits in the stable
`PolicyAction` order for every learning-usable pair observation. A feasibility
mask is applied to those logits before normalization, so an impossible action
has exactly zero probability and receives no policy gradient.

Masks remain hardware/profile facts only. They may state that RF or VLC
hardware is absent or that fewer RF attempts can be reserved; they cannot use
current channel truth, sampled failures, occlusion, or conditional risk. Those
uncertain quantities remain inputs or targets for learning, never privileged
action suppression.

The distribution accepts either the repository `ActionMask`, broadcast across
the population, or a boolean tensor with one mask per row. Every row must keep
at least one of the nine actions available. A zero-row population is valid and
returns empty action, log-probability, entropy, and probability tensors.

## Selection and evaluation

Training samples from the renormalized allowed probabilities. The caller may
provide a dedicated `torch.Generator`, keeping policy exploration independent
of PyTorch's process-global random stream. Evaluation chooses the largest
allowed logit deterministically; ties resolve to the lowest persistent action
index. PPO minibatch evaluation rejects masked or out-of-range recorded
actions rather than silently returning an infinite loss.

The distribution returns one action index, log probability, and entropy value
per active pair. The same evaluation path is used immediately after selection
and when rollout actions are replayed during an update, avoiding separate
training and inference probability semantics.

Missing-observation fallback is intentionally outside this module. The
environment already applies its configured safe fallback to unusable rows;
only learning-usable actor rows should enter the categorical policy.

## Verification

Unit tests cover broadcast and row-specific masks, zero probability and zero
gradient for masked logits, deterministic tie-breaking, reproducible sampling,
canonical `ActionMask` indices, analytical probabilities/log probabilities and
entropy, malformed inputs, empty populations, and the configured two-by-64
tanh actor shape. Reward/cost advantages and PPO optimization remain separate
Phase 7 tasks.
