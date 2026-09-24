# Phase 7 per-density dual ascent

Date: 2026-09-23

## Implemented contract

`agents.dual_ascent.PerDensityDualAscent` owns one independent Lagrange
multiplier for each density declared in `training.density_multipliers`.  At the
end of a rollout, every represented density is updated once:

```text
estimate_rho = mean(cost_i for rows i labelled rho)
lambda_rho   = clip(
    lambda_rho + alpha_rho * (estimate_rho - active_miss_budget),
    0,
    configured_maximum_rho,
)
```

The estimate is the undiscounted arithmetic mean of the selected reliability
training signal.  `PHASE7_COST_SIGNAL.md` fixes that signal to conditional miss
probability for the primary learner.  Cost GAE uses the same conditional signal,
but its discounted recursion is not substituted for this density-level mean.
The active curriculum budget is supplied on each update, rather than being
captured from the final service configuration.

## Density isolation and PPO assignment

Rows carry the configured target-density label.  Labels must exactly match a
configured density; the controller does not round a realized traffic density
or silently choose the nearest multiplier.  Each estimate uses only rows with
its own label, so an unbalanced batch cannot pool one density's reliability
with another's.  A configured density absent from a rollout remains unchanged
and does not increment its update counter.

`penalty_weights()` maps the current multiplier back to each row in original
batch order.  The returned tensor is detached and preserves the input device
and dtype, ready for `PPOBatch.cost_penalty_weights`.  Density labels and dual
values remain outside the shared actor observation; they influence training
only through the constrained PPO objective.

## Safety and state boundary

Definitions, labels, costs, and budgets are validated before state mutation.
Costs must be detached finite values in `[0, 1]`.  The controller remains a
generic probability-valued numerical boundary, while the training pipeline
supplies only the configured conditional miss probability.  Invalid or unknown
labels fail atomically.  Updates are projected onto each configured
`[0, maximum]` interval, and immutable snapshots expose multiplier values plus
per-density update counts for later checkpoint integration.

## Verification

Unit tests cover violation and slack directions, the zero floor and configured
cap, independent within-density estimates, unchanged absent densities, exact
row-to-multiplier assignment, headline-configuration construction, detached
tensor behavior, and fail-closed atomic validation.
