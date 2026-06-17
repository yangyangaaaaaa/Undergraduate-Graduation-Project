# Code

This directory contains the cleaned implementation used for the graduation project experiments.

## Entry Points

- `pretrain.py`: trains the action-feature prediction module.
- `train.py`: trains the PPO policy with external reward, curiosity reward, distance-aware gating, and potential-based reward shaping.
- `validate.py`: runs fixed-policy evaluation.
- `config.py`: central runtime configuration, including environment-variable overrides.

## Main Modules

- `models/`: policy, PPO, and sequence modeling modules.
- `data_utils/`: image/text embedding preparation and trajectory sequence construction.
- `utils/`: fixed task generation, distance utilities, seeding, and dataset metadata.

## Reward Form

Training uses:

```text
r_t = r_ex,t + lambda_t r_in,t + r_p,t
```

where `lambda_t` is computed by `gate_weight()` in `train.py`, and `r_p,t` is computed by `pbrs_bonus()`.

Inference does not use the training reward. It loads the trained policy and selects the highest-probability legal action under the fixed evaluation protocol.
