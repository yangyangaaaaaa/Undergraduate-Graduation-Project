# Anchor0624 Factorial Generalization Evaluation

- design: all 16 anchor0624 factorial checkpoints, seed 321, 480k, best-val checkpoint.
- primary transfer mean: average of `mmgag_aerial`, `mmgag_ground`, `mmgag_text`, and `swissviewmonuments_aerial_ground`.
- protocol: greedy, `5x5`, `B=10`, `C={4,5,6,7,8}`, fixed generated task seed `20260516`; no additional training.

| Rank | Branch | G | P | E | V | Role | Primary Gen Mean | All Mean | Masa | MM-GAG I | MM-GAG G | MM-GAG T | SwissMon |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | g1_p1_e1_v1 | 1 | 1 | 1 | 1 | full_anchor0624 | 0.6211 | 0.6145 | 0.5880 | 0.6170 | 0.6391 | 0.6247 | 0.6036 |
| 2 | g1_p0_e1_v1 | 1 | 0 | 1 | 1 | factorial_ablation_cell | 0.5896 | 0.5893 | 0.5880 | 0.5966 | 0.6119 | 0.5949 | 0.5552 |
| 3 | g1_p0_e0_v1 | 1 | 0 | 0 | 1 | factorial_ablation_cell | 0.5865 | 0.6004 | 0.6560 | 0.6128 | 0.5889 | 0.5974 | 0.5469 |
| 4 | g1_p1_e0_v0 | 1 | 1 | 0 | 0 | factorial_ablation_cell | 0.5810 | 0.5816 | 0.5840 | 0.6034 | 0.6060 | 0.5600 | 0.5546 |
| 5 | g0_p0_e1_v1 | 0 | 0 | 1 | 1 | factorial_ablation_cell | 0.5719 | 0.5943 | 0.6840 | 0.5864 | 0.5940 | 0.5498 | 0.5575 |
| 6 | g0_p0_e0_v0 | 0 | 0 | 0 | 0 | same_data_no_added_mechanism_control | 0.5545 | 0.5532 | 0.5480 | 0.5651 | 0.5711 | 0.5387 | 0.5431 |
| 7 | g1_p0_e0_v0 | 1 | 0 | 0 | 0 | factorial_ablation_cell | 0.5265 | 0.5420 | 0.6040 | 0.5345 | 0.5328 | 0.5268 | 0.5120 |
| 8 | g0_p0_e0_v1 | 0 | 0 | 0 | 1 | factorial_ablation_cell | 0.5227 | 0.5342 | 0.5800 | 0.5404 | 0.5294 | 0.5174 | 0.5035 |
| 9 | g0_p1_e1_v0 | 0 | 1 | 1 | 0 | factorial_ablation_cell | 0.5014 | 0.5131 | 0.5600 | 0.4970 | 0.5072 | 0.5013 | 0.4999 |
| 10 | g1_p0_e1_v0 | 1 | 0 | 1 | 0 | factorial_ablation_cell | 0.4835 | 0.4884 | 0.5080 | 0.4817 | 0.4757 | 0.4681 | 0.5084 |
| 11 | g1_p1_e0_v1 | 1 | 1 | 0 | 1 | factorial_ablation_cell | 0.4834 | 0.5027 | 0.5800 | 0.4902 | 0.4894 | 0.4579 | 0.4961 |
| 12 | g0_p1_e0_v1 | 0 | 1 | 0 | 1 | factorial_ablation_cell | 0.4788 | 0.4766 | 0.4680 | 0.4800 | 0.5064 | 0.4681 | 0.4608 |
| 13 | g0_p1_e0_v0 | 0 | 1 | 0 | 0 | factorial_ablation_cell | 0.4395 | 0.4484 | 0.4840 | 0.4315 | 0.4664 | 0.4323 | 0.4279 |
| 14 | g0_p0_e1_v0 | 0 | 0 | 1 | 0 | factorial_ablation_cell | 0.4350 | 0.4488 | 0.5040 | 0.4400 | 0.4272 | 0.4383 | 0.4343 |
| 15 | g1_p1_e1_v0 | 1 | 1 | 1 | 0 | factorial_ablation_cell | 0.3848 | 0.3983 | 0.4520 | 0.4009 | 0.3728 | 0.3617 | 0.4040 |
| 16 | g0_p1_e1_v1 | 0 | 1 | 1 | 1 | factorial_ablation_cell | 0.3547 | 0.3454 | 0.3080 | 0.3617 | 0.3677 | 0.3260 | 0.3635 |

## Key Comparisons

- best transfer row: `g1_p1_e1_v1` with primary generalization mean `0.6211`.
- full anchor `g1_p1_e1_v1` primary generalization mean: `0.6211`.
- same-data control `g0_p0_e0_v0` primary generalization mean: `0.5545`.
- anchor minus control primary generalization mean: `0.0666`.

## Rigor Notes

- This is evaluation-only; no additional training or checkpoint selection is performed.
- All 16 trained factorial cells are evaluated to avoid top-row selection bias.
- Primary generalization mean excludes masa_aerial so in-domain MASA does not dominate the transfer conclusion.
- Single seed 321 limits seed-stability claims; this round supports mechanism diagnosis and transfer evidence.
