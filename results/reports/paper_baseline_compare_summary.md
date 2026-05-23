# Paper-Aligned Baseline Compare

- protocol: greedy/argmax, `5x5`, `B=10`, `C={4,5,6,7,8}`, fixed task-bank seed `20260516`.
- task counts: Masa/MM-GAG/SwissView100 use `5` start-goal samples per image and distance; SwissViewMonuments uses the paper/repository unseen-target setting with `1` fixed-goal sample per image and distance.
- scope: evaluates only method-task combinations with validated local interfaces; blocked combinations are listed below.
- separation: this is external method comparison, not the anchor0624 mechanism ablation.

## masa_aerial

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.5880 | 1.1440 | 0.2000 | 0.3400 | 0.6400 | 0.8200 | 0.9400 |
| GOMAA-Geo | 0.5640 | 1.3600 | 0.4000 | 0.4200 | 0.6200 | 0.6600 | 0.7200 |
| Random policy | 0.0480 | 4.0600 | 0.1200 | 0.1000 | 0.0200 | 0.0000 | 0.0000 |
| DiT-AGL | 0.0360 | 3.5920 | 0.1200 | 0.0400 | 0.0200 | 0.0000 | 0.0000 |

## mmgag_aerial

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.6170 | 1.1668 | 0.2681 | 0.3702 | 0.6809 | 0.8426 | 0.9234 |
| GOMAA-Geo | 0.5336 | 1.4647 | 0.3574 | 0.3872 | 0.5532 | 0.6298 | 0.7404 |
| Random policy | 0.0681 | 3.9574 | 0.1447 | 0.0936 | 0.0553 | 0.0170 | 0.0298 |
| DiT-AGL | 0.0196 | 3.6749 | 0.0809 | 0.0170 | 0.0000 | 0.0000 | 0.0000 |

## mmgag_ground

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.6391 | 1.1055 | 0.2766 | 0.4468 | 0.6511 | 0.9106 | 0.9106 |
| GOMAA-Geo | 0.5523 | 1.3430 | 0.3489 | 0.3745 | 0.5617 | 0.7064 | 0.7702 |

## mmgag_text

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.6247 | 1.1174 | 0.2681 | 0.4298 | 0.6596 | 0.8511 | 0.9149 |
| GOMAA-Geo | 0.5472 | 1.3940 | 0.3447 | 0.4213 | 0.5745 | 0.6596 | 0.7362 |

## swissview100_aerial

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.5772 | 1.3444 | 0.2840 | 0.4100 | 0.6120 | 0.7720 | 0.8080 |
| GOMAA-Geo | 0.5044 | 1.5692 | 0.3580 | 0.3620 | 0.5340 | 0.6120 | 0.6560 |

## swissviewmonuments_aerial

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.5059 | 1.5161 | 0.3173 | 0.3833 | 0.6400 | 0.8056 | 0.8500 |
| GOMAA-Geo | 0.4314 | 1.6886 | 0.2587 | 0.3778 | 0.5633 | 0.5944 | 0.6833 |

## swissviewmonuments_ground

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.4996 | 1.5388 | 0.3093 | 0.3861 | 0.6367 | 0.7722 | 0.8667 |
| GOMAA-Geo | 0.4573 | 1.5851 | 0.2987 | 0.3917 | 0.5733 | 0.6222 | 0.7667 |

## xbd_disaster_aerial

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.5856 | 1.2839 | 0.2893 | 0.3965 | 0.6070 | 0.7895 | 0.8458 |
| GOMAA-Geo | 0.5344 | 1.4194 | 0.3265 | 0.4135 | 0.5530 | 0.6645 | 0.7147 |
| Random policy | 0.0661 | 3.9239 | 0.1520 | 0.0765 | 0.0597 | 0.0245 | 0.0177 |
| DiT-AGL | 0.0149 | 3.8104 | 0.0553 | 0.0182 | 0.0010 | 0.0000 | 0.0000 |

## xbd_pre_aerial

| Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoExplorer-anchor0624 | 0.5851 | 1.2922 | 0.2888 | 0.3955 | 0.6035 | 0.8015 | 0.8365 |
| GOMAA-Geo | 0.5427 | 1.4132 | 0.3292 | 0.4223 | 0.5585 | 0.6755 | 0.7280 |
| Random policy | 0.0661 | 3.9239 | 0.1520 | 0.0765 | 0.0597 | 0.0245 | 0.0177 |
| DiT-AGL | 0.0131 | 3.8115 | 0.0485 | 0.0168 | 0.0003 | 0.0000 | 0.0000 |

## Blocked / Not Applicable

- `PPO policy`: No validated standalone PPO baseline checkpoint/evaluator is available in this local bundle for all requested paper settings.
- `AiRLoc`: Existing AiRLoc bundle is unimodal aerial and MASA-layout specific; do not report it on ground/text without a validated adapter.

## Rigor Notes

- This is paper-aligned method comparison, separate from the anchor0624 factorial mechanism ablation.
- Ground/text comparisons are limited to methods with validated multimodal goal interfaces.
- xBD rows use the deterministic paper-test800 subset when available; xBD-disaster uses post-disaster observations with pre-disaster aerial goals.
- All included rows use greedy/argmax evaluation, 5x5 grid, B=10, and C={4,5,6,7,8}.
- Masa, MM-GAG, and SwissView100 use 5 generated start-goal pairs per image and distance; SwissViewMonuments follows the repository unseen-target protocol with 1 fixed-goal configuration per image and distance.
