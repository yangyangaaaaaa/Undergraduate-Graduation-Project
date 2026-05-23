# Official Retrain Compare

- GeoExplorer source: official git HEAD commit `2d9fc71`.
- GeoExplorer train: seed `42`, pretrain `50` epochs, RL `480000` timesteps.
- GOMAA-Geo train: seed `42`, pretrain `50` epochs, RL `480000` timesteps.
- Evaluation: greedy/argmax, `5x5`, `B=10`, `C={4,5,6,7,8}`, task-bank seed `20260516`.

## Mean SR

| Method | Mean SR |
| --- | ---: |
| GOMAA-Geo-official-retrain-seed42 | 0.5188 |
| GeoExplorer-official-pristine-seed42 | 0.3409 |

## Benchmark Table

| Benchmark | Method | SR | SG | d4 | d5 | d6 | d7 | d8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| masa_aerial | GOMAA-Geo-official-retrain-seed42 | 0.6040 | 1.1800 | 0.3600 | 0.4000 | 0.6400 | 0.7400 | 0.8800 |
| masa_aerial | GeoExplorer-official-pristine-seed42 | 0.3280 | 2.5080 | 0.2400 | 0.3400 | 0.3000 | 0.3200 | 0.4400 |
| mmgag_aerial | GOMAA-Geo-official-retrain-seed42 | 0.5583 | 1.3191 | 0.3106 | 0.4128 | 0.5447 | 0.7106 | 0.8128 |
| mmgag_aerial | GeoExplorer-official-pristine-seed42 | 0.3609 | 2.4613 | 0.3149 | 0.2979 | 0.4170 | 0.3915 | 0.3830 |
| mmgag_ground | GOMAA-Geo-official-retrain-seed42 | 0.5506 | 1.3387 | 0.3021 | 0.3915 | 0.5957 | 0.7277 | 0.7362 |
| mmgag_ground | GeoExplorer-official-pristine-seed42 | 0.3260 | 2.6306 | 0.2936 | 0.3106 | 0.3872 | 0.3234 | 0.3149 |
| mmgag_text | GOMAA-Geo-official-retrain-seed42 | 0.5455 | 1.3923 | 0.3106 | 0.4553 | 0.5532 | 0.6596 | 0.7489 |
| mmgag_text | GeoExplorer-official-pristine-seed42 | 0.3021 | 2.7157 | 0.2213 | 0.2383 | 0.3702 | 0.3106 | 0.3702 |
| swissview100_aerial | GOMAA-Geo-official-retrain-seed42 | 0.5232 | 1.5640 | 0.2760 | 0.4080 | 0.5260 | 0.6640 | 0.7420 |
| swissview100_aerial | GeoExplorer-official-pristine-seed42 | 0.3464 | 2.4116 | 0.2880 | 0.3040 | 0.3800 | 0.3580 | 0.4020 |
| swissviewmonuments_aerial | GOMAA-Geo-official-retrain-seed42 | 0.4329 | 1.7984 | 0.2613 | 0.3667 | 0.5200 | 0.6833 | 0.7167 |
| swissviewmonuments_aerial | GeoExplorer-official-pristine-seed42 | 0.3482 | 2.3576 | 0.2773 | 0.3972 | 0.3700 | 0.3611 | 0.3500 |
| swissviewmonuments_ground | GOMAA-Geo-official-retrain-seed42 | 0.4275 | 1.8220 | 0.2853 | 0.3528 | 0.5167 | 0.6333 | 0.7000 |
| swissviewmonuments_ground | GeoExplorer-official-pristine-seed42 | 0.3561 | 2.2941 | 0.3013 | 0.3833 | 0.3600 | 0.3833 | 0.4333 |
| xbd_disaster_aerial | GOMAA-Geo-official-retrain-seed42 | 0.5157 | 1.5580 | 0.3095 | 0.3917 | 0.5437 | 0.6305 | 0.7027 |
| xbd_disaster_aerial | GeoExplorer-official-pristine-seed42 | 0.3520 | 2.4440 | 0.2898 | 0.3162 | 0.3770 | 0.3982 | 0.3787 |
| xbd_pre_aerial | GOMAA-Geo-official-retrain-seed42 | 0.5118 | 1.5850 | 0.3110 | 0.3920 | 0.5330 | 0.6200 | 0.7030 |
| xbd_pre_aerial | GeoExplorer-official-pristine-seed42 | 0.3482 | 2.4678 | 0.2940 | 0.3058 | 0.3670 | 0.3927 | 0.3812 |

## Notes

- GeoExplorer algorithm source was exported from official git HEAD; wrapper config bound our canonical data paths, seed, budget, and output roots.
- This is not the same as the previous historical pristine checkpoint fixed-eval row.
- GOMAA-Geo was copied into an isolated run root, retrained with the same seed/budget/data binding, and evaluated under the same paper-aligned task bank.
- xBD uses the deterministic OpenDataLab paper-test800 reproduction subset, not a guaranteed original private split.
