# Results

This directory stores the organized experiment results used by the thesis, defense, and repository showcase.

## Layout

- `tables/`: benchmark, ablation, appendix, long-distance, trajectory, reward-process, and statistical-analysis tables.
- `figures/showcase/`: final repository showcase figures and GIFs.
- `figures/defense_showcase/`: media extracted and organized from the final defense slides.
- `figures/acceptance_demo/`: selected acceptance-demo GIFs and overview images.
- `figures/reward_cases/`: reward-mechanism, action-attribution, route-reward, and boundary-case figures.
- `figures/presentation_assets/`: presentation candidate assets and indexes.
- `figures/chapter2_dataset/`, `figures/chapter3_method/`, `figures/chapter4_trajectories/`: thesis chapter figures and source assets.
- `reports/`: Chinese summaries, protocol notes, and analysis reports.
- `training_logs/`: lightweight logs retained for result traceability.

Large checkpoints and raw datasets are not stored here. Result tables keep run names and checkpoint paths so the source experiment can be matched to the saved model.
