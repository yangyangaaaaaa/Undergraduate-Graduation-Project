# GeoExplorer acceptance training runner

This is the clean server-side entrypoint for a training smoke check or a long training launch.

## One command

```bash
/root/geoexplorer/run_acceptance_train
```

This default command runs a short smoke training (`20` timesteps) to prove the training pipeline can load data, load the pretraining checkpoint, run PPO training, write logs, and save checkpoints.

## Long training

```bash
/root/geoexplorer/run_acceptance_train --full
```

The full mode launches a background `480000`-timestep training run and prints the PID/log path.

## Monitor

```bash
/root/geoexplorer/run_acceptance_train --status
/root/geoexplorer/run_acceptance_train --tail
```

## Directory layout

- Short wrapper: `/root/geoexplorer/run_acceptance_train`
- Code entrypoint: `/root/geoexplorer/GeoExplorer/acceptance_train`
- Generated outputs: `/root/geoexplorer/analysis/acceptance_train_<mode>_<timestamp>`
- Latest symlink: `/root/geoexplorer/analysis/acceptance_train_latest`

## Default training setup

- Dataset: `swissview`
- Data: `/root/geoexplorer/GeoExplorer/data/swissview/swissview100_sat_patches.npy`
- Reward: external + gated intrinsic (`GEOEXPLORER_REWARD=in`)
- Gate: linear gate with floor `0.405`
- PBRS: `0.10`
- Validation: max `5` validation images for smoke speed

## Notes

- The default command is intentionally short; it is for acceptance demonstration, not final paper-scale training.
- Use `--full` only when a long training run is intended.
- The wrapper unsets `LD_LIBRARY_PATH` and calls `/usr/bin/bash` internally to avoid conda/runtime GLIBC pollution.
- The script does not store server passwords.
