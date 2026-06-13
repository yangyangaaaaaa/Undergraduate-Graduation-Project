# GeoExplorer acceptance demo runner

This directory is the clean server-side entrypoint for tomorrow's acceptance check.

## One command

```bash
/root/geoexplorer/run_acceptance_demo
```

Optional visual-only rerun, reusing the latest inference records:

```bash
/root/geoexplorer/run_acceptance_demo --visual-only
```

Run only one dataset image by `img_idx`:

```bash
/root/geoexplorer/run_acceptance_demo --one-image 189
```

Quickly rebuild visuals for one image from existing records:

```bash
/root/geoexplorer/run_acceptance_demo --visual-only --one-image 189
```

Run a new custom image that already exists on the server:

```bash
/root/geoexplorer/run_acceptance_demo --custom-image /root/demo.png
```

Optionally force one exact route on that custom image, using 5x5 patch ids `0..24`:

```bash
/root/geoexplorer/run_acceptance_demo --custom-image /root/demo.png --start 20 --goal 4
```

## Directory layout

- Code entrypoint: `/root/geoexplorer/GeoExplorer/acceptance_demo`
- Required visual assets and copied tables: `/root/geoexplorer/acceptance_demo_assets`
- Fresh inference records: `/root/geoexplorer/analysis/pipeline_20260517_anchor0624_visualization`
- Generated acceptance outputs: `/root/geoexplorer/analysis/acceptance_demo_oneclick_<timestamp>`

## What the script does

1. Checks required checkpoints, tables, dataset scene assets, and visualization scripts.
2. Runs the qualitative SwissViewMonuments inference test with fixed greedy policy.
3. Rebuilds acceptance GIF/PNG visuals from the fresh trajectory records.
4. Writes a README and manifest under the timestamped output directory.

For `--custom-image`, the script first creates a temporary 5x5 Sat2Cap embedding bank for the provided image, then generates a route-only visual package. It does not mix in xBD/MM-GAG/ablation evidence pages for custom images.

## Output to open for review

- `figures/acceptance_visual_pack_index.png`
- `figures/acceptance_route_gallery.png`
- `figures/acceptance_three_method_hardcase.gif`
- `figures/acceptance_mmgag_text_route_setting.gif`
- `figures/acceptance_xbd_route_settings_compare.png`
- `acceptance_demo_visuals_zh.md`

## Notes

- The script does not train models.
- The script does not store server passwords.
- The short wrapper unsets `LD_LIBRARY_PATH` and then calls `/usr/bin/bash` internally, avoiding conda/runtime bash or GLIBC pollution.
- Reward/gate/PBRS are training-stage explanations only; this acceptance run loads trained checkpoints and performs inference/visualization.
