# Showcase Figures

This directory contains the visual evidence package used by the GitHub README and the Chinese visualization gallery.

## Contents

- `*.png` and `*.svg`: generated charts from curated experiment result tables.
- `dataset/`: copied dataset, method, and chapter trajectory overview figures.
- `trajectories/gifs/`: 21 animated trajectory cases.
- `trajectories/comparison_png/`: 7 static method-comparison trajectory figures.
- `trajectories/static_png/`: 21 single-method static trajectory figures.
- `reward_story/`: reward-mechanism figures and one reward replay GIF.
- `showcase_manifest.json`: generated file inventory.

## Rebuild

Run from the repository root:

```bash
python code/tools/build_visual_showcase.py
```

The script reads tables from `results/tables/` and copies selected media from the local experiment analysis folders.
