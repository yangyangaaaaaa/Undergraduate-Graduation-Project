# Showcase Figures

This directory contains the visual evidence package used by the GitHub README and the Chinese visualization gallery.

## Contents

- `*.png` and `*.svg`: generated charts from curated experiment result tables.
- `experience/`: landing-page hero, method blueprint, evidence wall, storyboard, and cinematic synchronized GIFs.
- `polished/`: high-impact 16:9 scientific figure cards used by the README.
- `dataset/`: copied dataset, method, and chapter trajectory overview figures.
- `trajectories/gifs/`: 21 animated trajectory cases.
- `trajectories/triptych_gifs/`: 7 synchronized three-method GIFs for direct visual comparison.
- `trajectories/comparison_png/`: 7 static method-comparison trajectory figures.
- `trajectories/static_png/`: 21 single-method static trajectory figures.
- `reward_story/`: reward-mechanism figures and one reward replay GIF.
- `showcase_manifest.json`: generated file inventory.
- `experience/experience_manifest.json`: experience-level landing-page media inventory.
- `polished/polished_manifest.json`: polished figure-card and triptych GIF inventory.

## Rebuild

Run from the repository root:

```bash
python code/tools/build_visual_showcase.py
```

The script reads tables from `results/tables/`, copies selected media from the local experiment analysis folders, and rebuilds the experience and polished README/gallery figures.
