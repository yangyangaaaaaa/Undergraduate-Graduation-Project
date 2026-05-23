# Ultra-long Grid Stress Evaluation

This package runs an evaluation-only stress test for long-horizon aerial search.

Purpose:
- Test whether the current method keeps its advantage when the grid is enlarged beyond the original 5x5 setting.
- Use aerial-only MASA test imagery, because larger-grid aerial patch embeddings can be regenerated deterministically from the existing raw dataset.
- Compare only three methods: GOMAA-Geo, GeoExplorer-pristine, and GeoExplorer-anchor0624.

Default protocol:
- Smoke test: grid 8x8, distances C={10,11,12,13,14}, budget B=24, two MASA test images, one repeat per distance.
- Formal main test: grid 8x8, distances C={10,11,12,13,14}, budget B=24, all MASA test images, 20 repeats per distance.
- Full stress appendix: grids 8x8 and 10x10. The 8x8 protocol uses C={10,11,12,13,14}, B=24; the 10x10 protocol uses C={14,15,16,17,18}, B=32. Both use all MASA test images and 20 repeats per distance by default.
- Distance C is the Manhattan distance between the start and target patches. Therefore the maximum valid distance is 14 for an 8x8 grid and 18 for a 10x10 grid.
- Exploratory 25x25 stress test: distances C={12,16,20,24,28,32,36,40,44,48}, budget B=60. This uses more distance buckets with fewer repeats per bucket to inspect the trend shape at a much larger grid scale.

Notes:
- This is not a paper-original protocol. It should be described as an additional ultra-long-distance stress test.
- The test is inference-only. It reuses trained checkpoints and changes the evaluation grid, distance buckets, and budget.
- No server configuration changes are required. The launcher uses the project-local NVIDIA compatibility library when present.
