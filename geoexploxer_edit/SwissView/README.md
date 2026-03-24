---
license: apache-2.0
---


# Dataset Card for SwissView Dataset
## Project Page

https://limirs.github.io/GeoExplorer/

GeoExplorer: Active Geo-localization with Curiosity-Driven Exploration


## Dataset Summary

This dataset consists of two subsets:

- **SwissViewMonuments**: which includes 15 images of atypical or distinctive scenes, such as unusual buildings and landscapes, with corresponding ground level images.
- **SwissView100**, which comprises 100 images randomly selected from across the Swiss territory, thereby providing diverse natural and urban environment.

## Dataset Structure

### Data Fields

For **SwissViewMonuments**:
- `id`: unique identifier
- `ground_view`: file path to the ground view image
- `aerial_view`: file path to the aerial view image

For **SwissView100**:
- `id`: unique identifier
- `aerial_view`: file path to the image
- `LV95_coordinates`: coordinates under LV95 coordinate system