# PODO_niftis

**P**arcellate, **O**verlay, **D**raw, **O**rient niftis.

A small desktop viewer for NIfTI images, written in Rust. It can stack several images,
align one to another, and build label masks by growing regions from a clicked voxel.

This is an early, personal project. Expect rough edges.

## What it does

- **View** `.nii` / `.nii.gz` images in axial, coronal and sagittal panes (1x3, 2x2 or single).
  Images are reoriented to RAS+ on load. Radiological or neurological display is a toggle.
- **Overlay** several images on the first one, with colormaps, opacity and window/level per layer.
  Images with a different grid are resampled onto the first image's grid.
- **Orient** (align) a layer to the first image:
  - by hand: translate, rotate, scale and shear, then apply, or save/load the transform as JSON;
  - automatically: a built-in rigid or affine registration (correlation or mutual information).
- **Parcellate / draw** masks:
  - Ctrl+click or right-click a voxel to select a region (shown in yellow). The region updates live as
    you change settings. Enter (or the button) adds it to the mask as a new parcel with its own label.
  - Methods: connected threshold (relative to the seed or absolute), seed ± tolerance,
    confidence connected, and a ball of a given radius. 6/18/26 connectivity, optional single-slice (2D).
  - A brush for manual edits, with an eraser, adjustable radius and optional 3D.
  - Edits are undoable. Masks save as NIfTI label images on the first image's grid.

## Build and run

You need a recent Rust toolchain (edition 2024).

```
cargo build --release
./target/release/PODO_niftis your_image.nii.gz [overlay.nii.gz[:colormap=hot:opacity=0.5]] ...
```

You can also use **Open...** or drag files onto the window. Run the tests with `cargo test`. They use synthetic images generated in code.

Pre-built downloads for Linux, macOS and Windows are on the Releases page when available. They are not
code-signed, so macOS and Windows will warn about an unidentified developer (macOS: right-click the app and
choose Open; Windows: More info, then Run anyway).

## Controls

| Input | Action |
|---|---|
| Left click / drag | Move the crosshair (Navigate tool) |
| Mouse wheel | Change slice |
| Ctrl (or Cmd) + wheel, or pinch | Zoom |
| Middle drag, or Alt + drag | Pan |
| Right drag | Window / level of the selected layer |
| Ctrl (or Cmd) + click, or right click | Select a region from the clicked voxel |
| Enter / Esc | Add the selected region to the mask / cancel it |
| Brush tool: drag, Shift + drag | Paint, erase (`[` and `]` change the radius) |

## Notes and limitations

- The first image loaded defines the grid everything else is resampled onto.
- Only the first 3D volume of a 4D file is used.
- Oblique (tilted) images are shown in their own voxel grid by default. A "Straighten oblique" option
  resamples the first image onto world axes instead.
- The automatic registration was checked on synthetic and simulated cases (including a comparison with
  ANTsPy on 16 synthetic pairs). It has not been tested on a real multi-modal clinical pair, so please
  check results by eye. It is not a replacement for ANTs or FSL.
- Mask results depend on image resolution and the chosen settings; review them before using them in analysis.
- The fourth pane of the 2x2 layout is an empty placeholder.
- This is not a medical device and is not intended for clinical use.

## Layout of the code

- `src/volume.rs`: NIfTI loading, affines, reorientation.
- `src/view.rs`: planes, window/level, screen to voxel mapping.
- `src/layers.rs`: layers, colormaps, resampling, manual alignment, masks.
- `src/region.rs`: region growing and brush geometry.
- `src/register.rs`: automatic registration.
- `src/main.rs`: the app and its UI.

## Feedback and contributions

Bug reports, ideas and suggestions for improvement are very welcome. Please open an issue on GitHub:
for a bug, say what you did, what you expected and what happened (the layer note and the file's
orientation, e.g. from `nibabel`, help a lot); for a new idea, describe what you are trying to do.

## Credits

Project idea, direction and testing: antogeo.

Most of the code, the tests and this README were written by Claude Code, 
under antogeo's direction. All of it should be reviewed like any other code.

## License

MIT, see [LICENSE](LICENSE).
