# PODO_niftis

**P**arcellate, **O**verlay, **D**raw, **O**rient niftis.

Rust desktop NIfTI viewer (eframe/egui 0.37, ndarray, nifti 0.18, rayon, rfd).
Multi-layer viewing, manual and automatic alignment, and mask building by region growing.

The crate and binary are `PODO_niftis`. The transform JSON format marker stays `nii_viewer_transform`
so previously saved transforms still load.

## Build and test
- Cargo is in the project-local toolchain, not on PATH:
  `export PATH=$PWD/.toolchain/cargo/bin:$PATH CARGO_HOME=$PWD/.toolchain/cargo RUSTUP_HOME=$PWD/.toolchain/rustup`
- `cargo test --offline` (45+ tests; UI tests use egui_kittest). `[profile.test]` is opt-level 2,
  because the registration tests are too slow unoptimized. The first run after a profile change rebuilds every dependency (minutes).
- `cargo build --release --offline`, then `./target/release/PODO_niftis <image.nii.gz>`.
- The release binary goes stale if you only build debug. Rebuild it after changes.
- No git commits yet. Everything is untracked on `master` (main branch is `main`).

## Layout
- `src/volume.rs`: NIfTI loading, affines, reorientation to RAS+. Voxel index is x-fastest.
  Orientation is tested by storing a phantom in all 48 axis orders/flips and loading it back
  (`all_48_orientations_load_identically`). `load_with_info` reports the stored axcodes and obliquity; the
  layer note shows them. Oblique images stay in their own voxel grid by default; top-bar 'Straighten oblique' resamples an oblique first image to world axes (`deoblique`).
- `src/view.rs`: planes, window/level, screen <-> voxel placement (zoom, pan, radiological flip).
- `src/layers.rs`: colormaps, `Grid`, resampling, `Layer`, manual alignment
  (`TransformParams`, JSON save/load), `compose`, `overlay_selection`, mask layers
  (`Layer::mask`, `paint`, `undo_paint`, `next_label`, `save_mask`).
- `src/region.rs`: seeded region growing (pure functions, unit-tested).
- `src/register.rs`: automatic rigid/affine registration (multi-resolution search, correlation or
  mutual information). Runs on a worker thread and reports through an mpsc channel.
- `src/main.rs`: App state, panes, sidebar, input handling, UI tests.
- Tests use only synthetic images generated in code (`src/testdata.rs`). No image data is stored in the repo;
  never commit real or third-party data, file names or paths of it.

## Concepts
- The first loaded image defines the reference grid. Every other layer is resampled onto it.
- All volumes are stored in RAS+; the affine maps voxel to world mm.
- Alignment: a per-layer world-space affine. `base` is the applied alignment and `xf` is the adjustment on top of it.
- Auto-registration (layer sidebar > Align to first image > Auto-register): the first image is fixed,
  the selected layer's original volume is moving. The result is written into `xf` (relative to `base`),
  so it shows up in the manual fields and can be applied, saved or fine-tuned. Needs no external tools.

## Mask / region growing
- Ctrl+click or right-click selects a seed. The grown region appears in yellow
  (`App.pending`) and is recomputed live when any setting changes.
- Enter, the "Add region" button or the right-click menu commits it to the mask
  as a new parcel (next unused label). "Remove from mask" subtracts it. Esc cancels.
- The target is the selected mask layer, else the topmost mask, else a new one is created.
- Methods:
  - Connected threshold: relative to the seed (default, -x / +y in % or intensity) or an absolute range.
  - Seed +/- tolerance.
  - Confidence connected (mean +/- k*SD).
  - Ball (radius in mm).
- Options: 6/18/26 connectivity, and restricting growth to the clicked slice (2D).
- Brush tool (sidebar > Masks > Brush): left-drag paints into the target mask with the brush label, Shift
  or the Eraser erases, radius in mm (`[` / `]` resize), optional 3D ball. One stroke = one undo step
  (`Layer::paint_stroke`). Ctrl+click still selects a region while the brush is active.
- Mask edits are undoable (the last 50). Masks save as u16 NIfTI on the reference grid.

## Conventions
- Keep pure logic in `region.rs` / `layers.rs` with unit tests. UI behaviour goes in
  the kittest modules at the bottom of `main.rs`.
- Mouse: left = cursor, middle-drag = pan, right-drag = window/level, wheel = slice, Ctrl+wheel = zoom.

## Status / next
- Done: layers, manual alignment, mask region growing with live preview, built-in automatic registration.
- Registration (`src/register.rs`, benchmarks in `src/register/bench.rs`): Mattes-style MI (B-spline Parzen,
  32 bins) or correlation, analytic gradients + L-BFGS, 8/4/2/1 mm pyramid with random sampling at fine levels,
  multi-start (centres of mass, FOV centres, 24 axis orientations, principal axes). Benchmarked against ANTsPy
  on 16 synthetic pairs: matches or beats it, and handles large rotations/flips that ANTs failed. Heavy
  benchmarks are `#[ignore]`d (`cargo test --offline -- --ignored`). Not yet tried on a real T2 scan.
  No ANTs/FSL is installed here; an optional `antsRegistration` backend is not needed for now.
- The 2x2 layout's fourth pane is a "3D view (milestone 4)" placeholder.
- Nothing is committed yet.
