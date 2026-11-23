//! Image layers: colormaps, resampling onto the reference grid, and slice compositing.

use crate::view::{Plane, WindowLevel};
use crate::volume::{self, Affine, Volume};
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Colormap {
    Gray,
    Hot,
    Jet,
    Viridis,
    Plasma,
    CoolWarm,
    Label,
}

impl Colormap {
    pub const ALL: [Colormap; 7] = [
        Colormap::Gray,
        Colormap::Hot,
        Colormap::Jet,
        Colormap::Viridis,
        Colormap::Plasma,
        Colormap::CoolWarm,
        Colormap::Label,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Colormap::Gray => "gray",
            Colormap::Hot => "hot",
            Colormap::Jet => "jet",
            Colormap::Viridis => "viridis",
            Colormap::Plasma => "plasma",
            Colormap::CoolWarm => "coolwarm",
            Colormap::Label => "label",
        }
    }
    pub fn parse(s: &str) -> Option<Colormap> {
        Colormap::ALL.into_iter().find(|c| c.name().eq_ignore_ascii_case(s))
    }
    fn stops(self) -> &'static [(f32, [u8; 3])] {
        match self {
            Colormap::Gray | Colormap::Label => &[(0.0, [0, 0, 0]), (1.0, [255, 255, 255])],
            Colormap::Hot => &[
                (0.0, [0, 0, 0]),
                (0.33, [255, 0, 0]),
                (0.66, [255, 255, 0]),
                (1.0, [255, 255, 255]),
            ],
            Colormap::Jet => &[
                (0.0, [0, 0, 128]),
                (0.125, [0, 0, 255]),
                (0.375, [0, 255, 255]),
                (0.625, [255, 255, 0]),
                (0.875, [255, 0, 0]),
                (1.0, [128, 0, 0]),
            ],
            Colormap::Viridis => &[
                (0.0, [68, 1, 84]),
                (0.25, [59, 82, 139]),
                (0.5, [33, 145, 140]),
                (0.75, [94, 201, 98]),
                (1.0, [253, 231, 37]),
            ],
            Colormap::Plasma => &[
                (0.0, [13, 8, 135]),
                (0.25, [126, 3, 168]),
                (0.5, [204, 71, 120]),
                (0.75, [248, 149, 64]),
                (1.0, [240, 249, 33]),
            ],
            Colormap::CoolWarm => &[(0.0, [59, 76, 192]), (0.5, [221, 221, 221]), (1.0, [180, 4, 38])],
        }
    }
    /// 256-entry lookup table.
    pub fn lut(self) -> [[u8; 3]; 256] {
        let st = self.stops();
        let mut out = [[0u8; 3]; 256];
        for (n, o) in out.iter_mut().enumerate() {
            let t = n as f32 / 255.0;
            let mut k = 0;
            while k + 2 < st.len() && t > st[k + 1].0 {
                k += 1;
            }
            let (t0, c0) = st[k];
            let (t1, c1) = st[k + 1];
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            for c in 0..3 {
                o[c] = (c0[c] as f32 + (c1[c] as f32 - c0[c] as f32) * f).round() as u8;
            }
        }
        out
    }
}

/// Distinct colour per integer label (golden-ratio hue stepping).
pub fn label_color(v: i64) -> [u8; 3] {
    let h = ((v as f64) * 0.618_033_988_75).fract().abs() as f32;
    let (s, val) = (0.7f32, 0.95f32);
    let i = (h * 6.0).floor();
    let f = h * 6.0 - i;
    let (p, q, t) = (val * (1.0 - s), val * (1.0 - s * f), val * (1.0 - s * (1.0 - f)));
    let (r, g, b) = match i as i32 % 6 {
        0 => (val, t, p),
        1 => (q, val, p),
        2 => (p, val, t),
        3 => (p, q, val),
        4 => (t, p, val),
        _ => (val, p, q),
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// The reference grid every layer is resampled onto (defined by the first loaded image).
#[derive(Clone, Debug)]
pub struct Grid {
    pub dims: [usize; 3],
    pub affine: Affine,
}

impl Grid {
    pub fn of(v: &Volume) -> Grid {
        Grid { dims: v.dims, affine: v.affine }
    }
    pub fn same_as(&self, v: &Volume) -> bool {
        self.dims == v.dims
            && (0..3).all(|r| (0..4).all(|c| (self.affine[r][c] - v.affine[r][c]).abs() < 1e-4))
    }
}

/// Manual alignment of a layer to the reference: a world-space affine built from
/// scale -> shear -> rotation (Rz*Ry*Rx) about a pivot, then translation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformParams {
    /// mm
    pub translate: [f64; 3],
    /// degrees about x, y, z
    pub rotate: [f64; 3],
    pub scale: [f64; 3],
    /// shear terms: [xy, xz, yz] (x += xy*y + xz*z, y += yz*z)
    pub shear: [f64; 3],
}

impl Default for TransformParams {
    fn default() -> Self {
        TransformParams { translate: [0.0; 3], rotate: [0.0; 3], scale: [1.0; 3], shear: [0.0; 3] }
    }
}

impl TransformParams {
    pub fn is_identity(&self) -> bool {
        // tolerant: composing a transform with its inverse leaves ~1e-15 residue
        let near = |v: &[f64; 3], t: f64| v.iter().all(|x| (x - t).abs() < 1e-9);
        near(&self.translate, 0.0) && near(&self.rotate, 0.0) && near(&self.scale, 1.0) && near(&self.shear, 0.0)
    }

    /// World-to-world matrix moving image content: p' = pivot + t + R*Sh*S*(p - pivot).
    pub fn matrix(&self, pivot: [f64; 3]) -> Affine {
        let [rx, ry, rz] = self.rotate.map(|d| d.to_radians());
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        let rx_m = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
        let ry_m = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
        let rz_m = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
        let m3 = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| {
            let mut r = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        r[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
            r
        };
        let rot = m3(rz_m, m3(ry_m, rx_m));
        let sh = [[1.0, self.shear[0], self.shear[1]], [0.0, 1.0, self.shear[2]], [0.0, 0.0, 1.0]];
        let sc = [[self.scale[0], 0.0, 0.0], [0.0, self.scale[1], 0.0], [0.0, 0.0, self.scale[2]]];
        let lin = m3(rot, m3(sh, sc));
        let mut a = volume::identity();
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = lin[i][j];
            }
            // p' = lin*(p - pivot) + pivot + t
            a[i][3] = pivot[i] + self.translate[i] - (lin[i][0] * pivot[0] + lin[i][1] * pivot[1] + lin[i][2] * pivot[2]);
        }
        a
    }
}

impl TransformParams {
    /// Recover the editable parameters from a world-to-world affine (the inverse of `matrix`).
    /// Exact for any proper affine (positive determinant); mirror images are rejected.
    pub fn from_matrix(a: &Affine, pivot: [f64; 3]) -> Result<Self, String> {
        if a.iter().flatten().any(|v| !v.is_finite()) {
            return Err("transform contains non-finite numbers".into());
        }
        if a[3] != [0.0, 0.0, 0.0, 1.0] {
            return Err("not an affine transform (bottom row must be 0 0 0 1)".into());
        }
        // Gram-Schmidt on the columns: L = Q * U, U upper-triangular with positive diagonal.
        let col = |j: usize| [a[0][j], a[1][j], a[2][j]];
        let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
        let norm = |u: [f64; 3]| dot(u, u).sqrt();
        let sub = |u: [f64; 3], v: [f64; 3], k: f64| [u[0] - k * v[0], u[1] - k * v[1], u[2] - k * v[2]];
        let (c0, c1, c2) = (col(0), col(1), col(2));
        let u00 = norm(c0);
        if u00 < 1e-9 {
            return Err("transform is singular".into());
        }
        let q0 = [c0[0] / u00, c0[1] / u00, c0[2] / u00];
        let u01 = dot(q0, c1);
        let r1 = sub(c1, q0, u01);
        let u11 = norm(r1);
        if u11 < 1e-9 {
            return Err("transform is singular".into());
        }
        let q1 = [r1[0] / u11, r1[1] / u11, r1[2] / u11];
        let u02 = dot(q0, c2);
        let u12 = dot(q1, c2);
        let r2 = sub(sub(c2, q0, u02), q1, u12);
        let u22 = norm(r2);
        if u22 < 1e-9 {
            return Err("transform is singular".into());
        }
        let q2 = [r2[0] / u22, r2[1] / u22, r2[2] / u22];
        // Q has columns q0,q1,q2; reject reflections (det < 0).
        let det = q0[0] * (q1[1] * q2[2] - q1[2] * q2[1]) - q1[0] * (q0[1] * q2[2] - q0[2] * q2[1])
            + q2[0] * (q0[1] * q1[2] - q0[2] * q1[1]);
        if det < 0.0 {
            return Err("transform contains a mirror flip, which is not supported".into());
        }
        let r = |i: usize, j: usize| [q0, q1, q2][j][i];
        // R = Rz*Ry*Rx
        let ry = (-r(2, 0)).clamp(-1.0, 1.0).asin();
        let (rx, rz) = if ry.cos().abs() > 1e-6 {
            (r(2, 1).atan2(r(2, 2)), r(1, 0).atan2(r(0, 0)))
        } else {
            ((-r(1, 2)).atan2(r(1, 1)), 0.0) // gimbal lock: fold into x
        };
        let lin = |i: usize| a[i][0] * pivot[0] + a[i][1] * pivot[1] + a[i][2] * pivot[2];
        Ok(TransformParams {
            translate: [a[0][3] - pivot[0] + lin(0), a[1][3] - pivot[1] + lin(1), a[2][3] - pivot[2] + lin(2)],
            rotate: [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()],
            scale: [u00, u11, u22],
            shear: [u01 / u11, u02 / u22, u12 / u22],
        })
    }
}

pub fn affine_is_identity(a: &Affine) -> bool {
    let id = volume::identity();
    (0..4).all(|i| (0..4).all(|j| (a[i][j] - id[i][j]).abs() < 1e-9))
}

/// Serialise a transform for saving. `matrix` maps moving-image world positions to the reference.
pub fn transform_to_json(matrix: &Affine, params: &TransformParams, pivot: [f64; 3], moving: &str, reference: &str, total: bool) -> String {
    let v = serde_json::json!({
        "format": "nii_viewer_transform",
        "version": 1,
        "description": "World-space (RAS+, mm) affine that moves the moving image onto the reference: p_reference = matrix * p_moving.",
        "kind": if total { "total alignment of the image" } else { "adjustment relative to the applied alignment" },
        "moving_image": moving,
        "reference_image": reference,
        "matrix": matrix,
        "parameters": {
            "pivot_ras_mm": pivot,
            "translate_mm": params.translate,
            "rotate_deg_xyz": params.rotate,
            "scale_xyz": params.scale,
            "shear_xy_xz_yz": params.shear,
        },
    });
    serde_json::to_string_pretty(&v).unwrap()
}

pub fn transform_from_json(text: &str) -> Result<Affine, String> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("not a valid transform file: {e}"))?;
    if v["format"] != "nii_viewer_transform" {
        return Err("not a nii_viewer transform file (missing format marker)".into());
    }
    let rows = v["matrix"].as_array().filter(|r| r.len() == 4).ok_or("matrix must have 4 rows")?;
    let mut m = [[0.0; 4]; 4];
    for (i, row) in rows.iter().enumerate() {
        let row = row.as_array().filter(|r| r.len() == 4).ok_or("matrix rows must have 4 numbers")?;
        for (j, x) in row.iter().enumerate() {
            m[i][j] = x.as_f64().ok_or("matrix entries must be numbers")?;
        }
    }
    Ok(m)
}

impl Grid {
    /// World position of the grid's centre (used as the pivot for rotation/scale).
    pub fn center(&self) -> [f64; 3] {
        volume::apply(
            &self.affine,
            [(self.dims[0] as f64 - 1.0) / 2.0, (self.dims[1] as f64 - 1.0) / 2.0, (self.dims[2] as f64 - 1.0) / 2.0],
        )
    }
}

/// Resample `src` (moved by the world transform `xform`) onto `grid`.
/// Trilinear for intensity, nearest-neighbour for labels. Voxels outside the source become 0.
pub fn resample(src: &Volume, grid: &Grid, nearest: bool, xform: &Affine) -> Volume {
    let m = volume::mul(&volume::invert(&volume::mul(xform, &src.affine)), &grid.affine);
    let [nx, ny, nz] = grid.dims;
    let [sx, sy, sz] = src.dims;
    let mut data = vec![0f32; nx * ny * nz];
    let (fx, fy, fz) = (sx as f64, sy as f64, sz as f64);
    data.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slab)| {
        for j in 0..ny {
            for i in 0..nx {
                let p = volume::apply(&m, [i as f64, j as f64, k as f64]);
                if p[0] < -0.5 || p[1] < -0.5 || p[2] < -0.5 || p[0] > fx - 0.5 || p[1] > fy - 0.5 || p[2] > fz - 0.5 {
                    continue;
                }
                let v = if nearest {
                    src.get(
                        (p[0].round() as usize).min(sx - 1),
                        (p[1].round() as usize).min(sy - 1),
                        (p[2].round() as usize).min(sz - 1),
                    )
                } else {
                    let q = [p[0].clamp(0.0, fx - 1.0), p[1].clamp(0.0, fy - 1.0), p[2].clamp(0.0, fz - 1.0)];
                    let i0 = [q[0].floor() as usize, q[1].floor() as usize, q[2].floor() as usize];
                    let i1 = [(i0[0] + 1).min(sx - 1), (i0[1] + 1).min(sy - 1), (i0[2] + 1).min(sz - 1)];
                    let t = [(q[0] - i0[0] as f64) as f32, (q[1] - i0[1] as f64) as f32, (q[2] - i0[2] as f64) as f32];
                    let g = |a: usize, b: usize, c: usize| src.get(a, b, c);
                    let c00 = g(i0[0], i0[1], i0[2]) * (1.0 - t[0]) + g(i1[0], i0[1], i0[2]) * t[0];
                    let c10 = g(i0[0], i1[1], i0[2]) * (1.0 - t[0]) + g(i1[0], i1[1], i0[2]) * t[0];
                    let c01 = g(i0[0], i0[1], i1[2]) * (1.0 - t[0]) + g(i1[0], i0[1], i1[2]) * t[0];
                    let c11 = g(i0[0], i1[1], i1[2]) * (1.0 - t[0]) + g(i1[0], i1[1], i1[2]) * t[0];
                    let c0 = c00 * (1.0 - t[1]) + c10 * t[1];
                    let c1 = c01 * (1.0 - t[1]) + c11 * t[1];
                    c0 * (1.0 - t[2]) + c1 * t[2]
                };
                slab[i + nx * j] = v;
            }
        }
    });
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &data {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if !min.is_finite() {
        min = 0.0;
        max = 1.0;
    }
    Volume { dims: grid.dims, data, affine: grid.affine, min, max }
}

pub struct Layer {
    pub name: String,
    /// Shown under the name, e.g. "resampled from 91x109x91 @ 2.00 mm".
    pub note: Option<String>,
    /// Data on the reference grid.
    pub vol: Volume,
    /// Original data. Present for every layer except the reference, so it can be re-resampled
    /// when the colormap's interpolation mode or the alignment changes.
    pub orig: Option<Volume>,
    pub colormap: Colormap,
    pub opacity: f32,
    pub wl: WindowLevel,
    pub visible: bool,
    /// Make values <= min transparent (for overlays).
    pub clear_below: bool,
    pub nearest: bool,
    /// Manual alignment to the reference (ignored for the reference layer), relative to `base`.
    pub xf: TransformParams,
    /// Alignment already "applied" (committed); `xf` is a further adjustment on top of it.
    pub base: Affine,
    /// A user-drawn mask (label volume on the reference grid) that region growing paints into.
    pub is_mask: bool,
    /// Edit history of a mask: for each edit, the (voxel index, previous value) it overwrote.
    pub undo: Vec<Vec<(usize, f32)>>,
    /// True while consecutive `paint_stroke` calls are being merged into the last undo entry.
    pub stroke_open: bool,
    /// How the file's orientation was interpreted at load (shown in the layer note).
    pub load_note: Option<String>,
}

impl Layer {
    /// The reference layer: defines the grid, is never resampled or transformed.
    pub fn reference(name: String, src: Volume, colormap: Colormap) -> Layer {
        let wl = WindowLevel::auto(&src);
        Layer {
            name,
            note: None,
            vol: src,
            orig: None,
            colormap,
            opacity: 1.0,
            wl,
            visible: true,
            clear_below: false,
            nearest: false,
            xf: TransformParams::default(),
            base: volume::identity(),
            is_mask: false,
            undo: Vec::new(),
            stroke_open: false,
            load_note: None,
        }
    }

    /// A layer resampled onto `grid`.
    pub fn new(name: String, src: Volume, grid: &Grid, colormap: Colormap) -> Layer {
        let nearest = colormap == Colormap::Label;
        let xf = TransformParams::default();
        let vol = resample(&src, grid, nearest, &xf.matrix(grid.center()));
        let wl = if colormap == Colormap::Label {
            WindowLevel { lo: 0.0, hi: vol.max.max(1.0) }
        } else {
            WindowLevel::auto(&vol)
        };
        let mut l = Layer {
            name,
            note: None,
            vol,
            orig: Some(src),
            colormap,
            opacity: 1.0,
            wl,
            visible: true,
            clear_below: false,
            nearest,
            xf,
            base: volume::identity(),
            is_mask: false,
            undo: Vec::new(),
            stroke_open: false,
            load_note: None,
        };
        l.update_note(grid);
        l
    }

    /// An empty mask on `grid`, shown as a translucent label overlay.
    pub fn mask(name: String, grid: &Grid) -> Layer {
        let n = grid.dims[0] * grid.dims[1] * grid.dims[2];
        let vol = Volume { dims: grid.dims, data: vec![0.0; n], affine: grid.affine, min: 0.0, max: 1.0 };
        Layer {
            name,
            note: Some("mask".into()),
            vol,
            orig: None,
            colormap: Colormap::Label,
            opacity: 0.6,
            wl: WindowLevel { lo: 0.0, hi: 1.0 },
            visible: true,
            clear_below: false,
            nearest: true,
            xf: TransformParams::default(),
            base: volume::identity(),
            is_mask: true,
            undo: Vec::new(),
            stroke_open: false,
            load_note: None,
        }
    }

    /// Set the given voxels to `value` (0 erases). Only voxels that actually change are
    /// recorded for undo. Returns how many changed.
    pub fn paint(&mut self, idx: &[usize], value: f32) -> usize {
        self.stroke_open = false;
        let mut changed = Vec::new();
        for &i in idx {
            let old = self.vol.data[i];
            if old != value {
                changed.push((i, old));
                self.vol.data[i] = value;
            }
        }
        let n = changed.len();
        if n > 0 {
            self.undo.push(changed);
            if self.undo.len() > 50 {
                self.undo.remove(0);
            }
            self.vol.max = self.vol.data.iter().copied().fold(1.0, f32::max);
        }
        n
    }

    /// Paint as part of a brush stroke: all calls after one with `new_stroke` merge into a single
    /// undo step. Returns how many voxels changed.
    pub fn paint_stroke(&mut self, idx: &[usize], value: f32, new_stroke: bool) -> usize {
        if new_stroke {
            self.stroke_open = false;
        }
        let mut changed = Vec::new();
        for &i in idx {
            let old = self.vol.data[i];
            if old != value {
                changed.push((i, old));
                self.vol.data[i] = value;
            }
        }
        let n = changed.len();
        if n == 0 {
            return 0;
        }
        match self.undo.last_mut() {
            Some(last) if self.stroke_open => last.extend(changed),
            _ => {
                self.undo.push(changed);
                if self.undo.len() > 50 {
                    self.undo.remove(0);
                }
                self.stroke_open = true;
            }
        }
        self.vol.max = self.vol.max.max(value);
        n
    }

    /// Revert the last edit; false if there is nothing to undo.
    pub fn undo_paint(&mut self) -> bool {
        self.stroke_open = false;
        match self.undo.pop() {
            Some(edit) => {
                for (i, old) in edit.into_iter().rev() {
                    self.vol.data[i] = old;
                }
                self.vol.max = self.vol.data.iter().copied().fold(1.0, f32::max);
                true
            }
            None => false,
        }
    }

    /// The next unused parcel label (1 for an empty mask).
    pub fn next_label(&self) -> f32 {
        self.vol.data.iter().copied().fold(0.0, f32::max).round() + 1.0
    }

    pub fn mask_voxels(&self) -> usize {
        self.vol.data.iter().filter(|&&v| v != 0.0).count()
    }

    /// Write the mask as a NIfTI label image (RAS+ orientation, same grid as the reference).
    pub fn save_mask(&self, path: &std::path::Path) -> Result<String, String> {
        let [nx, ny, nz] = self.vol.dims;
        let mut hdr = nifti::NiftiHeader::default();
        let a = &self.vol.affine;
        hdr.sform_code = 1;
        hdr.qform_code = 0;
        hdr.srow_x = [0, 1, 2, 3].map(|c| a[0][c] as f32);
        hdr.srow_y = [0, 1, 2, 3].map(|c| a[1][c] as f32);
        hdr.srow_z = [0, 1, 2, 3].map(|c| a[2][c] as f32);
        let vs = self.vol.voxel_size();
        hdr.pixdim = [1.0, vs[0] as f32, vs[1] as f32, vs[2] as f32, 1.0, 1.0, 1.0, 1.0];
        let arr = ndarray::Array3::<u16>::from_shape_fn((nx, ny, nz), |(i, j, k)| {
            self.vol.get(i, j, k).round().clamp(0.0, 65535.0) as u16
        });
        nifti::writer::WriterOptions::new(path)
            .reference_header(&hdr)
            .write_nifti(&arr)
            .map_err(|e| format!("could not write {}: {e}", path.display()))?;
        Ok(format!("saved mask ({} voxels) to {}", self.mask_voxels(), path.display()))
    }

    pub fn update_note(&mut self, grid: &Grid) {
        let mut parts = Vec::new();
        if let Some(n) = &self.load_note {
            parts.push(n.clone());
        }
        if let Some(o) = &self.orig {
            if !grid.same_as(o) {
                let vs = o.voxel_size();
                parts.push(format!(
                    "resampled from {}x{}x{} @ {:.2}/{:.2}/{:.2} mm",
                    o.dims[0], o.dims[1], o.dims[2], vs[0], vs[1], vs[2]
                ));
            }
        }
        if !self.xf.is_identity() || !affine_is_identity(&self.base) {
            parts.push("aligned manually".to_string());
        }
        self.note = if parts.is_empty() { None } else { Some(parts.join(", ")) };
    }

    /// Redo the resampling from the original data (after a colormap or alignment change).
    pub fn refresh(&mut self, grid: &Grid) {
        if let Some(o) = &self.orig {
            self.vol = resample(o, grid, self.nearest, &self.total_matrix(grid));
        }
        self.update_note(grid);
    }

    /// Change colormap, redoing the resampling if the interpolation mode changes.
    pub fn set_colormap(&mut self, cm: Colormap, grid: &Grid) {
        self.colormap = cm;
        let nearest = cm == Colormap::Label;
        if nearest != self.nearest {
            self.nearest = nearest;
            self.refresh(grid);
        }
    }

    /// The current adjustment (relative to the applied alignment) as a world-space matrix.
    pub fn transform_matrix(&self, grid: &Grid) -> Affine {
        self.xf.matrix(grid.center())
    }

    /// The full movement of the image: current adjustment on top of the applied alignment.
    pub fn total_matrix(&self, grid: &Grid) -> Affine {
        volume::mul(&self.transform_matrix(grid), &self.base)
    }

    /// Lock in the current alignment as the new starting point; the adjustment fields reset.
    /// The displayed image does not change.
    pub fn apply_alignment(&mut self, grid: &Grid) {
        self.base = self.total_matrix(grid);
        self.xf = TransformParams::default();
        self.update_note(grid);
    }

    /// Forget every applied alignment and adjustment (back to the original position).
    pub fn discard_alignment(&mut self, grid: &Grid) {
        self.base = volume::identity();
        self.xf = TransformParams::default();
        self.refresh(grid);
    }

    /// Apply a world-space matrix (e.g. loaded from a file, or its inverse) as the alignment.
    pub fn set_matrix(&mut self, m: &Affine, grid: &Grid) -> Result<(), String> {
        let p = TransformParams::from_matrix(m, grid.center())?;
        self.set_transform(p, grid);
        Ok(())
    }

    pub fn set_transform(&mut self, xf: TransformParams, grid: &Grid) {
        if xf != self.xf {
            self.xf = xf;
            self.refresh(grid);
        }
    }
}

/// Write the layer's alignment to `path`: the adjustment relative to the applied alignment, or
/// with `total` the full movement from the original position. Returns a status message.
pub fn save_transform_file(path: &std::path::Path, l: &Layer, grid: &Grid, reference: &str, total: bool) -> Result<String, String> {
    let (m, params) = if total {
        let m = l.total_matrix(grid);
        (m, TransformParams::from_matrix(&m, grid.center())?)
    } else {
        (l.transform_matrix(grid), l.xf)
    };
    let text = transform_to_json(&m, &params, grid.center(), &l.name, reference, total);
    std::fs::write(path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    Ok(format!(
        "saved {} transform to {}",
        if total { "total" } else if affine_is_identity(&l.base) { "" } else { "relative" },
        path.display()
    )
    .replace("saved  ", "saved "))
}

/// Load a transform file and apply it to the layer, or apply its inverse (`reverse`).
pub fn load_transform_file(path: &std::path::Path, l: &mut Layer, grid: &Grid, reverse: bool) -> Result<String, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let m = transform_from_json(&text)?;
    // Validate first (rejects singular / mirrored / non-affine matrices) so we never invert garbage.
    TransformParams::from_matrix(&m, grid.center())?;
    // Forward: the file's transform becomes the alignment (loading twice is idempotent).
    // Reverse: undo the file's transform from where the image is now, so
    // "apply T, then reverse T" returns the image to where it started.
    let m = if reverse { volume::mul(&volume::invert(&m), &l.transform_matrix(grid)) } else { m };
    l.set_matrix(&m, grid)?;
    Ok(format!(
        "applied {}transform from {}",
        if reverse { "reverse of " } else { "" },
        path.display()
    ))
}

/// Composite visible layers (bottom to top) into RGBA for one slice.
pub fn compose(layers: &[Layer], plane: Plane, idx: usize, radiological: bool, dims: [usize; 3]) -> Vec<u8> {
    let (ha, va, sa) = plane.axes();
    let (nh, nv) = (dims[ha], dims[va]);
    let flip = plane.flip_h(radiological);
    let luts: Vec<[[u8; 3]; 256]> = layers.iter().map(|l| l.colormap.lut()).collect();
    let mut out = vec![255u8; nh * nv * 4];
    for row in 0..nv {
        let vi = nv - 1 - row;
        for col in 0..nh {
            let hi = if flip { nh - 1 - col } else { col };
            let mut ijk = [0usize; 3];
            ijk[ha] = hi;
            ijk[va] = vi;
            ijk[sa] = idx;
            let mut rgb = [0f32; 3];
            for (l, lut) in layers.iter().zip(&luts) {
                if !l.visible {
                    continue;
                }
                let x = l.vol.get(ijk[0], ijk[1], ijk[2]);
                let c = if l.colormap == Colormap::Label {
                    let n = x.round() as i64;
                    if n == 0 {
                        continue;
                    }
                    label_color(n)
                } else {
                    if l.clear_below && x <= l.wl.lo {
                        continue;
                    }
                    lut[l.wl.to_u8(x) as usize]
                };
                for k in 0..3 {
                    rgb[k] = rgb[k] * (1.0 - l.opacity) + c[k] as f32 * l.opacity;
                }
            }
            let o = (row * nh + col) * 4;
            out[o] = rgb[0] as u8;
            out[o + 1] = rgb[1] as u8;
            out[o + 2] = rgb[2] as u8;
        }
    }
    out
}

/// Blend `color` over the voxels flagged in `sel` (indexed like a volume) on one slice of `rgba`.
pub fn overlay_selection(rgba: &mut [u8], plane: Plane, idx: usize, radiological: bool, dims: [usize; 3], sel: &[bool], color: [u8; 3], alpha: f32) {
    let (ha, va, sa) = plane.axes();
    let (nh, nv) = (dims[ha], dims[va]);
    let flip = plane.flip_h(radiological);
    for row in 0..nv {
        let vi = nv - 1 - row;
        for col in 0..nh {
            let hi = if flip { nh - 1 - col } else { col };
            let mut ijk = [0usize; 3];
            ijk[ha] = hi;
            ijk[va] = vi;
            ijk[sa] = idx;
            if sel[ijk[0] + dims[0] * (ijk[1] + dims[1] * ijk[2])] {
                let o = (row * nh + col) * 4;
                for k in 0..3 {
                    rgba[o + k] = (rgba[o + k] as f32 * (1.0 - alpha) + color[k] as f32 * alpha) as u8;
                }
            }
        }
    }
}

/// A parsed command-line image argument: `path[:colormap=hot][:opacity=0.5][:min=..][:max=..]`.
#[derive(Debug, Default, PartialEq)]
pub struct ImageArg {
    pub path: String,
    pub colormap: Option<Colormap>,
    pub opacity: Option<f32>,
    pub min: Option<f32>,
    pub max: Option<f32>,
}

pub fn parse_arg(s: &str) -> Result<ImageArg, String> {
    let mut parts = s.split(':');
    let mut a = ImageArg { path: parts.next().unwrap_or("").to_string(), ..Default::default() };
    for p in parts {
        let (k, v) = p.split_once('=').ok_or_else(|| format!("bad option '{p}' (expected key=value)"))?;
        let num = |v: &str| v.parse::<f32>().map_err(|_| format!("bad number '{v}' for {k}"));
        match k {
            "colormap" | "cmap" => {
                a.colormap = Some(Colormap::parse(v).ok_or_else(|| format!("unknown colormap '{v}'"))?)
            }
            "opacity" => a.opacity = Some(num(v)?.clamp(0.0, 1.0)),
            "min" => a.min = Some(num(v)?),
            "max" => a.max = Some(num(v)?),
            _ => return Err(format!("unknown option '{k}'")),
        }
    }
    if a.path.is_empty() {
        return Err("empty path".into());
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::identity;

    fn vol(dims: [usize; 3], affine: Affine, f: impl Fn(usize, usize, usize) -> f32) -> Volume {
        let mut data = vec![0.0; dims[0] * dims[1] * dims[2]];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    data[i + dims[0] * (j + dims[1] * k)] = f(i, j, k);
                }
            }
        }
        Volume { dims, data, affine, min: 0.0, max: 100.0 }
    }

    #[test]
    fn resample_identity_is_exact() {
        let v = vol([5, 6, 7], identity(), |i, j, k| (i + 10 * j + 100 * k) as f32);
        let r = resample(&v, &Grid::of(&v), false, &identity());
        assert_eq!(r.data, v.data);
    }

    #[test]
    fn resample_coarse_onto_fine_preserves_world_position() {
        // Source: 2mm voxels; value = world x (mm) at voxel centre.
        let mut a2 = identity();
        for i in 0..3 {
            a2[i][i] = 2.0;
        }
        let src = vol([10, 10, 10], a2, |i, _, _| (2 * i) as f32);
        // Grid: 1mm voxels covering the same space.
        let grid = Grid { dims: [19, 19, 19], affine: identity() };
        let r = resample(&src, &grid, false, &identity());
        for i in 0..19 {
            // linear ramp: value should equal world x
            assert!((r.get(i, 5, 5) - i as f32).abs() < 1e-4, "i={i} got {}", r.get(i, 5, 5));
        }
    }

    #[test]
    fn resample_handles_flipped_source() {
        // Source has x flipped (affine x = -i + 4); value = i.
        let mut a = identity();
        a[0][0] = -1.0;
        a[0][3] = 4.0;
        let src = vol([5, 1, 1], a, |i, _, _| i as f32);
        let grid = Grid { dims: [5, 1, 1], affine: identity() };
        let r = resample(&src, &grid, true, &identity());
        // world x = 0 corresponds to source voxel i=4
        assert_eq!(r.data, vec![4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn outside_fov_is_zero() {
        let src = vol([2, 2, 2], identity(), |_, _, _| 5.0);
        let grid = Grid { dims: [6, 2, 2], affine: identity() };
        let r = resample(&src, &grid, true, &identity());
        assert_eq!(r.get(5, 0, 0), 0.0);
        assert_eq!(r.get(1, 0, 0), 5.0);
    }

    #[test]
    fn colormap_endpoints() {
        assert_eq!(Colormap::Gray.lut()[0], [0, 0, 0]);
        assert_eq!(Colormap::Gray.lut()[255], [255, 255, 255]);
        assert_eq!(Colormap::Hot.lut()[255], [255, 255, 255]);
        assert_eq!(Colormap::Viridis.lut()[0], [68, 1, 84]);
        assert_eq!(Colormap::Viridis.lut()[255], [253, 231, 37]);
    }

    #[test]
    fn parse_args() {
        let a = parse_arg("b.nii.gz:colormap=hot:opacity=0.5:min=1:max=9").unwrap();
        assert_eq!(a.path, "b.nii.gz");
        assert_eq!(a.colormap, Some(Colormap::Hot));
        assert_eq!(a.opacity, Some(0.5));
        assert_eq!((a.min, a.max), (Some(1.0), Some(9.0)));
        assert!(parse_arg("a.nii:colormap=nope").is_err());
        assert!(parse_arg("a.nii:foo=1").is_err());
        assert_eq!(parse_arg("a.nii").unwrap().colormap, None);
    }

    #[test]
    fn compose_marker_orientation_and_overlay() {
        let dims = [4, 5, 6];
        let mut base = vol(dims, identity(), |_, _, _| 0.0);
        let i = base.idx(3, 4, 5);
        base.data[i] = 1.0;
        let grid = Grid::of(&base);
        let mut l = Layer::new("m".into(), base, &grid, Colormap::Gray);
        l.wl = WindowLevel { lo: 0.0, hi: 1.0 };
        let img = compose(std::slice::from_ref(&l), Plane::Axial, 5, true, dims);
        let p = img.chunks(4).position(|px| px[0] == 255).unwrap();
        // Radiological axial: patient Right on screen-left, Anterior on top.
        assert_eq!((p % 4, p / 4), (0, 0));
        let find = |img: &[u8], w: usize| img.chunks(4).position(|px| px[0] == 255).map(|p| (p % w, p / w));
        // Neurological axial: Right on screen-right.
        let img = compose(std::slice::from_ref(&l), Plane::Axial, 5, false, dims);
        assert_eq!(find(&img, 4), Some((3, 0)));
        // Coronal (y index 4): Right on left, Superior on top.
        let img = compose(std::slice::from_ref(&l), Plane::Coronal, 4, true, dims);
        assert_eq!(find(&img, 4), Some((0, 0)));
        // Sagittal (x index 3): Anterior on screen-left, Superior on top.
        let img = compose(std::slice::from_ref(&l), Plane::Sagittal, 3, true, dims);
        assert_eq!(find(&img, 5), Some((0, 0)));
        // Hidden layer contributes nothing.
        l.visible = false;
        let img = compose(std::slice::from_ref(&l), Plane::Axial, 5, true, dims);
        assert!(img.chunks(4).all(|px| px[0] == 0));
    }

    fn marker(dims: [usize; 3], at: [usize; 3]) -> Volume {
        vol(dims, identity(), |i, j, k| if [i, j, k] == at { 1.0 } else { 0.0 })
    }
    fn where_is(v: &Volume) -> Vec<[usize; 3]> {
        let mut out = vec![];
        for k in 0..v.dims[2] {
            for j in 0..v.dims[1] {
                for i in 0..v.dims[0] {
                    if v.get(i, j, k) > 0.5 {
                        out.push([i, j, k]);
                    }
                }
            }
        }
        out
    }
    fn moved(at: [usize; 3], xf: TransformParams) -> Vec<[usize; 3]> {
        let src = marker([11, 11, 11], at);
        let grid = Grid::of(&src);
        let r = resample(&src, &grid, true, &xf.matrix(grid.center()));
        where_is(&r)
    }

    #[test]
    fn identity_params_give_identity_matrix() {
        let m = TransformParams::default().matrix([3.0, -4.0, 5.0]);
        for i in 0..4 {
            for j in 0..4 {
                assert!((m[i][j] - identity()[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn translation_moves_content_in_plus_direction() {
        let xf = TransformParams { translate: [2.0, -1.0, 3.0], ..Default::default() };
        assert_eq!(moved([5, 5, 5], xf), vec![[7, 4, 8]]);
    }

    #[test]
    fn rotation_about_pivot_90_degrees_z() {
        // pivot = (5,5,5); offset (3,0,0) rotates +90deg about z to (0,3,0)
        let xf = TransformParams { rotate: [0.0, 0.0, 90.0], ..Default::default() };
        assert_eq!(moved([8, 5, 5], xf), vec![[5, 8, 5]]);
        // the pivot itself never moves
        assert_eq!(moved([5, 5, 5], xf), vec![[5, 5, 5]]);
    }

    #[test]
    fn scale_about_pivot() {
        let xf = TransformParams { scale: [2.0, 2.0, 2.0], ..Default::default() };
        // A 1-voxel marker at offset +2 becomes a 2-voxel-wide block centred on offset +4
        // (voxel 9); its edge sits exactly on a voxel boundary so nearest picks 8 and 9.
        let hit = moved([7, 5, 5], xf);
        assert!(hit.contains(&[9, 5, 5]));
        assert!(hit.iter().all(|p| (8..=9).contains(&p[0]) && (4..=5).contains(&p[1]) && (4..=5).contains(&p[2])));
        assert!(hit.len() <= 8 && hit.len() >= 2);
    }

    #[test]
    fn shear_xy() {
        // x += 1.0 * y : offset y=2 shifts x by 2
        let xf = TransformParams { shear: [1.0, 0.0, 0.0], ..Default::default() };
        assert_eq!(moved([5, 7, 5], xf), vec![[7, 7, 5]]);
    }

    #[test]
    fn layer_refresh_applies_alignment_and_notes_it() {
        let src = marker([11, 11, 11], [5, 5, 5]);
        let grid = Grid::of(&src);
        let mut l = Layer::new("m".into(), src, &grid, Colormap::Gray);
        assert!(l.note.is_none());
        l.set_transform(TransformParams { translate: [1.0, 0.0, 0.0], ..Default::default() }, &grid);
        assert_eq!(l.note.as_deref(), Some("aligned manually"));
        assert!(l.vol.get(6, 5, 5) > 0.4);
        l.set_transform(TransformParams::default(), &grid);
        assert!(l.note.is_none());
        assert!(l.vol.get(5, 5, 5) > 0.99);
    }

    fn close(a: &Affine, b: &Affine, tol: f64) -> bool {
        (0..4).all(|i| (0..4).all(|j| (a[i][j] - b[i][j]).abs() < tol))
    }

    fn samples() -> Vec<TransformParams> {
        vec![
            TransformParams::default(),
            TransformParams { translate: [3.0, -2.5, 7.0], ..Default::default() },
            TransformParams { rotate: [10.0, -20.0, 35.0], ..Default::default() },
            TransformParams { rotate: [-80.0, 45.0, 170.0], scale: [1.1, 0.9, 1.3], ..Default::default() },
            TransformParams {
                translate: [-5.0, 1.0, 2.0],
                rotate: [5.0, 15.0, -25.0],
                scale: [0.8, 1.2, 1.05],
                shear: [0.1, -0.05, 0.2],
            },
            // gimbal lock: pitch of exactly 90 degrees
            TransformParams { rotate: [30.0, 90.0, 0.0], ..Default::default() },
        ]
    }

    #[test]
    fn params_matrix_roundtrip() {
        let pivot = [12.0, -7.0, 30.0];
        for p in samples() {
            let m = p.matrix(pivot);
            let back = TransformParams::from_matrix(&m, pivot).unwrap();
            // compare matrices (angles are not unique at gimbal lock)
            assert!(close(&m, &back.matrix(pivot), 1e-9), "{p:?} -> {back:?}");
        }
    }

    #[test]
    fn inverse_undoes_transform() {
        let pivot = [1.0, 2.0, 3.0];
        for p in samples() {
            let m = p.matrix(pivot);
            let inv = TransformParams::from_matrix(&volume::invert(&m), pivot).unwrap();
            assert!(close(&volume::mul(&inv.matrix(pivot), &m), &identity(), 1e-9), "{p:?}");
        }
    }

    #[test]
    fn reverse_brings_the_image_back() {
        let at = [4, 6, 5];
        let src = marker([11, 11, 11], at);
        let grid = Grid::of(&src);
        let mut l = Layer::new("m".into(), src, &grid, Colormap::Label);
        let fwd = TransformParams { translate: [2.0, -1.0, 3.0], rotate: [0.0, 0.0, 90.0], ..Default::default() };
        l.set_transform(fwd, &grid);
        assert_ne!(where_is(&l.vol), vec![at]);
        // "reverse transform": estimate the inverse of the saved matrix and apply it
        let saved = l.transform_matrix(&grid);
        l.set_matrix(&volume::invert(&saved), &grid).unwrap();
        // Inverse matrix applied to the ORIGINAL image moves it the opposite way, so applying
        // fwd then judging against the inverse of the same matrix must compose to identity:
        let net = volume::mul(&l.transform_matrix(&grid), &saved);
        assert!(close(&net, &identity(), 1e-9));
    }

    #[test]
    fn json_roundtrip_and_validation() {
        let pivot = [1.0, 2.0, 3.0];
        let p = samples()[4];
        let m = p.matrix(pivot);
        let text = transform_to_json(&m, &p, pivot, "b.nii.gz", "a.nii.gz", false);
        assert!(text.contains("b.nii.gz") && text.contains("rotate_deg_xyz"));
        assert!(close(&transform_from_json(&text).unwrap(), &m, 1e-12));
        assert!(transform_from_json("{}").is_err());
        assert!(transform_from_json("not json").is_err());
        assert!(transform_from_json(r#"{"format":"nii_viewer_transform","matrix":[[1,0,0]]}"#).is_err());
    }

    #[test]
    fn rejects_mirror_singular_and_non_affine() {
        let mut mirror = identity();
        mirror[0][0] = -1.0;
        assert!(TransformParams::from_matrix(&mirror, [0.0; 3]).is_err());
        let mut singular = identity();
        singular[2][2] = 0.0;
        assert!(TransformParams::from_matrix(&singular, [0.0; 3]).is_err());
        let mut proj = identity();
        proj[3][0] = 0.5;
        assert!(TransformParams::from_matrix(&proj, [0.0; 3]).is_err());
        let mut nan = identity();
        nan[0][3] = f64::NAN;
        assert!(TransformParams::from_matrix(&nan, [0.0; 3]).is_err());
    }

    #[test]
    fn save_load_and_reverse_via_files() {
        let dir = std::path::Path::new("target/test_tmp");
        std::fs::create_dir_all(dir).unwrap();
        let file = dir.join("t.json");
        let src = marker([11, 11, 11], [5, 5, 5]);
        let grid = Grid::of(&src);
        let mut a = Layer::new("moving.nii.gz".into(), src.clone(), &grid, Colormap::Label);
        a.set_transform(TransformParams { translate: [2.0, 0.0, 0.0], rotate: [0.0, 0.0, 90.0], ..Default::default() }, &grid);
        let msg = save_transform_file(&file, &a, &grid, "ref.nii.gz", false).unwrap();
        assert!(msg.contains("t.json"));

        // load onto a fresh layer: identical result
        let mut b = Layer::new("other.nii.gz".into(), src.clone(), &grid, Colormap::Label);
        load_transform_file(&file, &mut b, &grid, false).unwrap();
        assert_eq!(where_is(&b.vol), where_is(&a.vol));
        assert!(close(&b.transform_matrix(&grid), &a.transform_matrix(&grid), 1e-9));

        // reverse: matrix is the inverse of the saved one
        let mut c = Layer::new("third.nii.gz".into(), src, &grid, Colormap::Label);
        let msg = load_transform_file(&file, &mut c, &grid, true).unwrap();
        assert!(msg.contains("reverse"));
        let net = volume::mul(&c.transform_matrix(&grid), &a.transform_matrix(&grid));
        assert!(close(&net, &identity(), 1e-9));

        // bad files give errors and leave the layer untouched
        std::fs::write(dir.join("bad.json"), "{}").unwrap();
        let before = c.xf;
        assert!(load_transform_file(&dir.join("bad.json"), &mut c, &grid, false).is_err());
        assert!(load_transform_file(&dir.join("missing.json"), &mut c, &grid, false).is_err());
        assert_eq!(c.xf, before);
    }

    #[test]
    fn apply_then_reverse_returns_to_the_initial_image() {
        let dir = std::path::Path::new("target/test_tmp");
        std::fs::create_dir_all(dir).unwrap();
        let file = dir.join("roundtrip.json");
        let src = marker([11, 11, 11], [4, 6, 5]);
        let grid = Grid::of(&src);
        for t in samples().into_iter().skip(1).take(3) {
            let mut l = Layer::new("m.nii.gz".into(), src.clone(), &grid, Colormap::Label);
            l.set_transform(t, &grid);
            save_transform_file(&file, &l, &grid, "ref.nii.gz", false).unwrap();
            // image currently carries T; load the reverse of the saved T
            load_transform_file(&file, &mut l, &grid, true).unwrap();
            assert!(l.xf.is_identity(), "{t:?} -> {:?}", l.xf);
            assert!(l.note.is_none());
            assert_eq!(where_is(&l.vol), vec![[4, 6, 5]], "image did not return to the initial position for {t:?}");
        }
        // loading the forward transform twice is idempotent
        let mut l = Layer::new("m.nii.gz".into(), src, &grid, Colormap::Label);
        l.set_transform(samples()[1], &grid);
        save_transform_file(&file, &l, &grid, "ref.nii.gz", false).unwrap();
        let once = l.transform_matrix(&grid);
        load_transform_file(&file, &mut l, &grid, false).unwrap();
        load_transform_file(&file, &mut l, &grid, false).unwrap();
        assert!(close(&l.transform_matrix(&grid), &once, 1e-9));
    }

    #[test]
    fn apply_then_random_then_reverse_returns_to_the_applied_alignment() {
        let dir = std::path::Path::new("target/test_tmp");
        std::fs::create_dir_all(dir).unwrap();
        let file = dir.join("random.json");
        let src = marker([11, 11, 11], [4, 6, 5]);
        let grid = Grid::of(&src);
        let good = TransformParams { translate: [2.0, -1.0, 1.0], rotate: [0.0, 0.0, 90.0], ..Default::default() };
        let random = TransformParams { translate: [-3.0, 2.0, 0.0], rotate: [0.0, 90.0, 0.0], ..Default::default() };

        // reference result: just the good alignment
        let mut want = Layer::new("m".into(), src.clone(), &grid, Colormap::Label);
        want.set_transform(good, &grid);

        let mut l = Layer::new("m".into(), src, &grid, Colormap::Label);
        l.set_transform(good, &grid);
        let before = l.vol.data.clone();
        l.apply_alignment(&grid);
        assert_eq!(l.vol.data, before, "Apply must not change the displayed image");
        assert!(l.xf.is_identity(), "fields reset after Apply");
        assert_eq!(l.note.as_deref(), Some("aligned manually"));

        l.set_transform(random, &grid);
        assert_ne!(l.vol.data, before);
        let msg = save_transform_file(&file, &l, &grid, "ref", false).unwrap();
        assert!(msg.contains("relative"), "{msg}");

        load_transform_file(&file, &mut l, &grid, true).unwrap();
        assert!(l.xf.is_identity());
        assert_eq!(where_is(&l.vol), where_is(&want.vol), "must return to the good alignment, not the original");
        assert!(close(&l.total_matrix(&grid), &want.total_matrix(&grid), 1e-9));

        // saving the total includes the applied part
        l.set_transform(random, &grid);
        save_transform_file(&file, &l, &grid, "ref", true).unwrap();
        let tot = transform_from_json(&std::fs::read_to_string(&file).unwrap()).unwrap();
        assert!(close(&tot, &l.total_matrix(&grid), 1e-9));

        // discard goes back to the original position
        l.discard_alignment(&grid);
        assert_eq!(where_is(&l.vol), vec![[4, 6, 5]]);
        assert!(l.note.is_none());
    }
}
