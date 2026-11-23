//! Volume loading and RAS+ reorientation.
//!
//! Internally every volume is stored so that voxel index axis 0 increases toward
//! patient Right->(+x), axis 1 toward Anterior (+y), axis 2 toward Superior (+z)
//! ("RAS+" world, as in NIfTI). The affine maps voxel (i,j,k) -> world mm.

use nifti::{IntoNdArray, NiftiHeader, NiftiObject, NiftiVolume, ReaderOptions};
use std::path::Path;

pub type Affine = [[f64; 4]; 4];

#[derive(Clone, Debug)]
pub struct Volume {
    pub dims: [usize; 3],
    /// x-fastest: index = i + nx*(j + ny*k)
    pub data: Vec<f32>,
    pub affine: Affine,
    pub min: f32,
    pub max: f32,
}

impl Volume {
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }
    pub fn get(&self, i: usize, j: usize, k: usize) -> f32 {
        self.data[self.idx(i, j, k)]
    }
    pub fn voxel_size(&self) -> [f64; 3] {
        let a = &self.affine;
        let mut s = [0.0; 3];
        for c in 0..3 {
            s[c] = (a[0][c].powi(2) + a[1][c].powi(2) + a[2][c].powi(2)).sqrt();
        }
        s
    }
}

pub fn identity() -> Affine {
    let mut a = [[0.0; 4]; 4];
    for i in 0..4 {
        a[i][i] = 1.0;
    }
    a
}

pub fn mul(a: &Affine, b: &Affine) -> Affine {
    let mut r = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

pub fn apply(a: &Affine, p: [f64; 3]) -> [f64; 3] {
    let mut o = [0.0; 3];
    for r in 0..3 {
        o[r] = a[r][0] * p[0] + a[r][1] * p[1] + a[r][2] * p[2] + a[r][3];
    }
    o
}

/// General inverse of a 4x4 affine (last row 0 0 0 1).
#[allow(dead_code)]
pub fn invert(a: &Affine) -> Affine {
    let m = [
        [a[0][0], a[0][1], a[0][2]],
        [a[1][0], a[1][1], a[1][2]],
        [a[2][0], a[2][1], a[2][2]],
    ];
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let d = 1.0 / det;
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d,
        ],
    ];
    let t = [a[0][3], a[1][3], a[2][3]];
    let mut r = identity();
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = inv[i][j];
        }
        r[i][3] = -(inv[i][0] * t[0] + inv[i][1] * t[1] + inv[i][2] * t[2]);
    }
    r
}

/// Build the voxel->world affine from a NIfTI header (sform > qform > pixdim).
pub fn header_affine(h: &NiftiHeader) -> Affine {
    let mut a = identity();
    if h.sform_code > 0 {
        a[0] = [
            h.srow_x[0] as f64,
            h.srow_x[1] as f64,
            h.srow_x[2] as f64,
            h.srow_x[3] as f64,
        ];
        a[1] = [
            h.srow_y[0] as f64,
            h.srow_y[1] as f64,
            h.srow_y[2] as f64,
            h.srow_y[3] as f64,
        ];
        a[2] = [
            h.srow_z[0] as f64,
            h.srow_z[1] as f64,
            h.srow_z[2] as f64,
            h.srow_z[3] as f64,
        ];
    } else if h.qform_code > 0 {
        let (b, c, d) = (h.quatern_b as f64, h.quatern_c as f64, h.quatern_d as f64);
        // b,c,d are stored as f32; renormalise so near-180-degree rotations stay orthonormal.
        let n = b * b + c * c + d * d;
        let (b, c, d, a2) = if n > 1.0 {
            let k = 1.0 / n.sqrt();
            (b * k, c * k, d * k, 0.0)
        } else {
            (b, c, d, (1.0 - n).sqrt())
        };
        let qfac = if h.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        let (dx, dy, dz) = (
            h.pixdim[1] as f64,
            h.pixdim[2] as f64,
            h.pixdim[3] as f64 * qfac,
        );
        let r = [
            [
                a2 * a2 + b * b - c * c - d * d,
                2.0 * (b * c - a2 * d),
                2.0 * (b * d + a2 * c),
            ],
            [
                2.0 * (b * c + a2 * d),
                a2 * a2 + c * c - b * b - d * d,
                2.0 * (c * d - a2 * b),
            ],
            [
                2.0 * (b * d - a2 * c),
                2.0 * (c * d + a2 * b),
                a2 * a2 + d * d - c * c - b * b,
            ],
        ];
        for i in 0..3 {
            a[i][0] = r[i][0] * dx;
            a[i][1] = r[i][1] * dy;
            a[i][2] = r[i][2] * dz;
        }
        a[0][3] = h.quatern_x as f64;
        a[1][3] = h.quatern_y as f64;
        a[2][3] = h.quatern_z as f64;
    } else {
        // No orientation info: assume RAS with pixdim scaling.
        for i in 0..3 {
            a[i][i] = (h.pixdim[i + 1] as f64).abs().max(1e-6);
        }
    }
    a
}

/// Reorder voxel data so axes point toward R, A, S (closest-axis reorientation,
/// no interpolation). Returns the new dims, data and affine.
pub fn reorient_ras(dims: [usize; 3], data: &[f32], affine: &Affine) -> ([usize; 3], Vec<f32>, Affine) {
    // For each input voxel axis, find which world axis it mostly points along.
    let mut world_axis = [0usize; 3]; // input axis -> world axis
    let mut sign = [1.0f64; 3];
    for c in 0..3 {
        let col = [affine[0][c], affine[1][c], affine[2][c]];
        let mut best = 0;
        for w in 1..3 {
            if col[w].abs() > col[best].abs() {
                best = w;
            }
        }
        world_axis[c] = best;
        sign[c] = if col[best] >= 0.0 { 1.0 } else { -1.0 };
    }
    // Guard against degenerate (non-permutation) affines: fall back to no-op.
    let mut seen = [false; 3];
    for c in 0..3 {
        seen[world_axis[c]] = true;
    }
    if !seen.iter().all(|&s| s) {
        return (dims, data.to_vec(), *affine);
    }
    // out axis w <- input axis c with world_axis[c]==w
    let mut in_axis = [0usize; 3];
    for c in 0..3 {
        in_axis[world_axis[c]] = c;
    }
    let od = [dims[in_axis[0]], dims[in_axis[1]], dims[in_axis[2]]];
    let mut out = vec![0f32; data.len()];
    for ok in 0..od[2] {
        for oj in 0..od[1] {
            for oi in 0..od[0] {
                let o = [oi, oj, ok];
                let mut inp = [0usize; 3];
                for w in 0..3 {
                    let c = in_axis[w];
                    inp[c] = if sign[c] > 0.0 { o[w] } else { dims[c] - 1 - o[w] };
                }
                let src = inp[0] + dims[0] * (inp[1] + dims[1] * inp[2]);
                out[oi + od[0] * (oj + od[1] * ok)] = data[src];
            }
        }
    }
    // New affine: world = A_in * M * (out voxel), where M maps out voxel -> in voxel.
    let mut m = [[0.0; 4]; 4];
    m[3][3] = 1.0;
    for w in 0..3 {
        let c = in_axis[w];
        if sign[c] > 0.0 {
            m[c][w] = 1.0;
        } else {
            m[c][w] = -1.0;
            m[c][3] = (dims[c] - 1) as f64;
        }
    }
    (od, out, mul(affine, &m))
}

/// How the file stored its axes and how oblique it is.
#[derive(Clone, Debug)]
pub struct LoadInfo {
    /// Direction each stored voxel axis points toward, e.g. "LAS" (i toward Left, j Anterior, k Superior).
    pub axcodes: String,
    /// Largest angle (degrees) between a voxel axis and the nearest world axis.
    pub obliquity_deg: f64,
}

/// Orientation letters of the voxel axes of `affine` (closest world axis for each column).
pub fn axcodes(affine: &Affine) -> String {
    (0..3)
        .map(|c| {
            let col = [affine[0][c], affine[1][c], affine[2][c]];
            let w = (0..3).max_by(|&a, &b| col[a].abs().partial_cmp(&col[b].abs()).unwrap()).unwrap();
            (if col[w] >= 0.0 { "RAS" } else { "LPI" }).chars().nth(w).unwrap()
        })
        .collect()
}

/// Largest angle (degrees) between any voxel axis and its closest world axis.
pub fn obliquity_deg(affine: &Affine) -> f64 {
    (0..3)
        .map(|c| {
            let col = [affine[0][c], affine[1][c], affine[2][c]];
            let norm = (col[0] * col[0] + col[1] * col[1] + col[2] * col[2]).sqrt().max(1e-12);
            let m = col.iter().map(|x| x.abs()).fold(0.0, f64::max);
            (m / norm).clamp(0.0, 1.0).acos().to_degrees()
        })
        .fold(0.0, f64::max)
}

/// Resample an oblique (tilted) volume onto a world-aligned RAS grid with the same voxel sizes,
/// covering the whole rotated volume. Returns None if the volume is already axis-aligned.
/// Trilinear, or nearest-neighbour for integer-valued (label) data. World positions are preserved.
pub fn deoblique(v: &Volume) -> Option<Volume> {
    if obliquity_deg(&v.affine) < 0.5 {
        return None;
    }
    let vs = v.voxel_size();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for corner in 0..8 {
        let p = [
            if corner & 1 == 0 { -0.5 } else { v.dims[0] as f64 - 0.5 },
            if corner & 2 == 0 { -0.5 } else { v.dims[1] as f64 - 0.5 },
            if corner & 4 == 0 { -0.5 } else { v.dims[2] as f64 - 0.5 },
        ];
        let w = apply(&v.affine, p);
        for a in 0..3 {
            lo[a] = lo[a].min(w[a]);
            hi[a] = hi[a].max(w[a]);
        }
    }
    // After reorient_ras, voxel axis c is closest to world axis c.
    let dims = [0, 1, 2].map(|a| (((hi[a] - lo[a]) / vs[a]).ceil() as usize).max(1));
    let mut affine = identity();
    for a in 0..3 {
        affine[a][a] = vs[a];
        affine[a][3] = lo[a] + vs[a] / 2.0;
    }
    let nearest = v.data.iter().all(|x| !x.is_finite() || x.fract() == 0.0);
    let grid = crate::layers::Grid { dims, affine };
    Some(crate::layers::resample(v, &grid, nearest, &identity()))
}

#[allow(dead_code)]
pub fn load(path: &Path) -> Result<Volume, String> {
    load_with_info(path).map(|(v, _)| v)
}

pub fn load_with_info(path: &Path) -> Result<(Volume, LoadInfo), String> {
    let obj = ReaderOptions::new()
        .read_file(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let header = obj.header().clone();
    let vol = obj.into_volume();
    let d = vol.dim().to_vec();
    if d.len() < 3 {
        return Err("image has fewer than 3 dimensions".into());
    }
    let dims = [d[0] as usize, d[1] as usize, d[2] as usize];
    // 4D: only the first 3D volume is used for now.
    let nd = dims[0] * dims[1] * dims[2];
    let arr = vol
        .into_ndarray::<f32>()
        .map_err(|e| format!("could not decode voxel data: {e}"))?;
    // ndarray from nifti is indexed [x, y, z, (t..)] — iterate explicitly to be layout-safe.
    let mut data = vec![0f32; nd];
    let nd_dims = arr.ndim();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let mut ix = vec![0usize; nd_dims];
                ix[0] = i;
                ix[1] = j;
                ix[2] = k;
                data[i + dims[0] * (j + dims[1] * k)] = arr[ndarray::IxDyn(&ix)];
            }
        }
    }
    let affine = header_affine(&header);
    let info = LoadInfo { axcodes: axcodes(&affine), obliquity_deg: obliquity_deg(&affine) };
    let (dims, data, affine) = reorient_ras(dims, &data, &affine);
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
    Ok((Volume { dims, data, affine, min, max }, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A voxel's world coordinate must be unchanged by reorientation.
    fn check(affine: Affine) {
        let dims = [3usize, 4, 5];
        let data: Vec<f32> = (0..60).map(|v| v as f32).collect();
        let (od, out, oa) = reorient_ras(dims, &data, &affine);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let w = apply(&affine, [i as f64, j as f64, k as f64]);
                    // find that world pos in the output grid
                    let inv = invert(&oa);
                    let v = apply(&inv, w);
                    let (a, b, c) = (v[0].round() as usize, v[1].round() as usize, v[2].round() as usize);
                    assert!(a < od[0] && b < od[1] && c < od[2], "out of range");
                    assert_eq!(out[a + od[0] * (b + od[1] * c)], data[i + dims[0] * (j + dims[1] * k)]);
                }
            }
        }
        // output axes must point toward +R,+A,+S
        for w in 0..3 {
            assert!(oa[w][w] > 0.0);
        }
    }

    #[test]
    fn lps_affine() {
        let mut a = identity();
        a[0][0] = -1.0;
        a[1][1] = -1.0;
        a[0][3] = 10.0;
        a[1][3] = 20.0;
        check(a);
    }

    #[test]
    fn swapped_axes() {
        // voxel i -> +A, j -> -S, k -> +R
        let a = [
            [0.0, 0.0, 2.0, -5.0],
            [1.0, 0.0, 0.0, -6.0],
            [0.0, -1.5, 0.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        check(a);
    }

    #[test]
    fn invert_roundtrip() {
        let a = [
            [0.0, 0.0, 2.0, -5.0],
            [1.0, 0.0, 0.0, -6.0],
            [0.0, -1.5, 0.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let p = [3.0, 4.0, 5.0];
        let q = apply(&invert(&a), apply(&a, p));
        for c in 0..3 {
            assert!((p[c] - q[c]).abs() < 1e-9);
        }
    }
}

#[cfg(test)]
mod oblique_tests {
    use super::*;

    fn rot_z(deg: f64) -> Affine {
        let (s, c) = deg.to_radians().sin_cos();
        let mut a = identity();
        a[0][0] = c;
        a[0][1] = -s;
        a[1][0] = s;
        a[1][1] = c;
        a
    }

    #[test]
    fn axcodes_and_obliquity() {
        assert_eq!(axcodes(&identity()), "RAS");
        let mut lps = identity();
        lps[0][0] = -1.0;
        lps[1][1] = -1.0;
        assert_eq!(axcodes(&lps), "LPS");
        assert!(obliquity_deg(&identity()) < 1e-9);
        assert!((obliquity_deg(&rot_z(20.0)) - 20.0).abs() < 1e-6);
    }

    #[test]
    fn deoblique_keeps_world_positions_and_straightens_the_grid() {
        // Anisotropic oblique volume with one bright blob.
        let dims = [30, 40, 20];
        let mut aff = volume_mul(&rot_z(25.0), &{
            let mut s = identity();
            s[0][0] = 1.0;
            s[1][1] = 1.5;
            s[2][2] = 3.0;
            s
        });
        aff[0][3] = 10.0;
        aff[1][3] = -20.0;
        aff[2][3] = 5.0;
        let mut data = vec![0f32; 30 * 40 * 20];
        let blob = [20usize, 25, 10];
        for dk in 0..3 {
            for dj in 0..3 {
                for di in 0..3 {
                    data[(blob[0] + di) + 30 * ((blob[1] + dj) + 40 * (blob[2] + dk))] = 100.0;
                }
            }
        }
        let v = Volume { dims, data, affine: aff, min: 0.0, max: 100.0 };
        let want = apply(&aff, [blob[0] as f64 + 1.0, blob[1] as f64 + 1.0, blob[2] as f64 + 1.0]);
        let d = deoblique(&v).expect("oblique volume");
        assert!(obliquity_deg(&d.affine) < 1e-9);
        let vs = d.voxel_size();
        assert!((vs[0] - 1.0).abs() < 1e-9 && (vs[1] - 1.5).abs() < 1e-9 && (vs[2] - 3.0).abs() < 1e-9);
        // centre of mass of the blob in world space is unchanged (within a voxel)
        let (mut sw, mut s) = (0.0, [0.0; 3]);
        for (n, &x) in d.data.iter().enumerate() {
            if x > 1.0 {
                let p = apply(&d.affine, [(n % d.dims[0]) as f64, ((n / d.dims[0]) % d.dims[1]) as f64, (n / (d.dims[0] * d.dims[1])) as f64]);
                sw += x as f64;
                for a in 0..3 {
                    s[a] += x as f64 * p[a];
                }
            }
        }
        let got = s.map(|x| x / sw);
        assert!((0..3).all(|a| (got[a] - want[a]).abs() < 1.0), "got {got:?} want {want:?}");
        assert!(deoblique(&d).is_none(), "already aligned");
    }

    fn volume_mul(a: &Affine, b: &Affine) -> Affine {
        mul(a, b)
    }
}

#[cfg(test)]
mod orient_tests {
    use super::*;
    use crate::testdata::{head_phantom, temp_path, write_f32};

    /// Store a phantom with every axis order and flip combination (48 orientations, including
    /// mirror-handed ones); each file must load back as exactly the original RAS volume.
    #[test]
    fn all_48_orientations_load_identically() {
        let v = head_phantom([26, 30, 22], [3.0, 4.0, 5.0]);
        let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
        let mut bad = Vec::new();
        for perm in perms {
            for flips in 0..8u8 {
                let flip = [flips & 1 != 0, flips & 2 != 0, flips & 4 != 0];
                let sd = [v.dims[perm[0]], v.dims[perm[1]], v.dims[perm[2]]];
                let mut data = vec![0f32; v.data.len()];
                for k in 0..sd[2] {
                    for j in 0..sd[1] {
                        for i in 0..sd[0] {
                            let s = [i, j, k];
                            let mut vi = [0usize; 3];
                            for a in 0..3 {
                                vi[perm[a]] = if flip[a] { sd[a] - 1 - s[a] } else { s[a] };
                            }
                            data[i + sd[0] * (j + sd[1] * k)] = v.get(vi[0], vi[1], vi[2]);
                        }
                    }
                }
                // M maps a stored voxel index to the RAS volume's voxel index.
                let mut m = [[0.0; 4]; 4];
                m[3][3] = 1.0;
                for a in 0..3 {
                    m[perm[a]][a] = if flip[a] { -1.0 } else { 1.0 };
                    m[perm[a]][3] = if flip[a] { (sd[a] - 1) as f64 } else { 0.0 };
                }
                let stored = Volume { dims: sd, data, affine: mul(&v.affine, &m), min: v.min, max: v.max };
                let path = temp_path("orient.nii.gz");
                write_f32(&path, &stored);
                let back = load(&path).unwrap();
                let same_aff = (0..4).all(|r| (0..4).all(|c| (back.affine[r][c] - v.affine[r][c]).abs() < 1e-3));
                if back.dims != v.dims || back.data != v.data || !same_aff {
                    bad.push(format!("perm {perm:?} flips {flip:?}: dims {:?} affine_ok {same_aff}", back.dims));
                }
            }
        }
        assert!(bad.is_empty(), "{}", bad.join("\n"));
    }
}
