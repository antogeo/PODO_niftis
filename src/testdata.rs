//! Synthetic images for the tests. Everything here is generated in code; no real data is used.

use crate::volume::{self, Volume};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A made-up, left-right and front-back asymmetric "head": scalp, skull, cortex ring, white matter,
/// two ventricle-like cavities, two bright blobs and a faint texture, on a grid centred on the origin
/// (RAS+, voxel sizes `vs` in mm).
pub fn head_phantom(dims: [usize; 3], vs: [f64; 3]) -> Volume {
    let mut affine = volume::identity();
    for a in 0..3 {
        affine[a][a] = vs[a];
        affine[a][3] = -(dims[a] as f64 - 1.0) / 2.0 * vs[a];
    }
    let mut data = vec![0f32; dims[0] * dims[1] * dims[2]];
    let dist = |p: [f64; 3], c: [f64; 3]| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = volume::apply(&affine, [i as f64, j as f64, k as f64]);
                let e = ((p[0] / 80.0).powi(2) + (p[1] / 100.0).powi(2) + (p[2] / 70.0).powi(2)).sqrt();
                let mut v = if e > 1.0 {
                    0.0
                } else if e > 0.92 {
                    60.0 // scalp
                } else if e > 0.86 {
                    20.0 // skull
                } else if e > 0.7 {
                    100.0 // cortex
                } else {
                    160.0 // white matter
                };
                if e <= 0.86 {
                    let vent = |cx: f64| ((p[0] - cx) / 8.0).powi(2) + ((p[1] - 5.0) / 25.0).powi(2) + ((p[2] - 10.0) / 10.0).powi(2) < 1.0;
                    if vent(12.0) || vent(-12.0) {
                        v = 30.0;
                    }
                    if dist(p, [25.0, 10.0, 5.0]) < 12.0 {
                        v = 130.0;
                    }
                    if dist(p, [-30.0, -25.0, 15.0]) < 8.0 {
                        v = 190.0;
                    }
                    v += 8.0 * (p[0] / 9.0).sin() * (p[1] / 11.0).sin() * (p[2] / 10.0).sin();
                }
                data[i + dims[0] * (j + dims[1] * k)] = v as f32;
            }
        }
    }
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in &data {
        min = min.min(x);
        max = max.max(x);
    }
    Volume { dims, data, affine, min, max }
}

/// Write `v` as a float NIfTI file (affine stored as the sform).
pub fn write_f32(path: &Path, v: &Volume) {
    let [nx, ny, nz] = v.dims;
    let mut h = nifti::NiftiHeader::default();
    h.sform_code = 1;
    h.qform_code = 0;
    let a = &v.affine;
    h.srow_x = [0, 1, 2, 3].map(|c| a[0][c] as f32);
    h.srow_y = [0, 1, 2, 3].map(|c| a[1][c] as f32);
    h.srow_z = [0, 1, 2, 3].map(|c| a[2][c] as f32);
    let vs = v.voxel_size();
    h.pixdim = [1.0, vs[0] as f32, vs[1] as f32, vs[2] as f32, 1.0, 1.0, 1.0, 1.0];
    let arr = ndarray::Array3::<f32>::from_shape_fn((nx, ny, nz), |(i, j, k)| v.get(i, j, k));
    nifti::writer::WriterOptions::new(path).reference_header(&h).write_nifti(&arr).unwrap();
}

/// A fresh, unique path in the temp directory (tests run in parallel).
pub fn temp_path(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("PODO_niftis_test_{}_{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// The standard test image: an 80x100x70 phantom at 2 mm, written to a temp file.
pub fn head_file(name: &str) -> String {
    let p = temp_path(name);
    write_f32(&p, &head_phantom([80, 100, 70], [2.0; 3]));
    p.to_string_lossy().into_owned()
}
