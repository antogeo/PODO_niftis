//! Registration benchmarks: known transforms, different FOV/resolution/contrast/noise.
//! Heavy ones are #[ignore]d: `cargo test --offline bench_ -- --ignored --nocapture`.
//! Set BENCH_EXPORT=<dir> to also write each pair (fixed/moving NIfTI + truth) for ANTs/SimpleITK.

use super::*;
use crate::layers::{resample, Grid};

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 40) as f32) / (1u64 << 24) as f32
}

pub fn add_noise(v: &mut Volume, sigma: f32, seed: u64) {
    let mut s = seed;
    for x in &mut v.data {
        let g = lcg(&mut s) + lcg(&mut s) + lcg(&mut s) + lcg(&mut s) - 2.0; // ~N(0, 1/3)
        *x += g * 1.732 * sigma;
    }
}

pub fn grid_of(dims: [usize; 3], vs: [f64; 3], center: [f64; 3]) -> Grid {
    let mut a = volume::identity();
    for i in 0..3 {
        a[i][i] = vs[i];
        a[i][3] = center[i] - (dims[i] as f64 - 1.0) / 2.0 * vs[i];
    }
    Grid { dims, affine: a }
}

/// Analytic head-like phantom on any grid (mm units, centred on the world origin).
pub fn phantom_on(g: &Grid) -> Volume {
    let n = g.dims[0] * g.dims[1] * g.dims[2];
    let mut data = vec![0f32; n];
    for k in 0..g.dims[2] {
        for j in 0..g.dims[1] {
            for i in 0..g.dims[0] {
                let p = volume::apply(&g.affine, [i as f64, j as f64, k as f64]);
                let e = (p[0] / 70.0).powi(2) + (p[1] / 90.0).powi(2) + (p[2] / 60.0).powi(2);
                let mut v = if e < 1.0 { 100.0 } else { 0.0 };
                if e < 0.55 {
                    v += 15.0 * ((p[0] / 9.0).sin() + (p[2] / 11.0).cos()) as f32;
                }
                let b = ((p[0] - 25.0).powi(2) + (p[1] - 20.0).powi(2) + (p[2] - 10.0).powi(2)).sqrt();
                if b < 18.0 {
                    v += 80.0;
                }
                let d = ((p[0] + 20.0).powi(2) + (p[1] + 10.0).powi(2) + (p[2] - 25.0).powi(2)).sqrt();
                if d < 14.0 {
                    v -= 60.0;
                }
                data[i + g.dims[0] * (j + g.dims[1] * k)] = v;
            }
        }
    }
    Volume { dims: g.dims, data, affine: g.affine, min: -60.0, max: 195.0 }
}

pub struct Case {
    pub name: String,
    pub fixed: Volume,
    pub moving: Volume,
    /// World transform that moved the content: moving(tm p) = fixed(p).
    pub tm: Affine,
    pub opts: Options,
}

/// Move `fixed`'s content by `truth` onto `mgrid`, then change contrast and add noise.
pub fn make_case(name: &str, fixed: Volume, truth: &TransformParams, mgrid: &Grid, contrast: fn(f32, f32) -> f32, noise: f32, bias: f32, opts: Options) -> Case {
    let g = Grid::of(&fixed);
    let tm = truth.matrix(g.center());
    let mut moving = resample(&fixed, mgrid, false, &tm);
    let mx = fixed.max.max(1.0);
    let [nx, ny, _] = moving.dims;
    for (i, v) in moving.data.iter_mut().enumerate() {
        *v = contrast(*v, mx);
        if bias > 0.0 {
            let w = volume::apply(&moving.affine, [(i % nx) as f64, ((i / nx) % ny) as f64, (i / (nx * ny)) as f64]);
            *v *= 1.0 + bias * ((w[0] / 50.0).sin() as f32 + (w[2] / 40.0).cos() as f32) * 0.5;
        }
    }
    if noise > 0.0 {
        add_noise(&mut moving, noise * mx, 7);
    }
    let (mut mn, mut mxx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in &moving.data {
        mn = mn.min(v);
        mxx = mxx.max(v);
    }
    moving.min = mn;
    moving.max = mxx;
    Case { name: name.into(), fixed, moving, tm, opts }
}

pub fn same(v: f32, _mx: f32) -> f32 {
    v
}
pub fn invert(v: f32, mx: f32) -> f32 {
    mx * 1.2 - v
}
pub fn nonlinear(v: f32, mx: f32) -> f32 {
    255.0 * (1.0 - (v.max(0.0) / mx).powf(0.6)) + 20.0 * (v / mx * 9.0).sin()
}

fn eval_points(fixed: &Volume) -> Vec<[f64; 3]> {
    let thr = fixed.max * 0.4;
    let n = fixed.data.iter().filter(|&&v| v > thr).count().max(1);
    let stride = (n / 400).max(1);
    let [nx, ny, _] = fixed.dims;
    let mut out = Vec::new();
    let mut c = 0;
    for (i, &v) in fixed.data.iter().enumerate() {
        if v > thr {
            if c % stride == 0 {
                out.push(volume::apply(&fixed.affine, [(i % nx) as f64, ((i / nx) % ny) as f64, (i / (nx * ny)) as f64]));
            }
            c += 1;
        }
    }
    out
}

pub struct Outcome {
    pub secs: f64,
    pub mean: f64,
    pub max: f64,
    pub text: String,
}

pub fn run_case(c: &Case) -> Outcome {
    let grid = Grid::of(&c.fixed);
    let prob = Problem {
        fixed: c.fixed.clone(),
        moving: c.moving.clone(),
        base: volume::identity(),
        start: TransformParams::default(),
        pivot: grid.center(),
    };
    let result = std::sync::Mutex::new(None);
    let t = std::time::Instant::now();
    run(prob, c.opts, &AtomicBool::new(false), &|m| {
        if let Msg::Done { params, text } = m {
            *result.lock().unwrap() = Some((params, text));
        }
    });
    let secs = t.elapsed().as_secs_f64();
    let (params, text) = result.into_inner().unwrap().unwrap();
    let found = params.matrix(grid.center());
    let (mut sum, mut max) = (0.0f64, 0.0f64);
    let pts = eval_points(&c.fixed);
    for p in &pts {
        let q = volume::apply(&found, volume::apply(&c.tm, *p));
        let e = (0..3).map(|a| (q[a] - p[a]).powi(2)).sum::<f64>().sqrt();
        sum += e;
        max = max.max(e);
    }
    Outcome { secs, mean: sum / pts.len() as f64, max, text }
}

#[cfg(test)]
fn export(c: &Case) {
    use std::io::Write;
    let Ok(dir) = std::env::var("BENCH_EXPORT") else { return };
    let d = std::path::Path::new(&dir).join(&c.name);
    std::fs::create_dir_all(&d).unwrap();
    for (n, v) in [("fixed.nii.gz", &c.fixed), ("moving.nii.gz", &c.moving)] {
        let mut hdr = nifti::NiftiHeader::default();
        hdr.sform_code = 1;
        hdr.qform_code = 0;
        hdr.srow_x = [0, 1, 2, 3].map(|k| v.affine[0][k] as f32);
        hdr.srow_y = [0, 1, 2, 3].map(|k| v.affine[1][k] as f32);
        hdr.srow_z = [0, 1, 2, 3].map(|k| v.affine[2][k] as f32);
        let vs = v.voxel_size();
        hdr.pixdim = [1.0, vs[0] as f32, vs[1] as f32, vs[2] as f32, 1.0, 1.0, 1.0, 1.0];
        let [nx, ny, nz] = v.dims;
        let arr = ndarray::Array3::<f32>::from_shape_fn((nx, ny, nz), |(i, j, k)| v.get(i, j, k));
        nifti::writer::WriterOptions::new(d.join(n)).reference_header(&hdr).write_nifti(&arr).unwrap();
    }
    let mut f = std::fs::File::create(d.join("truth.json")).unwrap();
    let pts = eval_points(&c.fixed);
    write!(f, "{}", serde_json::json!({"tm": c.tm, "points": pts, "metric": format!("{:?}", c.opts.metric), "mode": format!("{:?}", c.opts.mode)})).unwrap();
}

fn report(rows: &[(String, Outcome)]) {
    println!("\n{:<26} {:>8} {:>10} {:>10}  {}", "case", "secs", "mean mm", "max mm", "result");
    for (n, o) in rows {
        println!("{:<26} {:>8.2} {:>10.2} {:>10.2}  {}", n, o.secs, o.mean, o.max, o.text);
    }
}

fn p(t: [f64; 3], r: [f64; 3], s: [f64; 3], sh: [f64; 3]) -> TransformParams {
    TransformParams { translate: t, rotate: r, scale: s, shear: sh }
}

/// A head-like phantom at 2 mm (80x100x70), generated in code.
fn head() -> Volume {
    crate::testdata::head_phantom([80, 100, 70], [2.0; 3])
}

fn mk(metric: Metric, mode: Mode) -> Options {
    Options { metric, mode, init_com: true, ..Default::default() }
}

pub fn phantom_cases() -> Vec<Case> {
    let c0 = [0.0; 3];
    let fg = grid_of([60, 72, 56], [3.0, 3.0, 3.0], c0);
    let mg = grid_of([44, 50, 30], [4.0, 4.0, 6.0], [5.0, -8.0, 3.0]);
    let id = [1.0; 3];
    let z = [0.0; 3];
    vec![
        make_case("ph_rigid18_ncc", phantom_on(&fg), &p([12.0, -10.0, 8.0], [18.0, -15.0, 20.0], id, z), &fg, same, 0.0, 0.0, mk(Metric::Ncc, Mode::Rigid)),
        make_case("ph_rigid18_mi_inv", phantom_on(&fg), &p([12.0, -10.0, 8.0], [18.0, -15.0, 20.0], id, z), &fg, invert, 0.02, 0.0, mk(Metric::Mi, Mode::Rigid)),
        make_case(
            "ph_affine_scaleshear_ncc",
            phantom_on(&fg),
            &p([8.0, -6.0, 5.0], [9.0, -7.0, 11.0], [1.09, 0.93, 1.05], [0.05, 0.0, -0.04]),
            &fg,
            same,
            0.02,
            0.0,
            mk(Metric::Ncc, Mode::Affine),
        ),
        make_case(
            "ph_aniso_fov_mi_nonlin",
            phantom_on(&fg),
            &p([10.0, -8.0, 6.0], [10.0, -8.0, 14.0], [1.05, 0.96, 1.03], [0.03, 0.0, 0.0]),
            &mg,
            nonlinear,
            0.03,
            0.0,
            mk(Metric::Mi, Mode::Affine),
        ),
    ]
}

pub fn head_cases() -> Vec<Case> {
    let f = head();
    let g = Grid::of(&f);
    let c = g.center();
    let mg2 = grid_of([84, 100, 80], [2.0, 2.0, 2.0], [c[0] + 3.0, c[1] - 5.0, c[2] + 2.0]);
    let mg3 = grid_of([60, 72, 36], [2.5, 2.5, 5.0], [c[0] - 4.0, c[1] + 4.0, c[2] - 3.0]);
    let id = [1.0; 3];
    let z = [0.0; 3];
    let mg4 = grid_of([170, 190, 26], [1.0, 1.0, 4.0], [c[0] + 2.0, c[1] + 3.0, c[2] + 10.0]);
    vec![
        make_case("head_rot30_mi_nonlin", f.clone(), &p([25.0, -20.0, 15.0], [30.0, -20.0, 25.0], id, z), &mg2, nonlinear, 0.04, 0.0, mk(Metric::Mi, Mode::Rigid)),
        make_case(
            "head_slab_t2like_bias",
            f.clone(),
            &p([12.0, -9.0, 14.0], [8.0, -6.0, 10.0], [1.03, 0.97, 1.0], [0.0; 3]),
            &mg4,
            nonlinear,
            0.05,
            0.25,
            mk(Metric::Mi, Mode::Affine),
        ),
        make_case("head_rigid12_same_ncc", f.clone(), &p([10.0, -8.0, 6.0], [12.0, -8.0, 15.0], id, z), &mg2, same, 0.0, 0.0, mk(Metric::Ncc, Mode::Rigid)),
        make_case("head_affine12_mi_nonlin", f.clone(), &p([10.0, -8.0, 6.0], [12.0, -8.0, 15.0], [1.04, 0.97, 1.02], [0.02, 0.0, 0.0]), &mg2, nonlinear, 0.03, 0.0, mk(Metric::Mi, Mode::Affine)),
        make_case(
            "head_hard20_aniso_mi",
            f.clone(),
            &p([15.0, -12.0, 10.0], [20.0, -15.0, 18.0], [1.08, 0.94, 1.04], [0.04, 0.0, -0.03]),
            &mg3,
            nonlinear,
            0.03,
            0.0,
            mk(Metric::Mi, Mode::Affine),
        ),
    ]
}

/// Run all cases, print the table, and require every case to land within `tol_mean`/`tol_max` mm.
fn check_all(cases: Vec<Case>, tol_mean: f64, tol_max: f64) {
    let rows: Vec<_> = cases.into_iter().map(|c| {
        export(&c);
        let o = run_case(&c);
        (c.name.clone(), o)
    }).collect();
    report(&rows);
    for (n, o) in &rows {
        assert!(o.mean < tol_mean && o.max < tol_max, "{n}: mean {:.2} mm / max {:.2} mm", o.mean, o.max);
    }
}

#[test]
fn bench_phantoms() {
    check_all(phantom_cases(), 0.8, 1.5);
}

#[test]
#[ignore]
fn bench_real_head() {
    check_all(head_cases(), 0.8, 1.5);
}

fn banded(v: f32, mx: f32) -> f32 {
    // non-monotonic contrast (several intensity bands swap order)
    let t = (v.max(0.0) / mx).min(1.0);
    100.0 * ((t * 7.0).sin().abs() + 0.4 * (1.0 - t))
}

pub fn stress_cases() -> Vec<Case> {
    let f = head();
    let g = Grid::of(&f);
    let c = g.center();
    let mg2 = grid_of([84, 100, 80], [2.0, 2.0, 2.0], [c[0] + 3.0, c[1] - 5.0, c[2] + 2.0]);
    let mg4 = grid_of([120, 140, 24], [1.5, 1.5, 4.5], [c[0] + 2.0, c[1] + 3.0, c[2] + 10.0]);
    let id = [1.0; 3];
    let z = [0.0; 3];
    let ph = grid_of([60, 72, 56], [3.0, 3.0, 3.0], [0.0; 3]);
    // A T2-like case: the head phantom moved by a made-up oblique rigid+scale transform onto a
    // thick-slab grid, with different contrast.
    let truth = p([-3.0, 18.0, -12.0], [12.0, -6.0, 9.0], [1.02, 0.98, 1.01], [0.01, 0.0, -0.01]);
    let mg5 = grid_of([200, 200, 30], [0.9, 0.9, 4.0], [c[0] + 4.0, c[1] - 6.0, c[2] + 5.0]);
    let mut v = vec![
        make_case("head_t2_like_slab", f.clone(), &truth, &mg5, nonlinear, 0.04, 0.2, mk(Metric::Mi, Mode::Affine)),
        make_case("head_rot55_mi_nonlin", f.clone(), &p([20.0, -15.0, 12.0], [55.0, -35.0, 50.0], id, z), &mg2, nonlinear, 0.03, 0.0, mk(Metric::Mi, Mode::Rigid)),
        make_case("head_flip180z_mi", f.clone(), &p([5.0, 5.0, 0.0], [0.0, 0.0, 180.0], id, z), &mg2, nonlinear, 0.03, 0.0, mk(Metric::Mi, Mode::Rigid)),
        make_case(
            "head_offset60_slab_rot25",
            f.clone(),
            &p([60.0, -40.0, 20.0], [25.0, -10.0, 20.0], [1.02, 1.0, 0.98], z),
            &mg4,
            nonlinear,
            0.04,
            0.2,
            mk(Metric::Mi, Mode::Affine),
        ),
        make_case("head_banded_contrast", f.clone(), &p([12.0, -9.0, 7.0], [14.0, -9.0, 16.0], id, z), &mg2, banded, 0.03, 0.0, mk(Metric::Mi, Mode::Rigid)),
        make_case("ph_noise10_mi", phantom_on(&ph), &p([10.0, -8.0, 6.0], [15.0, -12.0, 16.0], id, z), &ph, invert, 0.10, 0.0, mk(Metric::Mi, Mode::Rigid)),
    ];
    // partial-FOV phantom, 25 degrees, NCC
    let slab = grid_of([40, 48, 14], [3.0, 3.0, 6.0], [4.0, -6.0, 2.0]);
    v.push(make_case("ph_slab_rot25_ncc", phantom_on(&ph), &p([14.0, -10.0, 8.0], [25.0, -20.0, 22.0], id, z), &slab, same, 0.02, 0.0, mk(Metric::Ncc, Mode::Rigid)));
    v
}

#[test]
#[ignore]
fn bench_stress() {
    check_all(stress_cases(), 0.8, 1.5);
}
