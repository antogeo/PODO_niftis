//! Automatic affine registration of a moving image onto the reference.
//!
//! ANTs-style intensity registration in pure Rust:
//! - coarse-to-fine pyramid (8, 4, 2, 1 mm; block-averaged, never finer than the data),
//! - Mattes-style mutual information (cubic B-spline Parzen window, 32 bins) or normalised
//!   cross-correlation, with analytic gradients w.r.t. the 12 affine parameters,
//! - L-BFGS with a backtracking line search on parameters scaled to millimetres,
//! - random (jittered-stride) voxel sampling at fine levels,
//! - several starting points tried at the coarsest levels and the best one kept: as given,
//!   foreground centres of mass, FOV centres, and the 24 axis-aligned plus principal-axes
//!   orientations about the centres of mass (robust to large rotations and flips; the foreground
//!   is found independently of contrast, so this also works across modalities).
//!
//! The result is a `TransformParams` relative to the layer's applied alignment, so it lands in
//! the manual alignment fields.

use crate::layers::TransformParams;
use crate::volume::{self, Affine, Volume};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Metric {
    /// Normalised cross-correlation: same modality / contrast.
    Ncc,
    /// Mutual information (Mattes): works across modalities (e.g. T1 vs T2).
    Mi,
}

impl Metric {
    pub fn name(self) -> &'static str {
        match self {
            Metric::Ncc => "Correlation (same contrast)",
            Metric::Mi => "Mutual information (any contrast)",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Translation + rotation.
    Rigid,
    /// Rigid first, then the full 12-parameter affine.
    Affine,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Rigid => "Rigid (6 DOF)",
            Mode::Affine => "Affine (12 DOF)",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub metric: Metric,
    pub mode: Mode,
    /// Also try starting by matching centres of mass / FOV centres.
    pub init_com: bool,
    /// Also try principal-axes orientations at the coarse levels (robust to large rotations).
    pub multi_start: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { metric: Metric::Mi, mode: Mode::Affine, init_com: true, multi_start: true }
    }
}

pub enum Msg {
    /// A resolution level finished; `params` is the best transform so far.
    Progress { text: String, params: TransformParams },
    Done { text: String, params: TransformParams },
}

pub struct Problem {
    /// The reference image on the reference grid.
    pub fixed: Volume,
    /// The original (un-resampled) moving image.
    pub moving: Volume,
    /// The layer's applied alignment; the search is relative to it.
    pub base: Affine,
    pub start: TransformParams,
    pub pivot: [f64; 3],
}

type Vec12 = [f64; 12];

/// Block-average downsampling by integer factors (remainder voxels at the far end are dropped).
pub fn downsample(v: &Volume, f: [usize; 3]) -> Volume {
    let f = [0, 1, 2].map(|a| f[a].clamp(1, v.dims[a]));
    let nd = [0, 1, 2].map(|a| v.dims[a] / f[a]);
    let mut data = vec![0f32; nd[0] * nd[1] * nd[2]];
    data.par_chunks_mut(nd[0] * nd[1]).enumerate().for_each(|(k, slab)| {
        for j in 0..nd[1] {
            for i in 0..nd[0] {
                let (mut sum, mut cnt) = (0f64, 0u32);
                for dk in 0..f[2] {
                    for dj in 0..f[1] {
                        for di in 0..f[0] {
                            let x = v.get(i * f[0] + di, j * f[1] + dj, k * f[2] + dk);
                            if x.is_finite() {
                                sum += x as f64;
                                cnt += 1;
                            }
                        }
                    }
                }
                slab[i + nd[0] * j] = if cnt > 0 { (sum / cnt as f64) as f32 } else { 0.0 };
            }
        }
    });
    let mut s = volume::identity();
    for a in 0..3 {
        s[a][a] = f[a] as f64;
        s[a][3] = (f[a] as f64 - 1.0) / 2.0;
    }
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in &data {
        min = min.min(x);
        max = max.max(x);
    }
    Volume { dims: nd, data, affine: volume::mul(&v.affine, &s), min, max }
}

/// Trilinear value and gradient (intensity per voxel along each axis) at voxel coordinates `p`.
#[inline]
fn trilinear_grad(v: &Volume, p: [f64; 3]) -> Option<(f32, [f32; 3])> {
    let [nx, ny, nz] = v.dims;
    if !(p[0] >= 0.0 && p[1] >= 0.0 && p[2] >= 0.0 && p[0] <= (nx - 1) as f64 && p[1] <= (ny - 1) as f64 && p[2] <= (nz - 1) as f64) {
        return None;
    }
    let (x0, y0, z0) = (p[0] as usize, p[1] as usize, p[2] as usize);
    let (x1, y1, z1) = ((x0 + 1).min(nx - 1), (y0 + 1).min(ny - 1), (z0 + 1).min(nz - 1));
    let (tx, ty, tz) = ((p[0] - x0 as f64) as f32, (p[1] - y0 as f64) as f32, (p[2] - z0 as f64) as f32);
    let g = |x, y, z| v.data[x + nx * (y + ny * z)];
    let (c000, c100, c010, c110) = (g(x0, y0, z0), g(x1, y0, z0), g(x0, y1, z0), g(x1, y1, z0));
    let (c001, c101, c011, c111) = (g(x0, y0, z1), g(x1, y0, z1), g(x0, y1, z1), g(x1, y1, z1));
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    let val = c0 * (1.0 - tz) + c1 * tz;
    let gx = (1.0 - ty) * (1.0 - tz) * (c100 - c000) + ty * (1.0 - tz) * (c110 - c010) + (1.0 - ty) * tz * (c101 - c001) + ty * tz * (c111 - c011);
    let gy = (1.0 - tx) * (1.0 - tz) * (c010 - c000) + tx * (1.0 - tz) * (c110 - c100) + (1.0 - tx) * tz * (c011 - c001) + tx * tz * (c111 - c101);
    let gz = (1.0 - tx) * (1.0 - ty) * (c001 - c000) + tx * (1.0 - ty) * (c101 - c100) + (1.0 - tx) * ty * (c011 - c010) + tx * ty * (c111 - c110);
    Some((val, [gx, gy, gz]))
}

const NB: usize = 32;

/// Cubic B-spline and its derivative.
#[inline]
fn bspline3(x: f32) -> f32 {
    let a = x.abs();
    if a < 1.0 {
        2.0 / 3.0 - a * a + 0.5 * a * a * a
    } else if a < 2.0 {
        let t = 2.0 - a;
        t * t * t / 6.0
    } else {
        0.0
    }
}
#[inline]
fn dbspline3(x: f32) -> f32 {
    let a = x.abs();
    let d = if a < 1.0 {
        -2.0 * a + 1.5 * a * a
    } else if a < 2.0 {
        let t = 2.0 - a;
        -0.5 * t * t
    } else {
        0.0
    };
    if x < 0.0 {
        -d
    } else {
        d
    }
}

/// 0.5 .. 99.5 percentile range of the finite voxel values (robust to bright outliers).
fn robust_range(v: &Volume) -> (f32, f32) {
    // a strided subset of at most ~250k voxels is plenty for percentiles
    let stride = (v.data.len() / 250_000).max(1);
    let mut d: Vec<f32> = v.data.iter().step_by(stride).copied().filter(|x| x.is_finite()).collect();
    if d.is_empty() {
        return (0.0, 1.0);
    }
    let q = |d: &mut Vec<f32>, f: f64| {
        let k = ((d.len() - 1) as f64 * f) as usize;
        *d.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap()).1
    };
    let lo = q(&mut d, 0.005);
    let hi = q(&mut d, 0.995);
    (lo, if hi <= lo { lo + 1.0 } else { hi })
}

#[derive(Clone, Copy, Default)]
struct Rec {
    m: f32,
    g: [f32; 3],
    ok: bool,
}

enum Ev {
    /// Too little overlap: a continuous penalty, no gradient.
    Penalty(f64),
    Mi { value: f64, n: f64, p: Vec<f64> },
    Ncc { value: f64, n: f64, fbar: f64, mbar: f64, sf: f64, sm: f64 },
}

impl Ev {
    fn value(&self) -> f64 {
        match self {
            Ev::Penalty(v) => *v,
            Ev::Mi { value, .. } | Ev::Ncc { value, .. } => *value,
        }
    }
}

/// One resolution level: the moving image, the fixed samples and the similarity machinery.
struct Lvl<'a> {
    mv: std::borrow::Cow<'a, Volume>,
    /// (base * moving affine)^-1: world -> moving voxel.
    mb: Affine,
    xs: Vec<[f64; 4]>,
    fs: Vec<f32>,
    fb: Vec<u8>,
    recs: Vec<Rec>,
    metric: Metric,
    mlo: f32,
    mhi: f32,
    total: usize,
    /// RMS radius of the sampled foreground about the pivot (mm).
    radius: f64,
}

fn hash(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl<'a> Lvl<'a> {
    fn new(fx: &Volume, mv: std::borrow::Cow<'a, Volume>, base: &Affine, metric: Metric, max_samples: usize, pivot: [f64; 3]) -> Lvl<'a> {
        let n = fx.data.len();
        let (flo, fhi) = robust_range(fx);
        let (mlo, mhi) = robust_range(&mv);
        let fscale = NB as f32 / (fhi - flo);
        let thr = {
            let stride = (n / 250_000).max(1);
            let (mut s, mut c) = (0f64, 0usize);
            for &v in fx.data.iter().step_by(stride) {
                if v.is_finite() {
                    s += v as f64;
                    c += 1;
                }
            }
            (s / c.max(1) as f64) as f32
        };
        // Jittered-stride sampling: one random voxel in each block of `stride` voxels.
        let (count, stride) = if n <= max_samples { (n, 1) } else { (max_samples, n / max_samples) };
        let [nx, ny, _] = fx.dims;
        let (mut xs, mut fs, mut fb) = (Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count));
        let (mut r2, mut rc) = (0f64, 0usize);
        for t in 0..count {
            let i = if stride == 1 { t } else { t * stride + (hash(t as u64) % stride as u64) as usize };
            let f = fx.data[i];
            if !f.is_finite() {
                continue;
            }
            let w = volume::apply(&fx.affine, [(i % nx) as f64, ((i / nx) % ny) as f64, (i / (nx * ny)) as f64]);
            xs.push([w[0], w[1], w[2], 1.0]);
            fs.push(f);
            fb.push((((f - flo) * fscale).max(0.0) as usize).min(NB - 1) as u8);
            if f > thr {
                r2 += (0..3).map(|a| (w[a] - pivot[a]).powi(2)).sum::<f64>();
                rc += 1;
            }
        }
        let radius = if rc > 0 { (r2 / rc as f64).sqrt().clamp(20.0, 150.0) } else { 60.0 };
        let total = xs.len();
        Lvl {
            mb: volume::invert(&volume::mul(base, &mv.affine)),
            mv,
            xs,
            fs,
            fb,
            recs: vec![Rec::default(); total],
            metric,
            mlo,
            mhi,
            total,
            radius,
        }
    }

    /// Parzen window position of an intensity: (first bin of the 4-bin support, u).
    #[inline]
    fn parzen(&self, m: f32) -> (usize, f32) {
        let t = ((m - self.mlo) / (self.mhi - self.mlo)).clamp(0.0, 0.9999);
        let u = 1.0 + t * (NB as f32 - 3.0);
        (u as usize - 1, u)
    }

    /// Sample the moving image through `q` (fixed world -> moving voxel) and evaluate the metric.
    fn eval(&mut self, q: &Affine) -> Ev {
        let (r0, r1, r2) = (q[0], q[1], q[2]);
        let mv = &self.mv;
        self.recs.par_iter_mut().zip(self.xs.par_iter()).for_each(|(rec, x)| {
            let p = [
                r0[0] * x[0] + r0[1] * x[1] + r0[2] * x[2] + r0[3],
                r1[0] * x[0] + r1[1] * x[1] + r1[2] * x[2] + r1[3],
                r2[0] * x[0] + r2[1] * x[1] + r2[2] * x[2] + r2[3],
            ];
            match trilinear_grad(mv, p) {
                Some((m, g)) if m.is_finite() => *rec = Rec { m, g, ok: true },
                _ => rec.ok = false,
            }
        });
        const CH: usize = 8192;
        match self.metric {
            Metric::Mi => {
                let this = &*self;
                let (h, n) = this
                    .recs
                    .par_chunks(CH)
                    .zip(this.fb.par_chunks(CH))
                    .fold(
                        || (vec![0f64; NB * NB], 0f64),
                        |mut acc, (rc, fbc)| {
                            for (r, &fb) in rc.iter().zip(fbc) {
                                if !r.ok {
                                    continue;
                                }
                                let (b0, u) = this.parzen(r.m);
                                for j in 0..4 {
                                    acc.0[fb as usize * NB + b0 + j] += bspline3(u - (b0 + j) as f32) as f64;
                                }
                                acc.1 += 1.0;
                            }
                            acc
                        },
                    )
                    .reduce(
                        || (vec![0f64; NB * NB], 0f64),
                        |mut a, b| {
                            for (x, y) in a.0.iter_mut().zip(&b.0) {
                                *x += y;
                            }
                            a.1 += b.1;
                            a
                        },
                    );
                let frac = n / self.total as f64;
                if n < 50.0 || frac < 0.05 {
                    return Ev::Penalty(-(1.0 - frac));
                }
                let p: Vec<f64> = h.iter().map(|x| x / n).collect();
                let (mut pf, mut pm) = ([0f64; NB], [0f64; NB]);
                for f in 0..NB {
                    for b in 0..NB {
                        pf[f] += p[f * NB + b];
                        pm[b] += p[f * NB + b];
                    }
                }
                let mut mi = 0.0;
                for f in 0..NB {
                    for b in 0..NB {
                        let x = p[f * NB + b];
                        if x > 1e-12 {
                            mi += x * (x / (pf[f] * pm[b])).ln();
                        }
                    }
                }
                Ev::Mi { value: mi, n, p }
            }
            Metric::Ncc => {
                let this = &*self;
                let (n, sf, sm, sff, smm, sfm) = this
                    .recs
                    .par_chunks(CH)
                    .zip(this.fs.par_chunks(CH))
                    .map(|(rc, fc)| {
                        let mut a = [0f64; 6];
                        for (r, &f) in rc.iter().zip(fc) {
                            if r.ok {
                                let (f, m) = (f as f64, r.m as f64);
                                a[0] += 1.0;
                                a[1] += f;
                                a[2] += m;
                                a[3] += f * f;
                                a[4] += m * m;
                                a[5] += f * m;
                            }
                        }
                        a
                    })
                    .reduce(
                        || [0f64; 6],
                        |mut a, b| {
                            for i in 0..6 {
                                a[i] += b[i];
                            }
                            a
                        },
                    )
                    .into();
                let frac = n / self.total as f64;
                if n < 50.0 || frac < 0.05 {
                    return Ev::Penalty(-1.0 - (1.0 - frac));
                }
                let (fbar, mbar) = (sf / n, sm / n);
                let vf = sff / n - fbar * fbar;
                let vm = smm / n - mbar * mbar;
                if vf <= 1e-12 || vm <= 1e-12 {
                    return Ev::Penalty(-1.0);
                }
                let cov = sfm / n - fbar * mbar;
                let (sfd, smd) = (vf.sqrt(), vm.sqrt());
                Ev::Ncc { value: cov / (sfd * smd), n, fbar, mbar, sf: sfd, sm: smd }
            }
        }
    }

    /// H = sum_s w_s * grad_s (x) [x_s;1]: the gradient of the metric w.r.t. the 3x4 matrix `q`.
    fn grad_h(&self, ev: &Ev) -> [[f64; 4]; 3] {
        const CH: usize = 8192;
        let acc = match ev {
            Ev::Penalty(_) => return [[0.0; 4]; 3],
            Ev::Mi { n, p, .. } => {
                // L[f][b] = ln(p(f,b) / pm(b))
                let mut pm = [0f64; NB];
                for f in 0..NB {
                    for b in 0..NB {
                        pm[b] += p[f * NB + b];
                    }
                }
                let mut l = vec![0f32; NB * NB];
                for f in 0..NB {
                    for b in 0..NB {
                        let x = p[f * NB + b];
                        if x > 1e-12 && pm[b] > 0.0 {
                            l[f * NB + b] = (x / pm[b]).ln() as f32;
                        }
                    }
                }
                let dudm = (NB as f32 - 3.0) / (self.mhi - self.mlo);
                let inv_n = (1.0 / n) as f32;
                self.recs
                    .par_chunks(CH)
                    .zip(self.fb.par_chunks(CH))
                    .zip(self.xs.par_chunks(CH))
                    .map(|((rc, fbc), xc)| {
                        let mut h = [0f64; 12];
                        for ((r, &fb), x) in rc.iter().zip(fbc).zip(xc) {
                            if !r.ok || r.m <= self.mlo || r.m >= self.mhi {
                                continue;
                            }
                            let (b0, u) = self.parzen(r.m);
                            let mut w = 0f32;
                            for j in 0..4 {
                                w += l[fb as usize * NB + b0 + j] * dbspline3(u - (b0 + j) as f32);
                            }
                            let w = (w * dudm * inv_n) as f64;
                            for i in 0..3 {
                                let wg = w * r.g[i] as f64;
                                for k in 0..4 {
                                    h[i * 4 + k] += wg * x[k];
                                }
                            }
                        }
                        h
                    })
                    .reduce(|| [0f64; 12], add12)
            }
            Ev::Ncc { value, n, fbar, mbar, sf, sm } => {
                let (c1, c2) = (1.0 / (sf * sm), value / (sm * sm));
                self.recs
                    .par_chunks(CH)
                    .zip(self.fs.par_chunks(CH))
                    .zip(self.xs.par_chunks(CH))
                    .map(|((rc, fc), xc)| {
                        let mut h = [0f64; 12];
                        for ((r, &f), x) in rc.iter().zip(fc).zip(xc) {
                            if !r.ok {
                                continue;
                            }
                            let w = ((f as f64 - fbar) * c1 - (r.m as f64 - mbar) * c2) / n;
                            for i in 0..3 {
                                let wg = w * r.g[i] as f64;
                                for k in 0..4 {
                                    h[i * 4 + k] += wg * x[k];
                                }
                            }
                        }
                        h
                    })
                    .reduce(|| [0f64; 12], add12)
            }
        };
        let mut h = [[0.0; 4]; 3];
        for i in 0..3 {
            for k in 0..4 {
                h[i][k] = acc[i * 4 + k];
            }
        }
        h
    }
}

fn add12(mut a: [f64; 12], b: [f64; 12]) -> [f64; 12] {
    for i in 0..12 {
        a[i] += b[i];
    }
    a
}

/// Parameters in "millimetre" units: 1 unit of any parameter moves a point at the foreground's
/// RMS radius by about 1 mm. Maps to TransformParams' natural units.
struct Scaling {
    unit: Vec12,
}

const IDENT: Vec12 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];

impl Scaling {
    fn new(radius: f64) -> Scaling {
        let r = radius;
        Scaling { unit: [1.0, 1.0, 1.0, (180.0 / std::f64::consts::PI) / r, (180.0 / std::f64::consts::PI) / r, (180.0 / std::f64::consts::PI) / r, 1.0 / r, 1.0 / r, 1.0 / r, 1.0 / r, 1.0 / r, 1.0 / r] }
    }
    fn params(&self, theta: &Vec12) -> TransformParams {
        let mut v = IDENT;
        for k in 0..12 {
            v[k] += self.unit[k] * theta[k];
        }
        TransformParams { translate: [v[0], v[1], v[2]], rotate: [v[3], v[4], v[5]], scale: [v[6], v[7], v[8]], shear: [v[9], v[10], v[11]] }
    }
    fn matrix(&self, theta: &Vec12, pivot: [f64; 3]) -> Affine {
        self.params(theta).matrix(pivot)
    }
    fn clamp(&self, theta: &mut Vec12) {
        for k in 6..9 {
            theta[k] = theta[k].clamp((0.5 - 1.0) / self.unit[k], (2.0 - 1.0) / self.unit[k]);
        }
        for k in 9..12 {
            theta[k] = theta[k].clamp(-0.3 / self.unit[k], 0.3 / self.unit[k]);
        }
    }
}

/// Negative metric and its gradient w.r.t. `theta` (the first `free` parameters).
fn objective(l: &mut Lvl, sc: &Scaling, ym: &Affine, pivot: [f64; 3], theta: &Vec12, free: usize) -> (f64, Vec12) {
    let d = sc.matrix(theta, pivot);
    let q = volume::mul(ym, &d);
    let ev = l.eval(&q);
    let value = ev.value();
    let mut g = [0.0; 12];
    if !matches!(ev, Ev::Penalty(_)) {
        let h = l.grad_h(&ev);
        for k in 0..free {
            let eps = 1e-4;
            let (mut tp, mut tm) = (*theta, *theta);
            tp[k] += eps;
            tm[k] -= eps;
            let (dp, dm) = (sc.matrix(&tp, pivot), sc.matrix(&tm, pivot));
            let mut dd = [[0.0; 4]; 4];
            for i in 0..3 {
                for j in 0..4 {
                    dd[i][j] = (dp[i][j] - dm[i][j]) / (2.0 * eps);
                }
            }
            // G = Ym * dD (top 3 rows)
            let gk = volume::mul(ym, &dd);
            let mut s = 0.0;
            for i in 0..3 {
                for j in 0..4 {
                    s += gk[i][j] * h[i][j];
                }
            }
            g[k] = -s;
        }
    }
    (-value, g)
}

fn dot(a: &Vec12, b: &Vec12) -> f64 {
    (0..12).map(|i| a[i] * b[i]).sum()
}

/// L-BFGS with backtracking line search over the first `free` parameters.
fn minimize(
    f: &mut dyn FnMut(&Vec12) -> (f64, Vec12),
    sc: &Scaling,
    mut x: Vec12,
    max_iter: usize,
    step0: f64,
    cancel: &AtomicBool,
) -> (Vec12, f64) {
    const M: usize = 8;
    let (mut fx, mut g) = f(&x);
    let mut hist: Vec<(Vec12, Vec12, f64)> = Vec::new();
    let mut stalls = 0;
    for _ in 0..max_iter {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let gn = dot(&g, &g).sqrt();
        if gn < 1e-12 {
            break;
        }
        let steepest = |g: &Vec12| -> Vec12 {
            let mut d = [0.0; 12];
            for k in 0..12 {
                d[k] = -g[k] * step0 / gn;
            }
            d
        };
        let mut d = if hist.is_empty() {
            steepest(&g)
        } else {
            let mut q = g;
            let mut alpha = vec![0.0; hist.len()];
            for (i, (s, y, rho)) in hist.iter().enumerate().rev() {
                alpha[i] = rho * dot(s, &q);
                for k in 0..12 {
                    q[k] -= alpha[i] * y[k];
                }
            }
            let (s, y, _) = hist.last().unwrap();
            let gamma = dot(s, y) / dot(y, y);
            for k in 0..12 {
                q[k] *= gamma;
            }
            for (i, (s, y, rho)) in hist.iter().enumerate() {
                let beta = rho * dot(y, &q);
                for k in 0..12 {
                    q[k] += s[k] * (alpha[i] - beta);
                }
            }
            for k in 0..12 {
                q[k] = -q[k];
            }
            q
        };
        let mut gd = dot(&g, &d);
        if gd >= 0.0 {
            hist.clear();
            d = steepest(&g);
            gd = dot(&g, &d);
        }
        let dn = dot(&d, &d).sqrt();
        let mut t = (20.0 / dn).min(1.0);
        let mut accepted = None;
        for _ in 0..14 {
            let mut xn = x;
            for k in 0..12 {
                xn[k] += t * d[k];
            }
            sc.clamp(&mut xn);
            let (fnew, gnew) = f(&xn);
            if fnew <= fx + 1e-4 * t * gd {
                accepted = Some((xn, fnew, gnew));
                break;
            }
            t *= 0.5;
        }
        let Some((xn, fnew, gnew)) = accepted else {
            if hist.is_empty() {
                break;
            }
            hist.clear();
            continue;
        };
        let s: Vec12 = std::array::from_fn(|k| xn[k] - x[k]);
        let y: Vec12 = std::array::from_fn(|k| gnew[k] - g[k]);
        let sy = dot(&s, &y);
        if sy > 1e-14 {
            if hist.len() == M {
                hist.remove(0);
            }
            hist.push((s, y, 1.0 / sy));
        }
        let drop = fx - fnew;
        let sn = dot(&s, &s).sqrt();
        x = xn;
        g = gnew;
        let prev = fx;
        fx = fnew;
        if drop < 1e-9 * (prev.abs() + fx.abs() + 1e-9) || sn < 2e-4 {
            stalls += 1;
            if stalls >= 3 {
                break;
            }
        } else {
            stalls = 0;
        }
    }
    (x, fx)
}

/// Centroid and covariance (world mm) of the foreground, independent of contrast: the background
/// is the most frequent intensity, the foreground everything that clearly differs from it.
fn moments(v: &Volume) -> Option<([f64; 3], [[f64; 3]; 3])> {
    let (lo, hi) = robust_range(v);
    let range = (hi - lo).max(1e-9);
    const HB: usize = 64;
    let mut hist = [0usize; HB];
    for &x in &v.data {
        if x.is_finite() {
            hist[((((x - lo) / range) * HB as f32).max(0.0) as usize).min(HB - 1)] += 1;
        }
    }
    let top = (0..HB).max_by_key(|&b| hist[b])?;
    let bg = lo + (top as f32 + 0.5) / HB as f32 * range;
    let thr = 0.12 * range;
    let [nx, ny, _] = v.dims;
    let (mut n, mut s) = (0.0, [0.0; 3]);
    let mut pts = Vec::new();
    for (i, &x) in v.data.iter().enumerate() {
        if x.is_finite() && (x - bg).abs() > thr {
            let p = volume::apply(&v.affine, [(i % nx) as f64, ((i / nx) % ny) as f64, (i / (nx * ny)) as f64]);
            n += 1.0;
            for a in 0..3 {
                s[a] += p[a];
            }
            pts.push(p);
        }
    }
    if n < 20.0 {
        return None;
    }
    let c = s.map(|x| x / n);
    let mut cov = [[0.0; 3]; 3];
    for p in pts {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += (p[i] - c[i]) * (p[j] - c[j]) / n;
            }
        }
    }
    Some((c, cov))
}

/// The 24 proper rotations that map the coordinate axes onto themselves.
fn cube_rotations() -> Vec<[[f64; 3]; 3]> {
    let mut out = Vec::new();
    for perm in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        for signs in 0..8u32 {
            let mut r = [[0.0; 3]; 3];
            for c in 0..3 {
                r[perm[c]][c] = if signs >> c & 1 == 1 { -1.0 } else { 1.0 };
            }
            let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
            if det > 0.0 {
                out.push(r);
            }
        }
    }
    out
}

/// Jacobi eigen-decomposition of a symmetric 3x3: eigenvalues descending, eigenvectors as columns
/// (right-handed).
fn eig_sym3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..60 {
        let (mut p, mut q, mut big) = (0, 1, 0.0);
        for i in 0..3 {
            for j in i + 1..3 {
                if a[i][j].abs() > big {
                    big = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if big < 1e-12 {
            break;
        }
        let th = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let (s, c) = th.sin_cos();
        for k in 0..3 {
            let (akp, akq) = (a[k][p], a[k][q]);
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let (apk, aqk) = (a[p][k], a[q][k]);
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
        }
        for k in 0..3 {
            let (vkp, vkq) = (v[k][p], v[k][q]);
            v[k][p] = c * vkp - s * vkq;
            v[k][q] = s * vkp + c * vkq;
        }
    }
    let mut idx = [0, 1, 2];
    idx.sort_by(|&i, &j| a[j][j].partial_cmp(&a[i][i]).unwrap());
    let vals = idx.map(|i| a[i][i]);
    let mut vec = [[0.0; 3]; 3];
    for (c, &i) in idx.iter().enumerate() {
        for r in 0..3 {
            vec[r][c] = v[r][i];
        }
    }
    let det = vec[0][0] * (vec[1][1] * vec[2][2] - vec[1][2] * vec[2][1]) - vec[0][1] * (vec[1][0] * vec[2][2] - vec[1][2] * vec[2][0])
        + vec[0][2] * (vec[1][0] * vec[2][1] - vec[1][1] * vec[2][0]);
    if det < 0.0 {
        for r in 0..3 {
            vec[r][2] = -vec[r][2];
        }
    }
    (vals, vec)
}

fn fix_bottom(mut a: Affine) -> Affine {
    a[3] = [0.0, 0.0, 0.0, 1.0];
    a
}

fn factors(vol: &Volume, target_mm: f64) -> [usize; 3] {
    vol.voxel_size().map(|s| ((target_mm / s.max(1e-6)).round() as usize).max(1))
}

/// World position of the geometric centre of a volume's grid.
fn grid_center(v: &Volume) -> [f64; 3] {
    volume::apply(&v.affine, [(v.dims[0] as f64 - 1.0) / 2.0, (v.dims[1] as f64 - 1.0) / 2.0, (v.dims[2] as f64 - 1.0) / 2.0])
}

/// Candidate starting transforms Z (fixed world -> moving world, before `base`): the
/// translation-only ones (always optimised) and the rotated ones (ranked first by raw similarity).
fn candidates(fx: &Volume, mv: &Volume, z0: &Affine, base: &Affine, opts: &Options) -> (Vec<Affine>, Vec<Affine>) {
    let mut out = vec![*z0];
    let mut rotated = Vec::new();
    if !opts.init_com && !opts.multi_start {
        return (out, rotated);
    }
    let shifted = |z: &Affine, from: [f64; 3], to_moving_world: [f64; 3]| -> Affine {
        // translate so that z(from) = base(to_moving_world)
        let want = volume::apply(base, to_moving_world);
        let now = volume::apply(z, from);
        let mut n = *z;
        for a in 0..3 {
            n[a][3] += want[a] - now[a];
        }
        n
    };
    let (mf, mm) = (moments(fx), moments(mv));
    if opts.init_com {
        if let (Some((cf, _)), Some((cm, _))) = (&mf, &mm) {
            out.push(shifted(z0, *cf, *cm));
        }
        if let Some((cf, _)) = &mf {
            out.push(shifted(z0, *cf, grid_center(mv)));
        }
        out.push(shifted(z0, grid_center(fx), grid_center(mv)));
    }
    if opts.multi_start {
        // Rotations about the foreground centroids: the 24 axis-aligned orientations plus the
        // principal-axes alignment (4 sign choices).
        if let (Some((cf, covf)), Some((cm, covm))) = (&mf, &mm) {
            let (_, ef) = eig_sym3(*covf);
            let (_, em) = eig_sym3(*covm);
            let mut rots = cube_rotations();
            for s in [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]] {
                let mut r = [[0.0; 3]; 3];
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            r[i][j] += em[i][k] * s[k] * ef[j][k];
                        }
                    }
                }
                rots.push(r);
            }
            for r in rots {
                let mut z = volume::identity();
                for i in 0..3 {
                    for j in 0..3 {
                        z[i][j] = r[i][j];
                    }
                    z[i][3] = cm[i] - (0..3).map(|j| r[i][j] * cf[j]).sum::<f64>();
                }
                // z maps fixed world -> moving world; Z = base * z
                rotated.push(volume::mul(base, &z));
            }
        }
    }
    (out, rotated)
}

struct Step {
    target: f64,
    free: usize,
    iters: usize,
    samples: usize,
}

/// Run the registration, reporting after each resolution level. Blocks; call from a thread.
pub fn run(prob: Problem, opts: Options, cancel: &AtomicBool, send: &dyn Fn(Msg)) {
    let fine = prob.fixed.voxel_size().into_iter().fold(f64::INFINITY, f64::min);
    let mut plan = match opts.mode {
        Mode::Rigid => vec![
            Step { target: 8.0, free: 6, iters: 50, samples: usize::MAX },
            Step { target: 4.0, free: 6, iters: 40, samples: usize::MAX },
            Step { target: 2.0, free: 6, iters: 30, samples: 120_000 },
        ],
        Mode::Affine => vec![
            Step { target: 8.0, free: 6, iters: 50, samples: usize::MAX },
            Step { target: 4.0, free: 6, iters: 40, samples: usize::MAX },
            Step { target: 4.0, free: 12, iters: 40, samples: usize::MAX },
            Step { target: 2.0, free: 12, iters: 30, samples: 120_000 },
        ],
    };
    if fine <= 1.3 {
        let free = plan.last().unwrap().free;
        plan.push(Step { target: 1.0, free, iters: 15, samples: 200_000 });
    }
    let z_start = volume::invert(&prob.start.matrix(prob.pivot));
    let mut cands: Vec<Affine> = vec![z_start];
    let mut last_factors: Option<([usize; 3], [usize; 3], usize)> = None;
    let mut best = (z_start, f64::NAN);
    let nlev = plan.len();
    let mut cur_params = prob.start;
    for (li, st) in plan.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let (ff, fm) = (factors(&prob.fixed, st.target), factors(&prob.moving, st.target));
        if last_factors == Some((ff, fm, st.free)) {
            continue;
        }
        let first = last_factors.is_none();
        last_factors = Some((ff, fm, st.free));
        let fx_owned;
        let fx: &Volume = if ff == [1, 1, 1] {
            &prob.fixed
        } else {
            fx_owned = downsample(&prob.fixed, ff);
            &fx_owned
        };
        let mv: std::borrow::Cow<Volume> = if fm == [1, 1, 1] {
            std::borrow::Cow::Borrowed(&prob.moving)
        } else {
            std::borrow::Cow::Owned(downsample(&prob.moving, fm))
        };
        let mut rotated = Vec::new();
        if first {
            (cands, rotated) = candidates(fx, &mv, &z_start, &prob.base, &opts);
        }
        let mut lvl = Lvl::new(fx, mv, &prob.base, opts.metric, st.samples, prob.pivot);
        let sc = Scaling::new(lvl.radius);
        if !rotated.is_empty() {
            // Rank the rotated starting points by raw similarity; optimise the best few too.
            let mut scored: Vec<(f64, Affine)> = rotated
                .iter()
                .map(|z| {
                    let q = volume::mul(&lvl.mb, z);
                    (lvl.eval(&q).value(), *z)
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            cands.extend(scored.iter().take(6).map(|s| s.1));
        }
        // Many starts: a short screening run for each, then the full run for the two best.
        let screen = if cands.len() > 3 { 25.min(st.iters) } else { st.iters };
        let optimise = |lvl: &mut Lvl, starts: &[Affine], iters: usize| -> Vec<(Affine, f64)> {
            let mut out = Vec::new();
            for z0 in starts {
                if cancel.load(Ordering::Relaxed) && !out.is_empty() {
                    break;
                }
                let ym = volume::mul(&lvl.mb, z0);
                let mut f = |t: &Vec12| objective(lvl, &sc, &ym, prob.pivot, t, st.free);
                let (theta, fval) = minimize(&mut f, &sc, [0.0; 12], iters, 0.5 * st.target, cancel);
                out.push((fix_bottom(volume::mul(z0, &sc.matrix(&theta, prob.pivot))), -fval));
            }
            // best first, distinct optima only (several starts often converge to the same one)
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut distinct: Vec<(Affine, f64)> = Vec::new();
            for r in out {
                let dup = distinct.iter().any(|d| (0..3).all(|i| (0..4).all(|j| (d.0[i][j] - r.0[i][j]).abs() < if j == 3 { 0.5 } else { 0.01 })));
                if !dup {
                    distinct.push(r);
                }
            }
            distinct
        };
        let mut results = optimise(&mut lvl, &cands, screen);
        let keep = if li == 0 { 2 } else { 1 };
        results.truncate(if screen < st.iters { 3 } else { keep });
        if screen < st.iters {
            let starts: Vec<Affine> = results.iter().map(|r| r.0).collect();
            results = optimise(&mut lvl, &starts, st.iters - screen);
            results.truncate(keep);
        }
        best = results[0];
        cands = results.iter().map(|r| r.0).collect();
        if let Ok(p) = TransformParams::from_matrix(&fix_bottom(volume::invert(&best.0)), prob.pivot) {
            cur_params = p;
        }
        send(Msg::Progress {
            text: format!("level {}/{} ({} mm): score {:.4}", li + 1, nlev, st.target, best.1),
            params: cur_params,
        });
    }
    let text = if cancel.load(Ordering::Relaxed) {
        "registration cancelled (kept the best result so far)".to_string()
    } else {
        format!("registration finished: {} score {:.4}", if opts.metric == Metric::Mi { "MI" } else { "NCC" }, best.1)
    };
    send(Msg::Done { text, params: cur_params });
}

#[cfg(test)]
mod bench;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{resample, Grid};

    /// A smooth anisotropic "head": bright ellipsoid with an off-centre bright blob and a dark core.
    fn phantom() -> Volume {
        let dims = [48, 56, 44];
        let mut affine = volume::identity();
        affine[0][0] = 4.0;
        affine[1][1] = 4.0;
        affine[2][2] = 4.0;
        affine[0][3] = -96.0;
        affine[1][3] = -112.0;
        affine[2][3] = -80.0;
        let mut data = vec![0f32; dims[0] * dims[1] * dims[2]];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = volume::apply(&affine, [i as f64, j as f64, k as f64]);
                    let e = (p[0] / 70.0).powi(2) + (p[1] / 90.0).powi(2) + (p[2] / 60.0).powi(2);
                    let mut v = if e < 1.0 { 100.0 } else { 0.0 };
                    let b = ((p[0] - 25.0).powi(2) + (p[1] - 20.0).powi(2) + (p[2] - 10.0).powi(2)).sqrt();
                    if b < 18.0 {
                        v += 80.0;
                    }
                    let d = ((p[0] + 20.0).powi(2) + (p[1] + 10.0).powi(2) + (p[2] - 25.0).powi(2)).sqrt();
                    if d < 14.0 {
                        v -= 60.0;
                    }
                    data[i + dims[0] * (j + dims[1] * k)] = v;
                }
            }
        }
        Volume { dims, data, affine, min: 0.0, max: 180.0 }
    }

    fn check(opts: Options, invert_contrast: bool) {
        let fixed = phantom();
        let grid = Grid::of(&fixed);
        let truth = TransformParams { translate: [9.0, -7.0, 6.0], rotate: [6.0, -5.0, 7.0], ..Default::default() };
        let tm = truth.matrix(grid.center());
        let mut moving = resample(&fixed, &grid, false, &tm);
        if invert_contrast {
            for v in &mut moving.data {
                *v = 200.0 - *v;
            }
        }
        let prob = Problem { fixed, moving, base: volume::identity(), start: TransformParams::default(), pivot: grid.center() };
        let result = std::sync::Mutex::new(None);
        run(prob, opts, &AtomicBool::new(false), &|m| {
            if let Msg::Done { params, .. } = m {
                *result.lock().unwrap() = Some(params);
            }
        });
        let found = result.into_inner().unwrap().unwrap().matrix(grid.center());
        // found should undo truth: found * truth ~ identity on points inside the head.
        let mut worst: f64 = 0.0;
        for p in [[30.0, 20.0, 10.0], [-30.0, 40.0, -20.0], [0.0, -50.0, 30.0], [40.0, -30.0, -10.0]] {
            let q = volume::apply(&found, volume::apply(&tm, p));
            worst = worst.max((0..3).map(|a| (q[a] - p[a]).powi(2)).sum::<f64>().sqrt());
        }
        assert!(worst < 2.5, "worst point error {worst:.2} mm with {opts:?}");
    }

    #[test]
    fn recovers_rigid_motion_ncc() {
        check(Options { metric: Metric::Ncc, mode: Mode::Rigid, ..Default::default() }, false);
    }

    #[test]
    fn recovers_motion_affine_mutual_information_across_inverted_contrast() {
        check(Options { metric: Metric::Mi, mode: Mode::Affine, ..Default::default() }, true);
    }

    #[test]
    fn downsample_preserves_world_position() {
        let v = phantom();
        let d = downsample(&v, [2, 2, 2]);
        assert_eq!(d.dims, [24, 28, 22]);
        // centre of the first coarse voxel = centre of the first 2x2x2 block in world space
        let a = volume::apply(&d.affine, [0.0, 0.0, 0.0]);
        let b = volume::apply(&v.affine, [0.5, 0.5, 0.5]);
        assert!((0..3).all(|i| (a[i] - b[i]).abs() < 1e-9));
    }

    /// The analytic gradient must agree with finite differences of the metric.
    #[test]
    fn analytic_gradient_matches_finite_differences() {
        let fixed = phantom();
        let grid = Grid::of(&fixed);
        let truth = TransformParams { translate: [4.0, -3.0, 2.0], rotate: [4.0, -3.0, 5.0], scale: [1.02, 0.98, 1.01], ..Default::default() };
        let tm = truth.matrix(grid.center());
        for metric in [Metric::Ncc, Metric::Mi] {
            let mut moving = resample(&fixed, &grid, false, &tm);
            if metric == Metric::Mi {
                for v in &mut moving.data {
                    *v = 250.0 - *v * 0.8;
                }
            }
            let mut lvl = Lvl::new(&fixed, std::borrow::Cow::Owned(moving), &volume::identity(), metric, usize::MAX, grid.center());
            let sc = Scaling::new(lvl.radius);
            // start 3 mm / 2 degrees away from the truth so the gradient is non-trivial
            let off = TransformParams { translate: [3.0, -2.0, 2.0], rotate: [2.0, -1.5, 2.0], ..Default::default() };
            let z0 = volume::mul(&tm, &off.matrix(grid.center()));
            let ym = volume::mul(&lvl.mb, &z0);
            let theta = [0.0; 12];
            let (_, g) = objective(&mut lvl, &sc, &ym, grid.center(), &theta, 12);
            let mut fd = [0.0; 12];
            for k in 0..12 {
                let h = 0.2;
                let (mut a, mut b) = (theta, theta);
                a[k] += h;
                b[k] -= h;
                fd[k] = (objective(&mut lvl, &sc, &ym, grid.center(), &a, 0).0 - objective(&mut lvl, &sc, &ym, grid.center(), &b, 0).0) / (2.0 * h);
            }
            let (dg, nf, ng) = (dot(&g, &fd), dot(&fd, &fd).sqrt(), dot(&g, &g).sqrt());
            assert!(nf > 1e-6, "{metric:?}: degenerate test (zero gradient)");
            // NCC is exact up to the sample set changing at the FOV edge; the Mattes-style MI
            // gradient (like ITK's) additionally ignores the change in the number of overlapping
            // samples, which finite differences do see.
            let (min_cos, max_ratio) = if metric == Metric::Ncc { (0.97, 0.15) } else { (0.92, 0.4) };
            assert!(dg / (nf * ng) > min_cos, "{metric:?}: gradient direction off: cos {:.3}\n g  {g:?}\n fd {fd:?}", dg / (nf * ng));
            assert!((ng / nf - 1.0).abs() < max_ratio, "{metric:?}: gradient magnitude off: ratio {:.3}", ng / nf);
        }
    }
}
