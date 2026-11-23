//! Seeded region growing, used to build masks by clicking on a spot.

use crate::volume::Volume;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    /// Connected voxels whose value lies inside an absolute [lo, hi] range.
    ConnectedThreshold,
    /// Connected voxels within +/- tolerance of the clicked voxel's value.
    SeedTolerance,
    /// Iteratively re-estimated mean +/- k*sd of the growing region (ITK "confidence connected").
    ConfidenceConnected,
    /// Every voxel within a radius (mm) of the seed, ignoring intensities.
    Ball,
}

impl Method {
    pub const ALL: [Method; 4] =
        [Method::ConnectedThreshold, Method::SeedTolerance, Method::ConfidenceConnected, Method::Ball];
    pub fn name(self) -> &'static str {
        match self {
            Method::ConnectedThreshold => "Connected threshold",
            Method::SeedTolerance => "Seed +/- tolerance",
            Method::ConfidenceConnected => "Confidence connected",
            Method::Ball => "Ball (radius)",
        }
    }
    pub fn help(self) -> &'static str {
        match self {
            Method::ConnectedThreshold => "Grow through neighbours whose value is within a range: relative to the clicked voxel (e.g. -5% / +15%, or -4 / +4 intensity) or an absolute [min, max].",
            Method::SeedTolerance => "Grow through neighbours within +/- tolerance of the clicked voxel's value.",
            Method::ConfidenceConnected => {
                "Start from the clicked neighbourhood, then repeatedly accept values within mean +/- k*SD of the region found so far."
            }
            Method::Ball => "Select everything within the radius, regardless of intensity.",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Connectivity {
    Face6,
    Edge18,
    Corner26,
}

impl Connectivity {
    pub const ALL: [Connectivity; 3] = [Connectivity::Face6, Connectivity::Edge18, Connectivity::Corner26];
    pub fn name(self) -> &'static str {
        match self {
            Connectivity::Face6 => "6 (faces)",
            Connectivity::Edge18 => "18 (+edges)",
            Connectivity::Corner26 => "26 (+corners)",
        }
    }
    fn offsets(self) -> Vec<[i32; 3]> {
        let max = match self {
            Connectivity::Face6 => 1,
            Connectivity::Edge18 => 2,
            Connectivity::Corner26 => 3,
        };
        let mut v = Vec::new();
        for dz in -1..=1i32 {
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let n = dx.abs() + dy.abs() + dz.abs();
                    if n >= 1 && n <= max {
                        v.push([dx, dy, dz]);
                    }
                }
            }
        }
        v
    }
}

/// How the connected-threshold range is given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RangeMode {
    /// Offsets below/above the clicked voxel's value.
    RelativeToSeed,
    /// A fixed [min, max].
    Absolute,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OffsetUnit {
    /// Percent of the seed value's magnitude.
    Percent,
    /// Intensity units.
    Intensity,
}

impl OffsetUnit {
    pub fn name(self) -> &'static str {
        match self {
            OffsetUnit::Percent => "%",
            OffsetUnit::Intensity => "intensity",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub range_mode: RangeMode,
    pub offset_unit: OffsetUnit,
    /// Connected threshold, relative mode: accept [seed - minus, seed + plus] (both >= 0).
    pub minus: f32,
    pub plus: f32,
    pub method: Method,
    pub lo: f32,
    pub hi: f32,
    pub tolerance: f32,
    /// Confidence-connected: SD multiplier and maximum number of re-estimations.
    pub k: f32,
    pub iterations: u32,
    pub radius_mm: f32,
    pub connectivity: Connectivity,
    /// Restrict growth to the slice that was clicked: `Some(axis)`.
    pub slice_axis: Option<usize>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            range_mode: RangeMode::RelativeToSeed,
            offset_unit: OffsetUnit::Percent,
            minus: 10.0,
            plus: 10.0,
            method: Method::ConnectedThreshold,
            lo: 0.0,
            hi: 1.0,
            tolerance: 10.0,
            k: 2.5,
            iterations: 5,
            radius_mm: 5.0,
            connectivity: Connectivity::Face6,
            slice_axis: None,
        }
    }
}

/// Flood fill from `seed` through voxels accepted by `ok`; returns linear indices.
fn flood(vol: &Volume, seed: [usize; 3], offs: &[[i32; 3]], slice_axis: Option<usize>, ok: &dyn Fn(f32) -> bool) -> Vec<usize> {
    let [nx, ny, nz] = vol.dims;
    let s = vol.idx(seed[0], seed[1], seed[2]);
    if !ok(vol.data[s]) {
        return Vec::new();
    }
    let mut seen = vec![false; vol.data.len()];
    seen[s] = true;
    let mut out = vec![s];
    let mut head = 0;
    while head < out.len() {
        let p = out[head];
        head += 1;
        let (i, j, k) = (p % nx, (p / nx) % ny, p / (nx * ny));
        for o in offs {
            if let Some(a) = slice_axis {
                if o[a] != 0 {
                    continue;
                }
            }
            let (ii, jj, kk) = (i as i32 + o[0], j as i32 + o[1], k as i32 + o[2]);
            if ii < 0 || jj < 0 || kk < 0 || ii >= nx as i32 || jj >= ny as i32 || kk >= nz as i32 {
                continue;
            }
            let q = ii as usize + nx * (jj as usize + ny * kk as usize);
            if !seen[q] && ok(vol.data[q]) {
                seen[q] = true;
                out.push(q);
            }
        }
    }
    out
}

fn mean_sd(vol: &Volume, idx: &[usize]) -> (f32, f32) {
    let n = idx.len() as f64;
    let mean = idx.iter().map(|&i| vol.data[i] as f64).sum::<f64>() / n;
    let var = idx.iter().map(|&i| (vol.data[i] as f64 - mean).powi(2)).sum::<f64>() / n;
    (mean as f32, var.sqrt() as f32)
}

/// The accepted [lo, hi] for connected threshold given the seed value.
pub fn threshold_range(p: &Params, seed_value: f32) -> (f32, f32) {
    match p.range_mode {
        RangeMode::Absolute => (p.lo.min(p.hi), p.lo.max(p.hi)),
        RangeMode::RelativeToSeed => {
            let (m, q) = (p.minus.abs(), p.plus.abs());
            match p.offset_unit {
                OffsetUnit::Percent => {
                    let a = seed_value.abs();
                    (seed_value - a * m / 100.0, seed_value + a * q / 100.0)
                }
                OffsetUnit::Intensity => (seed_value - m, seed_value + q),
            }
        }
    }
}

/// Voxels (linear indices into `vol.data`) selected by growing from `seed`.
pub fn grow(vol: &Volume, seed: [usize; 3], p: &Params) -> Vec<usize> {
    if (0..3).any(|a| seed[a] >= vol.dims[a]) {
        return Vec::new();
    }
    let offs = p.connectivity.offsets();
    let sv = vol.get(seed[0], seed[1], seed[2]);
    if !sv.is_finite() && p.method != Method::Ball {
        return Vec::new();
    }
    match p.method {
        Method::ConnectedThreshold => {
            let (lo, hi) = threshold_range(p, sv);
            flood(vol, seed, &offs, p.slice_axis, &|v| v >= lo && v <= hi)
        }
        Method::SeedTolerance => {
            let t = p.tolerance.abs();
            flood(vol, seed, &offs, p.slice_axis, &|v| (v - sv).abs() <= t)
        }
        Method::ConfidenceConnected => {
            // Initial statistics from the seed's 3x3x3 neighbourhood (in-slice if restricted).
            let mut init = Vec::new();
            for dk in -1..=1i32 {
                for dj in -1..=1i32 {
                    for di in -1..=1i32 {
                        let d = [di, dj, dk];
                        if p.slice_axis.is_some_and(|a| d[a] != 0) {
                            continue;
                        }
                        let q = [seed[0] as i32 + di, seed[1] as i32 + dj, seed[2] as i32 + dk];
                        if (0..3).all(|a| q[a] >= 0 && (q[a] as usize) < vol.dims[a]) {
                            let v = vol.get(q[0] as usize, q[1] as usize, q[2] as usize);
                            if v.is_finite() {
                                init.push(vol.idx(q[0] as usize, q[1] as usize, q[2] as usize));
                            }
                        }
                    }
                }
            }
            let (mut mean, mut sd) = mean_sd(vol, &init);
            let mut region = Vec::new();
            for _ in 0..p.iterations.max(1) {
                let (lo, hi) = (mean - p.k * sd, mean + p.k * sd);
                // The seed must be inside the range, otherwise nothing grows from it.
                let next = flood(vol, seed, &offs, p.slice_axis, &|v| v >= lo && v <= hi);
                let changed = next.len() != region.len();
                region = next;
                if region.is_empty() || !changed {
                    break;
                }
                (mean, sd) = mean_sd(vol, &region);
            }
            region
        }
        Method::Ball => {
            let vs = vol.voxel_size();
            let r = p.radius_mm.max(0.0) as f64;
            let reach = [0, 1, 2].map(|a| (r / vs[a].max(1e-9)).floor() as i64);
            let mut out = Vec::new();
            for dk in -reach[2]..=reach[2] {
                for dj in -reach[1]..=reach[1] {
                    for di in -reach[0]..=reach[0] {
                        let d = [di, dj, dk];
                        if p.slice_axis.is_some_and(|a| d[a] != 0) {
                            continue;
                        }
                        let dist = (0..3).map(|a| (d[a] as f64 * vs[a]).powi(2)).sum::<f64>().sqrt();
                        if dist > r {
                            continue;
                        }
                        let q = [seed[0] as i64 + di, seed[1] as i64 + dj, seed[2] as i64 + dk];
                        if (0..3).all(|a| q[a] >= 0 && (q[a] as usize) < vol.dims[a]) {
                            out.push(vol.idx(q[0] as usize, q[1] as usize, q[2] as usize));
                        }
                    }
                }
            }
            out
        }
    }
}

/// Voxels (linear indices) of a brush stamp centred on `c`: a disk in the plane spanned by
/// `axes.0`/`axes.1`, or with `three_d` a ball. `radius_mm` is measured with the voxel sizes `vs`.
pub fn brush_indices(dims: [usize; 3], vs: [f64; 3], c: [usize; 3], axes: (usize, usize, usize), radius_mm: f64, three_d: bool) -> Vec<usize> {
    let r = radius_mm.max(0.0);
    let reach = |a: usize| (r / vs[a].max(1e-9)).floor() as i64;
    let rs = if three_d { reach(axes.2) } else { 0 };
    let mut out = Vec::new();
    for ds in -rs..=rs {
        for dv in -reach(axes.1)..=reach(axes.1) {
            for dh in -reach(axes.0)..=reach(axes.0) {
                let d = [(axes.0, dh), (axes.1, dv), (axes.2, ds)];
                let dist = d.iter().map(|&(a, n)| (n as f64 * vs[a]).powi(2)).sum::<f64>().sqrt();
                if dist > r + 1e-9 {
                    continue;
                }
                let mut q = [c[0] as i64, c[1] as i64, c[2] as i64];
                for &(a, n) in &d {
                    q[a] += n;
                }
                if (0..3).all(|a| q[a] >= 0 && (q[a] as usize) < dims[a]) {
                    out.push(q[0] as usize + dims[0] * (q[1] as usize + dims[1] * q[2] as usize));
                }
            }
        }
    }
    out
}

/// Voxel positions from just after `a` up to and including `b`, at most `spacing` voxels apart.
pub fn segment_points(a: [usize; 3], b: [usize; 3], spacing: f64) -> Vec<[usize; 3]> {
    let d = [0, 1, 2].map(|i| b[i] as f64 - a[i] as f64);
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let n = (len / spacing.max(1.0)).ceil().max(1.0) as usize;
    let mut out = Vec::new();
    for s in 1..=n {
        let t = s as f64 / n as f64;
        let p = [0, 1, 2].map(|i| (a[i] as f64 + d[i] * t).round() as usize);
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::identity;

    /// 10^3 volume: background 0, a 3x3x3 block of 100 at (2..5), a separate block of 100 at (7..9).
    fn vol() -> Volume {
        let dims = [10, 10, 10];
        let mut data = vec![0f32; 1000];
        let mut set = |r: std::ops::Range<usize>| {
            for k in r.clone() {
                for j in r.clone() {
                    for i in r.clone() {
                        data[i + 10 * (j + 10 * k)] = 100.0;
                    }
                }
            }
        };
        set(2..5);
        set(7..9);
        Volume { dims, data, affine: identity(), min: 0.0, max: 100.0 }
    }

    fn p(method: Method) -> Params {
        Params { method, lo: 50.0, hi: 150.0, range_mode: RangeMode::Absolute, ..Default::default() }
    }

    #[test]
    fn connected_threshold_takes_only_the_connected_block() {
        let r = grow(&vol(), [3, 3, 3], &p(Method::ConnectedThreshold));
        assert_eq!(r.len(), 27);
        assert_eq!(grow(&vol(), [7, 7, 7], &p(Method::ConnectedThreshold)).len(), 8);
    }

    #[test]
    fn relative_range_in_percent_and_intensity() {
        let mut q = Params::default();
        q.minus = 5.0;
        q.plus = 15.0;
        let (lo, hi) = threshold_range(&q, 200.0);
        assert!((lo - 190.0).abs() < 1e-4 && (hi - 230.0).abs() < 1e-4);
        q.offset_unit = OffsetUnit::Intensity;
        q.minus = 4.0;
        q.plus = 4.0;
        assert_eq!(threshold_range(&q, 100.0), (96.0, 104.0));
        // Negative seed values: percent applies to the magnitude, range stays ordered.
        q.offset_unit = OffsetUnit::Percent;
        q.minus = 10.0;
        q.plus = 20.0;
        let (lo, hi) = threshold_range(&q, -100.0);
        assert!((lo + 110.0).abs() < 1e-4 && (hi + 80.0).abs() < 1e-4);
    }

    #[test]
    fn relative_threshold_grows_around_the_clicked_value() {
        // block of 100 (seed): +-5 % keeps it, a neighbour of 110 is excluded; +10 intensity includes it.
        let mut v = vol();
        v.data[3 + 10 * (3 + 10 * 5)] = 110.0; // touches the block from above (z=5)
        let mut q = Params { minus: 5.0, plus: 5.0, ..Default::default() };
        assert_eq!(grow(&v, [3, 3, 3], &q).len(), 27);
        q.plus = 10.0;
        assert_eq!(grow(&v, [3, 3, 3], &q).len(), 28);
    }

    #[test]
    fn seed_outside_range_gives_nothing() {
        assert!(grow(&vol(), [0, 0, 0], &p(Method::ConnectedThreshold)).is_empty());
    }

    #[test]
    fn seed_tolerance_follows_seed_value() {
        let mut q = p(Method::SeedTolerance);
        q.tolerance = 5.0;
        assert_eq!(grow(&vol(), [3, 3, 3], &q).len(), 27);
        // Seeding in the background grows the whole background (everything but the two blocks).
        assert_eq!(grow(&vol(), [0, 0, 0], &q).len(), 1000 - 27 - 8);
    }

    #[test]
    fn single_slice_restriction() {
        let mut q = p(Method::ConnectedThreshold);
        q.slice_axis = Some(2);
        assert_eq!(grow(&vol(), [3, 3, 3], &q).len(), 9);
    }

    #[test]
    fn connectivity_matters_for_diagonal_voxels() {
        let mut v = vol();
        v.data.iter_mut().for_each(|x| *x = 0.0);
        for (i, j) in [(1usize, 1usize), (2, 2)] {
            v.data[i + 10 * (j + 10 * 5)] = 100.0;
        }
        let mut q = p(Method::ConnectedThreshold);
        assert_eq!(grow(&v, [1, 1, 5], &q).len(), 1);
        q.connectivity = Connectivity::Edge18;
        assert_eq!(grow(&v, [1, 1, 5], &q).len(), 2);
    }

    #[test]
    fn confidence_connected_stops_at_the_edge_of_the_block() {
        let q = p(Method::ConfidenceConnected);
        assert_eq!(grow(&vol(), [3, 3, 3], &q).len(), 27);
    }

    #[test]
    fn ball_respects_voxel_size_and_bounds() {
        let mut q = p(Method::Ball);
        q.radius_mm = 1.0;
        assert_eq!(grow(&vol(), [5, 5, 5], &q).len(), 7); // centre + 6 face neighbours
        assert_eq!(grow(&vol(), [0, 0, 0], &q).len(), 4); // clipped at the corner
    }

    #[test]
    fn brush_disk_and_ball() {
        let dims = [20, 20, 20];
        let vs = [1.0, 1.0, 1.0];
        let disk = brush_indices(dims, vs, [10, 10, 10], (0, 1, 2), 2.0, false);
        assert_eq!(disk.len(), 13); // radius-2 disk in one slice
        let ball = brush_indices(dims, vs, [10, 10, 10], (0, 1, 2), 2.0, true);
        assert_eq!(ball.len(), 33);
        assert_eq!(brush_indices(dims, vs, [10, 10, 10], (0, 1, 2), 0.0, true).len(), 1);
        // clipped at the border, and anisotropic voxels shrink the reach along that axis
        assert_eq!(brush_indices(dims, vs, [0, 0, 5], (0, 1, 2), 1.0, false).len(), 3);
        assert_eq!(brush_indices(dims, [1.0, 2.0, 1.0], [10, 10, 10], (0, 1, 2), 2.0, false).len(), 7);
    }

    #[test]
    fn segment_points_leave_no_gaps() {
        let pts = segment_points([0, 0, 0], [10, 0, 0], 2.0);
        assert_eq!(pts.last(), Some(&[10, 0, 0]));
        assert!(pts.len() >= 5 && pts.windows(2).all(|w| w[1][0] - w[0][0] <= 2));
        assert_eq!(segment_points([3, 3, 3], [3, 3, 3], 1.0), vec![[3, 3, 3]]);
    }
}
