//! Pure (GUI-free) slice extraction and screen<->voxel mapping.

use crate::volume::Volume;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plane {
    Axial,
    Coronal,
    Sagittal,
}

impl Plane {
    pub const ALL: [Plane; 3] = [Plane::Axial, Plane::Coronal, Plane::Sagittal];
    pub fn name(self) -> &'static str {
        match self {
            Plane::Axial => "Axial",
            Plane::Coronal => "Coronal",
            Plane::Sagittal => "Sagittal",
        }
    }
    /// (horizontal axis, vertical axis, slice axis) as voxel axes (0=x/R, 1=y/A, 2=z/S).
    pub fn axes(self) -> (usize, usize, usize) {
        match self {
            Plane::Axial => (0, 1, 2),
            Plane::Coronal => (0, 2, 1),
            Plane::Sagittal => (1, 2, 0),
        }
    }
    /// Whether the horizontal axis is flipped on screen.
    /// Radiological: patient Right on screen-left (axial/coronal); sagittal: anterior on screen-left.
    pub fn flip_h(self, radiological: bool) -> bool {
        match self {
            Plane::Sagittal => true,
            _ => radiological,
        }
    }
    /// Labels for (left edge, right edge, top edge, bottom edge) of the image.
    pub fn edge_labels(self, radiological: bool) -> [&'static str; 4] {
        let f = self.flip_h(radiological);
        match self {
            Plane::Axial => [if f { "R" } else { "L" }, if f { "L" } else { "R" }, "A", "P"],
            Plane::Coronal => [if f { "R" } else { "L" }, if f { "L" } else { "R" }, "S", "I"],
            Plane::Sagittal => ["A", "P", "S", "I"],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WindowLevel {
    pub lo: f32,
    pub hi: f32,
}

impl WindowLevel {
    pub fn auto(v: &Volume) -> Self {
        // 1st..99th percentile of non-zero voxels (subsampled) for a robust initial window.
        let mut s: Vec<f32> = v
            .data
            .iter()
            .step_by((v.data.len() / 200_000).max(1))
            .copied()
            .filter(|x| x.is_finite() && *x != 0.0)
            .collect();
        if s.len() < 10 {
            return WindowLevel { lo: v.min, hi: if v.max > v.min { v.max } else { v.min + 1.0 } };
        }
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = s[s.len() / 100];
        let mut hi = s[s.len() * 99 / 100];
        if hi <= lo {
            hi = lo + 1.0;
        }
        WindowLevel { lo, hi }
    }
    pub fn to_u8(&self, x: f32) -> u8 {
        (((x - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0) * 255.0) as u8
    }
}

/// Geometry of a slice image placed in a screen rectangle.
#[derive(Clone, Copy, Debug)]
pub struct Placement {
    /// top-left of the image in screen px
    pub origin: [f32; 2],
    /// px per voxel along (horizontal, vertical)
    pub px: [f32; 2],
    pub nh: usize,
    pub nv: usize,
    pub flip_h: bool,
}

impl Placement {
    /// Fit the image into `pane` (x, y, w, h), preserving physical aspect, with zoom and pan (px).
    pub fn fit(dims: [usize; 3], vs: [f64; 3], p: Plane, radiological: bool, pane: [f32; 4], zoom: f32, pan: [f32; 2]) -> Self {
        let (ha, va, _) = p.axes();
        let (nh, nv) = (dims[ha], dims[va]);
        let (sh, sv) = (vs[ha] as f32, vs[va] as f32);
        let base = (pane[2] / (nh as f32 * sh)).min(pane[3] / (nv as f32 * sv)).max(1e-3);
        let scale = base * zoom;
        let (w, h) = (nh as f32 * sh * scale, nv as f32 * sv * scale);
        Placement {
            origin: [
                pane[0] + (pane[2] - w) * 0.5 + pan[0],
                pane[1] + (pane[3] - h) * 0.5 + pan[1],
            ],
            px: [sh * scale, sv * scale],
            nh,
            nv,
            flip_h: p.flip_h(radiological),
        }
    }
    pub fn size(&self) -> [f32; 2] {
        [self.nh as f32 * self.px[0], self.nv as f32 * self.px[1]]
    }
    /// Screen point -> (h voxel index, v voxel index) if inside the image.
    pub fn to_voxel(&self, pt: [f32; 2]) -> Option<(usize, usize)> {
        let c = ((pt[0] - self.origin[0]) / self.px[0]).floor();
        let r = ((pt[1] - self.origin[1]) / self.px[1]).floor();
        if c < 0.0 || r < 0.0 || c >= self.nh as f32 || r >= self.nv as f32 {
            return None;
        }
        let (c, r) = (c as usize, r as usize);
        Some((if self.flip_h { self.nh - 1 - c } else { c }, self.nv - 1 - r))
    }
    /// Screen position of the centre of voxel (h, v).
    pub fn voxel_center(&self, h: usize, v: usize) -> [f32; 2] {
        let c = if self.flip_h { self.nh - 1 - h } else { h };
        let r = self.nv - 1 - v;
        [
            self.origin[0] + (c as f32 + 0.5) * self.px[0],
            self.origin[1] + (r as f32 + 0.5) * self.px[1],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::identity;

    fn vol(dims: [usize; 3]) -> Volume {
        let mut a = identity();
        a[2][2] = 2.0; // anisotropic z
        Volume { dims, data: vec![0.0; dims[0] * dims[1] * dims[2]], affine: a, min: 0.0, max: 1.0 }
    }

    #[test]
    fn placement_roundtrip_with_zoom_pan() {
        let v = vol([40, 50, 30]);
        for p in Plane::ALL {
            for rad in [true, false] {
                let pl = Placement::fit(v.dims, v.voxel_size(), p, rad, [10.0, 20.0, 300.0, 200.0], 1.7, [13.0, -8.0]);
                let (ha, va, _) = p.axes();
                for &(h, vi) in &[(0, 0), (v.dims[ha] - 1, v.dims[va] - 1), (3, 7)] {
                    let c = pl.voxel_center(h, vi);
                    assert_eq!(pl.to_voxel(c), Some((h, vi)));
                }
            }
        }
    }

    #[test]
    fn anisotropic_aspect_is_physical() {
        let v = vol([40, 50, 30]); // z voxel = 2mm
        let pl = Placement::fit(v.dims, v.voxel_size(), Plane::Coronal, true, [0.0, 0.0, 400.0, 400.0], 1.0, [0.0, 0.0]);
        assert!((pl.px[1] / pl.px[0] - 2.0).abs() < 1e-4);
    }
}
