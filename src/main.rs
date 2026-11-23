#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod layers;
mod region;
mod register;
#[cfg(test)]
mod testdata;
mod view;
mod volume;

use eframe::egui::{self, Color32, ColorImage, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions, Vec2};
use layers::{Colormap, Grid, ImageArg, Layer, TransformParams};
use region::{Connectivity, Method, OffsetUnit, RangeMode};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use view::{Placement, Plane, WindowLevel};

#[derive(Clone, Copy, PartialEq)]
enum Layout {
    OneByThree,
    TwoByTwo,
    Single(Plane),
}

struct PaneState {
    zoom: f32,
    pan: [f32; 2],
    tex: Option<TextureHandle>,
}

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Navigate,
    /// Clicking a voxel grows a region from it into the selected mask layer.
    Region,
    /// Left-drag paints (Shift or the eraser erases) into the selected mask.
    Brush,
}

struct Brush {
    radius_mm: f32,
    /// Paint a ball through neighbouring slices instead of a flat disk.
    three_d: bool,
    label: u32,
    erase: bool,
}

/// Settings and state of the automatic-registration UI.
#[derive(Default)]
struct RegUi {
    opts: register::Options,
    /// Set by the UI when the user pressed "Register" on this layer.
    request: Option<usize>,
    /// Layer being registered right now.
    running: Option<usize>,
    status: String,
    cancel: bool,
}

struct RegJob {
    layer: usize,
    name: String,
    rx: mpsc::Receiver<register::Msg>,
    cancel: Arc<AtomicBool>,
}

/// A region selected by clicking a seed voxel, shown in yellow until it is added to a mask.
struct Pending {
    seed: [usize; 3],
    /// Slice axis of the pane that was clicked (for the 2D restriction).
    sa: usize,
    idx: Vec<usize>,
    mask: Vec<bool>,
    /// Settings the selection was computed with; it is recomputed live when they change.
    key: (region::Params, bool, usize),
}

enum LayerAction {
    Remove(usize),
}

struct App {
    /// Bottom to top.
    layers: Vec<Layer>,
    /// Reference grid, defined by the first loaded image.
    grid: Option<Grid>,
    selected: usize,
    /// Show the colormap/opacity controls under every image instead of only for the selected one.
    controls_each: bool,
    cursor: [usize; 3],
    layout: Layout,
    radiological: bool,
    /// Resample an oblique first image onto world axes when loading it.
    deoblique: bool,
    panes: [PaneState; 3],
    error: Option<String>,
    /// Last success message (e.g. "saved transform to ...").
    notice: Option<String>,
    tool: Tool,
    region: region::Params,
    /// Only grow within the slice that was clicked.
    region_in_slice: bool,
    /// Layer index the region is grown on (must not be a mask).
    region_src: usize,
    pending: Option<Pending>,
    brush: Brush,
    /// Last voxel painted in the current stroke (None between strokes).
    brush_last: Option<[usize; 3]>,
    reg_ui: RegUi,
    reg_job: Option<RegJob>,
    range_set: bool,
    mask_count: usize,
}

impl App {
    fn empty() -> Self {
        App {
            layers: Vec::new(),
            grid: None,
            selected: 0,
            controls_each: false,
            cursor: [0; 3],
            layout: Layout::TwoByTwo,
            radiological: true,
            deoblique: false,
            panes: std::array::from_fn(|_| PaneState { zoom: 1.0, pan: [0.0, 0.0], tex: None }),
            error: None,
            notice: None,
            tool: Tool::Navigate,
            region: region::Params::default(),
            region_in_slice: false,
            region_src: 0,
            pending: None,
            brush: Brush { radius_mm: 3.0, three_d: false, label: 1, erase: false },
            brush_last: None,
            reg_ui: RegUi::default(),
            reg_job: None,
            range_set: false,
            mask_count: 0,
        }
    }

    fn new() -> Self {
        let mut app = App::empty();
        for a in std::env::args().skip(1) {
            match layers::parse_arg(&a) {
                Ok(arg) => app.add_layer(arg),
                Err(e) => app.error = Some(format!("{a}: {e}")),
            }
        }
        app
    }

    fn add_path(&mut self, path: &std::path::Path) {
        self.add_layer(ImageArg { path: path.to_string_lossy().into_owned(), ..Default::default() });
    }

    fn add_layer(&mut self, arg: ImageArg) {
        let path = std::path::Path::new(&arg.path);
        let (mut vol, info) = match volume::load_with_info(path) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                return;
            }
        };
        let first = self.grid.is_none();
        let mut load_notes = Vec::new();
        if info.axcodes != "RAS" {
            load_notes.push(format!("stored {}, shown as RAS", info.axcodes));
        }
        if info.obliquity_deg >= 0.5 {
            let straightened = if first && self.deoblique { volume::deoblique(&vol) } else { None };
            if let Some(d) = straightened {
                vol = d;
                load_notes.push(format!("oblique {:.1}\u{b0}: resampled to world axes", info.obliquity_deg));
            } else if first {
                load_notes.push(format!(
                    "oblique {:.1}\u{b0}: shown in its own voxel grid (tick 'Straighten oblique' and reload to use world axes)",
                    info.obliquity_deg
                ));
            } else {
                load_notes.push(format!("oblique {:.1}\u{b0} (aligned in world space)", info.obliquity_deg));
            }
        }
        if first {
            self.grid = Some(Grid::of(&vol));
            self.cursor = [vol.dims[0] / 2, vol.dims[1] / 2, vol.dims[2] / 2];
            for p in &mut self.panes {
                p.zoom = 1.0;
                p.pan = [0.0, 0.0];
            }
        }
        let grid = self.grid.clone().unwrap();
        let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let cm = arg.colormap.unwrap_or(if first { Colormap::Gray } else { Colormap::Hot });
        let mut layer = if first { Layer::reference(name, vol, cm) } else { Layer::new(name, vol, &grid, cm) };
        if !first {
            layer.clear_below = cm != Colormap::Label;
            if arg.opacity.is_none() {
                layer.opacity = 0.7;
            }
        }
        if let Some(o) = arg.opacity {
            layer.opacity = o;
        }
        if let Some(m) = arg.min {
            layer.wl.lo = m;
        }
        if let Some(m) = arg.max {
            layer.wl.hi = m;
        }
        if !load_notes.is_empty() {
            layer.load_note = Some(load_notes.join(", "));
            layer.update_note(&grid);
        }
        self.layers.push(layer);
        self.selected = self.layers.len() - 1;
        self.error = None;
    }

    fn apply(&mut self, a: LayerAction) {
        match a {
            LayerAction::Remove(i) if i < self.layers.len() => {
                self.layers.remove(i);
                self.pending = None;
                if i < self.region_src {
                    self.region_src -= 1;
                }
                if self.layers.is_empty() {
                    self.grid = None;
                }
                self.selected = self.selected.min(self.layers.len().saturating_sub(1));
            }
            _ => {}
        }
    }

    fn pane_ui(&mut self, ui: &mut egui::Ui, pi: usize, plane: Plane, rect: Rect) {
        let Some(grid) = self.grid.clone() else { return };
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let (ha, va, sa) = plane.axes();
        let rad = self.radiological;
        let vs = {
            let a = &grid.affine;
            let mut s = [0.0; 3];
            for c in 0..3 {
                s[c] = (a[0][c].powi(2) + a[1][c].powi(2) + a[2][c].powi(2)).sqrt();
            }
            s
        };

        // --- interaction ---
        let hover = resp.hover_pos();
        if resp.hovered() {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            // Ctrl+wheel (and pinch) arrive as zoom_delta; plain wheel as scroll.
            if zoom != 1.0 {
                let z = &mut self.panes[pi].zoom;
                *z = (*z * zoom).clamp(0.2, 40.0);
            } else if scroll != 0.0 {
                let step = if scroll > 0.0 { 1i64 } else { -1 };
                let n = grid.dims[sa] as i64;
                self.cursor[sa] = (self.cursor[sa] as i64 + step).clamp(0, n - 1) as usize;
            }
        }
        let alt_pan = resp.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.modifiers.alt);
        if resp.dragged_by(egui::PointerButton::Middle) || alt_pan {
            let d = resp.drag_delta();
            self.panes[pi].pan[0] += d.x;
            self.panes[pi].pan[1] += d.y;
        }
        if resp.dragged_by(egui::PointerButton::Secondary) {
            // Freeview-style on the selected layer: horizontal = window width, vertical = level.
            if let Some(l) = self.layers.get_mut(self.selected) {
                let d = resp.drag_delta();
                let range = (l.wl.hi - l.wl.lo).max(1e-6);
                let center = (l.wl.hi + l.wl.lo) * 0.5 + d.y * range * 0.004;
                let width = (range * (1.0 + d.x * 0.004)).max(1e-6);
                l.wl = WindowLevel { lo: center - width * 0.5, hi: center + width * 0.5 };
            }
        }
        let pane = [rect.min.x, rect.min.y, rect.width(), rect.height()];
        let pl = Placement::fit(grid.dims, vs, plane, rad, pane, self.panes[pi].zoom, self.panes[pi].pan);
        let (ctrl_held, alt_held) = ui.input(|i| (i.modifiers.command || i.modifiers.ctrl, i.modifiers.alt));
        if self.tool == Tool::Brush
            && !ctrl_held
            && !alt_held
            && (resp.dragged_by(egui::PointerButton::Primary) || resp.clicked_by(egui::PointerButton::Primary))
        {
            if let Some(h) = hover.or(resp.interact_pointer_pos()) {
                if let Some((hi, vi)) = pl.to_voxel([h.x, h.y]) {
                    let mut c = self.cursor;
                    c[ha] = hi;
                    c[va] = vi;
                    let erase = self.brush.erase || ui.input(|i| i.modifiers.shift);
                    self.brush_stroke(c, (ha, va, sa), vs, erase);
                }
            }
        }
        if ui.input(|i| !i.pointer.primary_down()) {
            self.brush_last = None;
        }
        if (self.tool != Tool::Brush || ctrl_held)
            && !alt_held
            && (resp.dragged_by(egui::PointerButton::Primary) || resp.clicked_by(egui::PointerButton::Primary))
        {
            if let Some(h) = hover.or(resp.interact_pointer_pos()) {
                if let Some((hi, vi)) = pl.to_voxel([h.x, h.y]) {
                    self.cursor[ha] = hi;
                    self.cursor[va] = vi;
                    if resp.clicked_by(egui::PointerButton::Primary) {
                        let (ctrl, shift) = ui.input(|i| (i.modifiers.command || i.modifiers.ctrl, i.modifiers.shift));
                        if self.tool == Tool::Region || ctrl {
                            self.select_at(sa);
                            if shift {
                                self.notice = Some("selection made - use 'Remove from mask' to subtract it".into());
                            }
                        }
                    }
                }
            }
        }
        // Right-click (without dragging) selects the voxel and offers region actions.
        if resp.secondary_clicked() {
            if let Some(h) = resp.interact_pointer_pos() {
                if let Some((hi, vi)) = pl.to_voxel([h.x, h.y]) {
                    self.cursor[ha] = hi;
                    self.cursor[va] = vi;
                    self.select_at(sa);
                }
            }
        }
        resp.context_menu(|ui| {
            let [i, j, k] = self.cursor;
            ui.label(format!("voxel [{i}, {j}, {k}]"));
            let n = self.pending.as_ref().map_or(0, |p| p.idx.len());
            ui.add_enabled_ui(n > 0, |ui| {
                if ui.button(format!("Add region to mask as new parcel ({n} voxels)")).clicked() {
                    self.commit(false);
                    ui.close();
                }
                if ui.button("Remove region from mask").clicked() {
                    self.commit(true);
                    ui.close();
                }
            });
            if ui.button("Clear selection").clicked() {
                self.pending = None;
                ui.close();
            }
        });
        if matches!(self.tool, Tool::Region | Tool::Brush) && resp.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        }

        // --- draw ---
        let rgba = layers::compose(&self.layers, plane, self.cursor[sa], rad, grid.dims);
        let mut rgba = rgba;
        if let Some(p) = &self.pending {
            layers::overlay_selection(&mut rgba, plane, self.cursor[sa], rad, grid.dims, &p.mask, [255, 235, 0], 0.7);
        }
        let img = ColorImage::from_rgba_unmultiplied([grid.dims[ha], grid.dims[va]], &rgba);
        match &mut self.panes[pi].tex {
            Some(t) => t.set(img, TextureOptions::NEAREST),
            None => self.panes[pi].tex = Some(ui.ctx().load_texture(plane.name(), img, TextureOptions::NEAREST)),
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        let size = pl.size();
        let img_rect = Rect::from_min_size(Pos2::new(pl.origin[0], pl.origin[1]), Vec2::new(size[0], size[1]));
        painter.image(
            self.panes[pi].tex.as_ref().unwrap().id(),
            img_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        // crosshair
        let c = pl.voxel_center(self.cursor[ha], self.cursor[va]);
        let stroke = Stroke::new(1.0, Color32::from_rgb(255, 60, 60));
        painter.line_segment([Pos2::new(img_rect.min.x, c[1]), Pos2::new(img_rect.max.x, c[1])], stroke);
        painter.line_segment([Pos2::new(c[0], img_rect.min.y), Pos2::new(c[0], img_rect.max.y)], stroke);
        if self.tool == Tool::Brush {
            if let Some(h) = hover.filter(|h| rect.contains(*h)) {
                let ppv = size[0] / grid.dims[ha] as f32;
                let r = (self.brush.radius_mm as f64 / vs[ha].max(1e-9)) as f32 * ppv;
                let col = if self.brush.erase || ui.input(|i| i.modifiers.shift) { Color32::from_rgb(255, 80, 80) } else { Color32::from_rgb(255, 235, 0) };
                painter.circle_stroke(h, r.max(2.0), Stroke::new(1.5, col));
            }
        }
        // orientation labels + title
        let lab = plane.edge_labels(rad);
        let col = Color32::from_rgb(255, 230, 120);
        let f = egui::FontId::proportional(14.0);
        painter.text(Pos2::new(rect.min.x + 6.0, rect.center().y), egui::Align2::LEFT_CENTER, lab[0], f.clone(), col);
        painter.text(Pos2::new(rect.max.x - 6.0, rect.center().y), egui::Align2::RIGHT_CENTER, lab[1], f.clone(), col);
        painter.text(Pos2::new(rect.center().x, rect.min.y + 4.0), egui::Align2::CENTER_TOP, lab[2], f.clone(), col);
        painter.text(Pos2::new(rect.center().x, rect.max.y - 4.0), egui::Align2::CENTER_BOTTOM, lab[3], f.clone(), col);
        painter.text(
            Pos2::new(rect.min.x + 6.0, rect.min.y + 4.0),
            egui::Align2::LEFT_TOP,
            format!("{}  {}/{}", plane.name(), self.cursor[sa] + 1, grid.dims[sa]),
            f,
            Color32::from_gray(200),
        );
        painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::from_gray(60)), egui::StrokeKind::Inside);
    }

    /// The image region growing reads from: the chosen layer if valid, else the first non-mask.
    fn region_source(&self) -> Option<usize> {
        self.layers
            .get(self.region_src)
            .filter(|l| !l.is_mask)
            .map(|_| self.region_src)
            .or_else(|| self.layers.iter().position(|l| !l.is_mask))
    }

    /// The mask edits go to: the selected mask, else the topmost mask, else (if `create`) a new one.
    fn target_mask(&mut self, create: bool) -> Option<usize> {
        if let Some(i) = self.layers.get(self.selected).filter(|l| l.is_mask).map(|_| self.selected) {
            return Some(i);
        }
        if let Some(i) = self.layers.iter().rposition(|l| l.is_mask) {
            return Some(i);
        }
        if !create {
            return None;
        }
        let g = self.grid.clone()?;
        self.mask_count += 1;
        self.layers.push(Layer::mask(format!("mask{}", self.mask_count), &g));
        Some(self.layers.len() - 1)
    }

    /// Stamp the brush at `c` (joining it to the previous stamp of this stroke).
    fn brush_stroke(&mut self, c: [usize; 3], axes: (usize, usize, usize), vs: [f64; 3], erase: bool) {
        let Some(mi) = self.target_mask(true) else { return };
        let label = if erase { 0.0 } else { self.brush.label.max(1) as f32 };
        let r = self.brush.radius_mm as f64;
        let new_stroke = self.brush_last.is_none();
        let spacing = (r / vs[axes.0].max(1e-9) * 0.5).max(1.0);
        let pts = match self.brush_last {
            Some(l) => region::segment_points(l, c, spacing),
            None => vec![c],
        };
        let dims = self.layers[mi].vol.dims;
        let mut idx = Vec::new();
        for p in pts {
            idx.extend(region::brush_indices(dims, vs, p, axes, r, self.brush.three_d));
        }
        self.layers[mi].paint_stroke(&idx, label, new_stroke);
        self.brush_last = Some(c);
        self.error = None;
    }

    fn start_registration(&mut self, i: usize) {
        if self.reg_job.is_some() {
            return;
        }
        let Some(g) = self.grid.clone() else { return };
        let (Some(fixed), Some(l)) = (self.layers.first(), self.layers.get(i)) else { return };
        let Some(moving) = l.orig.clone() else {
            self.error = Some("the first image is the reference; select another image to register".into());
            return;
        };
        let prob = register::Problem { fixed: fixed.vol.clone(), moving, base: l.base, start: l.xf, pivot: g.center() };
        let opts = self.reg_ui.opts;
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let c2 = cancel.clone();
        std::thread::spawn(move || register::run(prob, opts, &c2, &|m| drop(tx.send(m))));
        self.reg_job = Some(RegJob { layer: i, name: l.name.clone(), rx, cancel });
        self.reg_ui.running = Some(i);
        self.reg_ui.status = "registering...".into();
        self.error = None;
    }

    /// Collect progress from the registration thread and apply the transform found so far.
    fn poll_registration(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.reg_job else { return };
        let (layer, name) = (job.layer, job.name.clone());
        if std::mem::take(&mut self.reg_ui.cancel) {
            job.cancel.store(true, Ordering::Relaxed);
        }
        let (mut update, mut finished, mut done) = (None, false, false);
        loop {
            match job.rx.try_recv() {
                Ok(register::Msg::Progress { text, params }) => {
                    self.reg_ui.status = text;
                    update = Some(params);
                }
                Ok(register::Msg::Done { text, params }) => {
                    self.notice = Some(text);
                    update = Some(params);
                    (finished, done) = (true, true);
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }
        if let (Some(params), Some(g)) = (update, self.grid.clone()) {
            if let Some(l) = self.layers.get_mut(layer).filter(|l| l.name == name) {
                l.set_transform(params, &g);
            }
        }
        if finished {
            if !done {
                self.error = Some("registration stopped unexpectedly".into());
            }
            self.reg_job = None;
            self.reg_ui.running = None;
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }

    fn region_key(&self, sa: usize) -> (region::Params, bool, usize) {
        let mut p = self.region.clone();
        p.slice_axis = self.region_in_slice.then_some(sa);
        (p, self.region_in_slice, self.region_source().unwrap_or(0))
    }

    /// Select (in yellow) the region grown from the cursor voxel.
    fn select_at(&mut self, sa: usize) {
        self.pending = Some(Pending { seed: self.cursor, sa, idx: Vec::new(), mask: Vec::new(), key: self.region_key(sa) });
        self.recompute_pending(true);
    }

    /// Recompute the selection if the settings changed (live preview) or `force`d.
    fn recompute_pending(&mut self, force: bool) {
        let Some(p) = &self.pending else { return };
        let Some(si) = self.region_source() else {
            self.pending = None;
            return;
        };
        let key = self.region_key(p.sa);
        if !force && key == p.key {
            return;
        }
        let vol = &self.layers[si].vol;
        let idx = region::grow(vol, p.seed, &key.0);
        let mut mask = vec![false; vol.data.len()];
        for &i in &idx {
            mask[i] = true;
        }
        let p = self.pending.as_mut().unwrap();
        p.idx = idx;
        p.mask = mask;
        p.key = key;
    }

    /// Add the yellow selection to the mask as a new parcel (its own label), or with
    /// `subtract` remove it from the mask.
    fn commit(&mut self, subtract: bool) {
        let Some(p) = self.pending.take_if(|p| !p.idx.is_empty()) else {
            self.error = Some("nothing selected: Ctrl+click a voxel first".into());
            return;
        };
        let Some(mi) = self.target_mask(true) else { return };
        let label = if subtract { 0.0 } else { self.layers[mi].next_label() };
        let n = self.layers[mi].paint(&p.idx, label);
        let vs = self.layers[mi].vol.voxel_size();
        let mm3 = n as f64 * vs[0] * vs[1] * vs[2];
        self.error = None;
        self.notice = Some(if subtract {
            format!("removed {n} voxels ({mm3:.0} mm3) from '{}'", self.layers[mi].name)
        } else {
            format!("added parcel {label} to '{}': {n} voxels ({mm3:.0} mm3)", self.layers[mi].name)
        });
    }

    fn region_panel(&mut self, ui: &mut egui::Ui) {
        let Some(grid) = self.grid.clone() else { return };
        egui::CollapsingHeader::new("Masks & region growing").default_open(true).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tool, Tool::Navigate, "Navigate");
                ui.selectable_value(&mut self.tool, Tool::Region, "Grow region");
                ui.selectable_value(&mut self.tool, Tool::Brush, "Brush");
            });
            if self.tool == Tool::Brush {
                ui.horizontal(|ui| {
                    ui.label("Radius");
                    ui.add(egui::DragValue::new(&mut self.brush.radius_mm).speed(0.1).range(0.0..=50.0).suffix(" mm"))
                        .on_hover_text("Also [ and ] on the keyboard");
                    ui.checkbox(&mut self.brush.three_d, "3D").on_hover_text("Paint a ball through neighbouring slices instead of a flat disk");
                });
                ui.horizontal(|ui| {
                    ui.label("Label");
                    ui.add(egui::DragValue::new(&mut self.brush.label).range(1..=65535));
                    if ui.small_button("New").on_hover_text("Use the next unused label").clicked() {
                        let next = self.target_mask(false).map_or(1.0, |m| self.layers[m].next_label());
                        self.brush.label = next as u32;
                    }
                    ui.checkbox(&mut self.brush.erase, "Eraser");
                });
                ui.label(egui::RichText::new("Drag to paint; hold Shift to erase. Ctrl+click still selects a region.").small().weak());
            }
            let src = self.region_source();
            if self.tool == Tool::Region && !self.range_set {
                if let Some(si) = src {
                    let wl = self.layers[si].wl;
                    (self.region.lo, self.region.hi) = (wl.lo, wl.hi);
                    self.range_set = true;
                }
            }
            ui.horizontal_wrapped(|ui| {
                if ui.button("New mask").clicked() {
                    self.mask_count += 1;
                    self.layers.push(Layer::mask(format!("mask{}", self.mask_count), &grid));
                    self.selected = self.layers.len() - 1;
                    self.tool = Tool::Region;
                }
                let sel_mask = self.layers.get(self.selected).is_some_and(|l| l.is_mask);
                ui.add_enabled_ui(sel_mask, |ui| {
                    if ui.button("Undo").on_hover_text("Undo the last add/remove on the selected mask").clicked() {
                        if let Some(l) = self.layers.get_mut(self.selected) {
                            if !l.undo_paint() {
                                self.notice = Some("nothing to undo".into());
                            }
                        }
                    }
                    if ui.button("Clear").on_hover_text("Empty the selected mask (undoable)").clicked() {
                        if let Some(l) = self.layers.get_mut(self.selected) {
                            let all: Vec<usize> = (0..l.vol.data.len()).filter(|&i| l.vol.data[i] != 0.0).collect();
                            l.paint(&all, 0.0);
                        }
                    }
                    if ui.button("Save mask...").clicked() {
                        if let Some(l) = self.layers.get(self.selected) {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("NIfTI", &["nii", "gz"])
                                .set_file_name(format!("{}.nii.gz", l.name))
                                .save_file()
                            {
                                match l.save_mask(&path) {
                                    Ok(m) => {
                                        self.notice = Some(m);
                                        self.error = None;
                                    }
                                    Err(e) => self.error = Some(e),
                                }
                            }
                        }
                    }
                });
            });
            match self.layers.get(self.selected).filter(|l| l.is_mask) {
                Some(l) => {
                    ui.label(format!("Painting into: {} ({} voxels)", l.name, l.mask_voxels()));
                }
                None => {
                    ui.label(egui::RichText::new("Select or create a mask layer to paint into").weak());
                }
            }
            // Selection status and actions.
            let (n, seed) = self.pending.as_ref().map_or((0, None), |p| (p.idx.len(), Some(p.seed)));
            match (&self.pending, src) {
                (Some(_), Some(si)) if n == 0 => {
                    let [i, j, k] = seed.unwrap();
                    ui.label(format!("Selection empty: seed value {:.3} is outside the criteria", self.layers[si].vol.get(i, j, k)));
                }
                (Some(_), _) => {
                    let vs = grid.affine;
                    let vox = (0..3).map(|c| (vs[0][c].powi(2) + vs[1][c].powi(2) + vs[2][c].powi(2)).sqrt()).product::<f64>();
                    ui.label(format!("Selected (yellow): {n} voxels, {:.0} mm3", n as f64 * vox));
                }
                _ => {
                    ui.label(egui::RichText::new("Ctrl+click a voxel to select a region").weak());
                }
            }
            ui.horizontal_wrapped(|ui| {
                ui.add_enabled_ui(n > 0, |ui| {
                    if ui.button("Add region (Enter)").on_hover_text("Add the yellow selection to the mask as a new parcel").clicked() {
                        self.commit(false);
                    }
                    if ui.button("Remove from mask").clicked() {
                        self.commit(true);
                    }
                });
                if ui.add_enabled(self.pending.is_some(), egui::Button::new("Cancel (Esc)")).clicked() {
                    self.pending = None;
                }
            });
            ui.separator();
            let names: Vec<(usize, String)> =
                self.layers.iter().enumerate().filter(|(_, l)| !l.is_mask).map(|(i, l)| (i, l.name.clone())).collect();
            if let Some(cur) = src {
                egui::ComboBox::from_id_salt("region_src")
                    .selected_text(format!("Grow on: {}", self.layers[cur].name))
                    .show_ui(ui, |ui| {
                        for (i, n) in &names {
                            if ui.selectable_label(*i == cur, n).clicked() {
                                self.region_src = *i;
                            }
                        }
                    });
            }
            let r = &mut self.region;
            egui::ComboBox::from_id_salt("region_method").selected_text(r.method.name()).show_ui(ui, |ui| {
                for m in Method::ALL {
                    ui.selectable_value(&mut r.method, m, m.name()).on_hover_text(m.help());
                }
            });
            ui.label(egui::RichText::new(r.method.help()).small().weak());
            let speed = src.map(|s| ((self.layers[s].vol.max - self.layers[s].vol.min).abs() / 300.0).max(1e-4) as f64).unwrap_or(1.0);
            match r.method {
                Method::ConnectedThreshold => {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut r.range_mode, RangeMode::RelativeToSeed, "Relative to seed");
                        ui.selectable_value(&mut r.range_mode, RangeMode::Absolute, "Absolute range");
                    });
                    match r.range_mode {
                        RangeMode::RelativeToSeed => {
                            ui.horizontal(|ui| {
                                let (step, max) = match r.offset_unit {
                                    OffsetUnit::Percent => (0.25, 1000.0),
                                    OffsetUnit::Intensity => (speed, f32::MAX),
                                };
                                ui.add(egui::DragValue::new(&mut r.minus).speed(step).range(0.0..=max).prefix("-"));
                                ui.add(egui::DragValue::new(&mut r.plus).speed(step).range(0.0..=max).prefix("+"));
                                egui::ComboBox::from_id_salt("offset_unit").selected_text(r.offset_unit.name()).show_ui(ui, |ui| {
                                    for u in [OffsetUnit::Percent, OffsetUnit::Intensity] {
                                        ui.selectable_value(&mut r.offset_unit, u, u.name());
                                    }
                                });
                            });
                            if let (Some(p), Some(si)) = (&self.pending, src) {
                                let [i, j, k] = p.seed;
                                let sv = self.layers[si].vol.get(i, j, k);
                                let (lo, hi) = region::threshold_range(r, sv);
                                ui.label(egui::RichText::new(format!("seed {sv:.3}: accepts [{lo:.3}, {hi:.3}]")).small().weak());
                            }
                        }
                        RangeMode::Absolute => {
                            ui.horizontal(|ui| {
                                ui.label("min");
                                ui.add(egui::DragValue::new(&mut r.lo).speed(speed));
                                ui.label("max");
                                ui.add(egui::DragValue::new(&mut r.hi).speed(speed));
                            });
                            if let Some(si) = src {
                                if ui.small_button("Range from window").clicked() {
                                    (r.lo, r.hi) = (self.layers[si].wl.lo, self.layers[si].wl.hi);
                                }
                            }
                        }
                    }
                }
                Method::SeedTolerance => {
                    ui.horizontal(|ui| {
                        ui.label("tolerance +/-");
                        ui.add(egui::DragValue::new(&mut r.tolerance).speed(speed).range(0.0..=f32::MAX));
                    });
                }
                Method::ConfidenceConnected => {
                    ui.horizontal(|ui| {
                        ui.label("k (SD)");
                        ui.add(egui::DragValue::new(&mut r.k).speed(0.05).range(0.1..=10.0));
                        ui.label("iterations");
                        ui.add(egui::DragValue::new(&mut r.iterations).range(1..=50));
                    });
                }
                Method::Ball => {
                    ui.horizontal(|ui| {
                        ui.label("radius");
                        ui.add(egui::DragValue::new(&mut r.radius_mm).speed(0.25).range(0.0..=200.0).suffix(" mm"));
                    });
                }
            }
            if r.method != Method::Ball {
                egui::ComboBox::from_id_salt("region_conn")
                    .selected_text(format!("Connectivity: {}", r.connectivity.name()))
                    .show_ui(ui, |ui| {
                        for c in Connectivity::ALL {
                            ui.selectable_value(&mut r.connectivity, c, c.name());
                        }
                    });
            }
            ui.checkbox(&mut self.region_in_slice, "Only in the clicked slice (2D)");
            ui.label(egui::RichText::new("Ctrl+click or right-click selects a region; changing tolerance, connectivity etc. updates it live. Enter, the button or the right-click menu adds it to the mask as a new parcel.").small().weak());
        });
        ui.separator();
    }

    /// Colormap, opacity and window controls for one layer.
    fn layer_controls(
        ui: &mut egui::Ui,
        l: &mut Layer,
        i: usize,
        grid: &Option<Grid>,
        reference: &str,
        reg: &mut RegUi,
    ) -> Option<Result<String, String>> {
        let mut msg = None;
        egui::ComboBox::from_id_salt(("cmap", i))
            .selected_text(l.colormap.name())
            .show_ui(ui, |ui| {
                for cm in Colormap::ALL {
                    if ui.selectable_label(l.colormap == cm, cm.name()).clicked() {
                        if let Some(g) = grid {
                            l.set_colormap(cm, g);
                        }
                    }
                }
            });
        ui.add(egui::Slider::new(&mut l.opacity, 0.0..=1.0).text("opacity"));
        if l.colormap != Colormap::Label {
            let speed = ((l.vol.max - l.vol.min).abs() / 300.0).max(1e-4) as f64;
            ui.horizontal(|ui| {
                ui.label("min");
                ui.add(egui::DragValue::new(&mut l.wl.lo).speed(speed));
                ui.label("max");
                ui.add(egui::DragValue::new(&mut l.wl.hi).speed(speed));
            });
            if l.wl.hi <= l.wl.lo {
                l.wl.hi = l.wl.lo + 1e-3;
            }
            ui.checkbox(&mut l.clear_below, "transparent below min");
            if ui.small_button("auto window").clicked() {
                l.wl = WindowLevel::auto(&l.vol);
            }
        }
        // Manual alignment (every layer except the reference, which has no original to resample).
        if let (true, Some(g)) = (l.orig.is_some(), grid) {
            egui::CollapsingHeader::new("Align to first image").id_salt(("align", i)).show(ui, |ui| {
                let mut xf = l.xf;
                let (mut apply_now, mut discard) = (false, false);
                egui::Grid::new(("align_grid", i)).num_columns(4).show(ui, |ui| {
                    let row = |ui: &mut egui::Ui, name: &str, v: &mut [f64; 3], speed: f64, unit: &str| {
                        ui.label(name);
                        for c in v.iter_mut() {
                            ui.add(egui::DragValue::new(c).speed(speed).suffix(unit));
                        }
                        ui.end_row();
                    };
                    row(ui, "Translate", &mut xf.translate, 0.25, " mm");
                    row(ui, "Rotate", &mut xf.rotate, 0.25, " deg");
                    row(ui, "Scale", &mut xf.scale, 0.005, "");
                    row(ui, "Shear xy xz yz", &mut xf.shear, 0.005, "");
                });
                for sc in &mut xf.scale {
                    *sc = sc.clamp(0.05, 20.0);
                }
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .small_button("Apply alignment")
                        .on_hover_text("Lock in the current alignment as the new starting point. The fields reset to zero; later changes and saved transforms are relative to it.")
                        .clicked()
                    {
                        apply_now = true;
                    }
                    if ui.small_button("Reset adjustment").on_hover_text("Zero the fields (back to the last applied alignment)").clicked() {
                        xf = TransformParams::default();
                    }
                    if !layers::affine_is_identity(&l.base)
                        && ui.small_button("Discard applied alignment").on_hover_text("Back to the image's original position").clicked()
                    {
                        discard = true;
                    }
                });
                l.set_transform(xf, g);
                if apply_now {
                    l.apply_alignment(g);
                }
                if discard {
                    l.discard_alignment(g);
                }
                ui.separator();
                ui.strong("Auto-register to first image");
                match reg.running {
                    Some(r) if r == i => {
                        ui.label(&reg.status);
                        if ui.button("Cancel registration").clicked() {
                            reg.cancel = true;
                        }
                    }
                    Some(_) => {
                        ui.label(egui::RichText::new("another image is being registered...").weak());
                    }
                    None => {
                        egui::ComboBox::from_id_salt(("reg_metric", i)).selected_text(reg.opts.metric.name()).show_ui(ui, |ui| {
                            for m in [register::Metric::Mi, register::Metric::Ncc] {
                                ui.selectable_value(&mut reg.opts.metric, m, m.name());
                            }
                        });
                        egui::ComboBox::from_id_salt(("reg_mode", i)).selected_text(reg.opts.mode.name()).show_ui(ui, |ui| {
                            for m in [register::Mode::Affine, register::Mode::Rigid] {
                                ui.selectable_value(&mut reg.opts.mode, m, m.name());
                            }
                        });
                        ui.checkbox(&mut reg.opts.init_com, "Start by matching centres of mass");
                        ui.checkbox(&mut reg.opts.multi_start, "Try several orientations (large rotations)");
                        if ui
                            .button("Register (affine)")
                            .on_hover_text("Find the transform that best matches this image to the first image. The result lands in the fields above; use 'Apply alignment' or the save buttons afterwards.")
                            .clicked()
                        {
                            reg.request = Some(i);
                        }
                    }
                }
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    let stem = |n: &str| n.trim_end_matches(".gz").trim_end_matches(".nii").to_string();
                    if ui.small_button("Save transform...").on_hover_text("Write the current adjustment (relative to the applied alignment) to a file").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("transform", &["json"])
                            .set_file_name(format!("{}_to_{}.json", stem(&l.name), stem(reference)))
                            .save_file()
                        {
                            msg = Some(layers::save_transform_file(&path, l, g, reference, false));
                        }
                    }
                    if !layers::affine_is_identity(&l.base)
                        && ui
                            .small_button("Save total...")
                            .on_hover_text("Save the full movement from the original position (applied alignment + adjustment)")
                            .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("transform", &["json"])
                            .set_file_name(format!("{}_to_{}_total.json", stem(&l.name), stem(reference)))
                            .save_file()
                        {
                            msg = Some(layers::save_transform_file(&path, l, g, reference, true));
                        }
                    }
                    if ui.small_button("Load transform...").on_hover_text("Apply a saved transform to this image").clicked() {
                        if let Some(path) = rfd::FileDialog::new().add_filter("transform", &["json"]).pick_file() {
                            msg = Some(layers::load_transform_file(&path, l, g, false));
                        }
                    }
                    if ui
                        .small_button("Load reverse transform...")
                        .on_hover_text("Apply the inverse of a saved transform to this image")
                        .clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new().add_filter("transform", &["json"]).pick_file() {
                            msg = Some(layers::load_transform_file(&path, l, g, true));
                        }
                    }
                });
            });
        }
        msg
    }

    fn layer_panel(&mut self, ui: &mut egui::Ui) {
        let grid = self.grid.clone();
        let mut action = None;
        let mut select = None;
        let mut msg: Option<Result<String, String>> = None;
        let ref_name = self.layers.first().map(|l| l.name.clone()).unwrap_or_default();

        // Mode "selected image only": one control panel at the bottom of the sidebar.
        if !self.controls_each {
            let sel = self.selected;
            if let Some(l) = self.layers.get_mut(sel) {
                egui::Panel::bottom("selected_controls").show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.strong(format!("Controls: {}", l.name));
                    if let Some(r) = Self::layer_controls(ui, l, sel, &grid, &ref_name, &mut self.reg_ui) {
                        msg = Some(r);
                    }
                    ui.add_space(4.0);
                });
            }
        }

        egui::ScrollArea::vertical().id_salt("region_scroll").max_height(ui.available_height() * 0.55).show(ui, |ui| {
            self.region_panel(ui);
        });
        ui.heading("Layers");
        ui.separator();
        egui::ScrollArea::vertical().show(ui, |ui| {
            // Top layer first.
            for i in (0..self.layers.len()).rev() {
                let selected = i == self.selected;
                let each = self.controls_each;
                let l = &mut self.layers[i];
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut l.visible, "");
                        if ui.selectable_label(selected, &l.name).clicked() {
                            select = Some(i);
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("x").on_hover_text("Remove layer").clicked() {
                                action = Some(LayerAction::Remove(i));
                            }
                        });
                    });
                    if let Some(n) = &l.note {
                        ui.label(egui::RichText::new(format!("({n})")).small().weak());
                    }
                    if each {
                        if let Some(r) = Self::layer_controls(ui, l, i, &grid, &ref_name, &mut self.reg_ui) {
                            msg = Some(r);
                        }
                    }
                });
            }
        });
        if let Some(i) = self.reg_ui.request.take() {
            self.start_registration(i);
        }
        if let Some(i) = select {
            self.selected = i;
        }
        if let Some(a) = action {
            self.apply(a);
        }
        match msg {
            Some(Ok(n)) => {
                self.notice = Some(n);
                self.error = None;
            }
            Some(Err(e)) => self.error = Some(e),
            None => {}
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.draw(ui);
    }
}

impl App {
    fn draw(&mut self, ui: &mut egui::Ui) {
        // drag & drop (several files allowed)
        let dropped: Vec<std::path::PathBuf> =
            ui.ctx().input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        for p in dropped {
            self.add_path(&p);
        }

        if self.grid.is_none() {
            self.pending = None;
        }
        let ctx = ui.ctx().clone();
        self.poll_registration(&ctx);
        if !ui.ctx().egui_wants_keyboard_input() && self.pending.is_some() {
            if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.commit(false);
            } else if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.pending = None;
            }
        }

        if self.tool == Tool::Brush && !ui.ctx().egui_wants_keyboard_input() {
            let (down, up) = ui.input(|i| (i.key_pressed(egui::Key::OpenBracket), i.key_pressed(egui::Key::CloseBracket)));
            if down {
                self.brush.radius_mm = (self.brush.radius_mm - 0.5).max(0.0);
            }
            if up {
                self.brush.radius_mm = (self.brush.radius_mm + 0.5).min(50.0);
            }
        }

        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open…").clicked() {
                    if let Some(ps) = rfd::FileDialog::new().add_filter("NIfTI", &["nii", "gz"]).pick_files() {
                        for p in ps {
                            self.add_path(&p);
                        }
                    }
                }
                ui.separator();
                ui.selectable_value(&mut self.layout, Layout::OneByThree, "1x3");
                ui.selectable_value(&mut self.layout, Layout::TwoByTwo, "2x2");
                for p in Plane::ALL {
                    ui.selectable_value(&mut self.layout, Layout::Single(p), p.name());
                }
                ui.separator();
                ui.checkbox(&mut self.radiological, "Radiological (R on screen-left)");
                ui.checkbox(&mut self.controls_each, "Controls for each image");
                ui.checkbox(&mut self.deoblique, "Straighten oblique")
                    .on_hover_text("When loading the first image: resample a tilted (oblique) acquisition onto world axes instead of showing its own voxel grid. Applies to the next image you load as the first one.");
                if ui.button("Reset view").clicked() {
                    for p in &mut self.panes {
                        p.zoom = 1.0;
                        p.pan = [0.0, 0.0];
                    }
                }
            });
        });

        egui::Panel::bottom("status").show(ui, |ui| {
            if let Some(g) = &self.grid {
                let [i, j, k] = self.cursor;
                let w = volume::apply(&g.affine, [i as f64, j as f64, k as f64]);
                let mut s = format!(
                    "voxel [{i}, {j}, {k}]   world (RAS) [{:.1}, {:.1}, {:.1}] mm  |",
                    w[0], w[1], w[2]
                );
                for l in self.layers.iter().rev() {
                    s += &format!("  {}: {:.3}", l.name, l.vol.get(i, j, k));
                }
                if let Some(n) = &self.notice {
                    s += &format!("   - {n}");
                }
                if let Some(e) = &self.error {
                    s += &format!("   [!] {e}");
                }
                ui.label(s);
            } else {
                ui.label(self.error.clone().unwrap_or_else(|| "Open or drag .nii / .nii.gz files onto the window.".into()));
            }
        });

        egui::Panel::left("layers").default_size(270.0).show(ui, |ui| {
            self.layer_panel(ui);
        });

        self.recompute_pending(false);

        egui::CentralPanel::default().show(ui, |ui| {
            if self.grid.is_none() {
                ui.centered_and_justified(|ui| ui.heading("Drop a NIfTI file here"));
                return;
            }
            let r = ui.available_rect_before_wrap();
            let g = 2.0;
            let cell = |col: usize, row: usize, cols: usize, rows: usize| {
                let w = (r.width() - g * (cols as f32 - 1.0)) / cols as f32;
                let h = (r.height() - g * (rows as f32 - 1.0)) / rows as f32;
                Rect::from_min_size(
                    Pos2::new(r.min.x + col as f32 * (w + g), r.min.y + row as f32 * (h + g)),
                    Vec2::new(w, h),
                )
            };
            match self.layout {
                Layout::OneByThree => {
                    for (n, p) in Plane::ALL.into_iter().enumerate() {
                        self.pane_ui(ui, n, p, cell(n, 0, 3, 1));
                    }
                }
                Layout::TwoByTwo => {
                    self.pane_ui(ui, 0, Plane::Axial, cell(0, 0, 2, 2));
                    self.pane_ui(ui, 1, Plane::Coronal, cell(1, 0, 2, 2));
                    self.pane_ui(ui, 2, Plane::Sagittal, cell(0, 1, 2, 2));
                    let rr = cell(1, 1, 2, 2);
                    ui.painter().rect_filled(rr, 0.0, Color32::from_gray(20));
                    ui.painter().text(rr.center(), egui::Align2::CENTER_CENTER, "3D view (milestone 4)", egui::FontId::proportional(14.0), Color32::from_gray(120));
                }
                Layout::Single(p) => {
                    let n = Plane::ALL.iter().position(|q| *q == p).unwrap();
                    self.pane_ui(ui, n, p, cell(0, 0, 1, 1));
                }
            }
        });
    }
}

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1300.0, 800.0]).with_title("PODO_niftis"),
        ..Default::default()
    };
    eframe::run_native("PODO_niftis", opts, Box::new(|_cc| Ok(Box::new(App::new()))))
}


#[cfg(test)]
mod ui_tests {
    use super::*;
    use egui_kittest::kittest::Queryable;
    use egui_kittest::Harness;

    /// Real files: every loaded image stays listed; removing one keeps the other and the viewer.
    #[test]
    fn layers_listed_and_removable() {
        let mut a = App::empty();
        a.add_layer(layers::parse_arg(&testdata::head_file("head_ref.nii.gz")).unwrap());
        let overlay = format!("{}:colormap=hot:opacity=0.6", testdata::head_file("head_overlay.nii.gz"));
        a.add_layer(layers::parse_arg(&overlay).unwrap());
        assert_eq!(a.layers.len(), 2, "error: {:?}", a.error);
        let mut h = Harness::builder()
            .with_size(egui::vec2(1300.0, 800.0))
            .with_accessibility_check(false)
            .build_ui_state(|ui, a: &mut App| a.draw(ui), a);
        h.run();
        h.get_by_label("head_ref.nii.gz");
        h.get_by_label("head_overlay.nii.gz");
        h.get_all_by_label("x").next().unwrap().click(); // top row = activation
        h.run();
        assert_eq!(h.state().layers.len(), 1);
        assert!(h.state().grid.is_some());
        h.get_by_label("head_ref.nii.gz");
    }
}

#[cfg(test)]
mod mode_tests {
    use super::*;
    use egui_kittest::kittest::Queryable;
    use egui_kittest::Harness;

    fn app() -> App {
        let vol = volume::Volume {
            dims: [8, 8, 8],
            data: (0..512).map(|v| v as f32).collect(),
            affine: volume::identity(),
            min: 0.0,
            max: 511.0,
        };
        let grid = Grid::of(&vol);
        let mut a = App::empty();
        a.grid = Some(grid.clone());
        a.cursor = [4, 4, 4];
        for n in ["first.nii.gz", "second.nii.gz", "third.nii.gz"] {
            a.layers.push(Layer::new(n.into(), vol.clone(), &grid, Colormap::Gray));
        }
        a.selected = 2;
        a
    }

    /// Each opacity slider appears twice in the widget tree (slider + its number field).
    const PER_SLIDER: usize = 2;

    fn harness() -> Harness<'static, App> {
        Harness::builder()
            .with_size(egui::vec2(1300.0, 900.0))
            .with_accessibility_check(false)
            .build_ui_state(|ui, a: &mut App| a.draw(ui), app())
    }

    #[test]
    fn selected_mode_has_one_control_panel_and_lists_all() {
        let mut h = harness();
        h.run();
        for n in ["first.nii.gz", "second.nii.gz", "third.nii.gz"] {
            h.get_by_label(n);
        }
        assert_eq!(h.get_all_by_label("opacity").count(), PER_SLIDER);
        // Selecting another image moves the single panel to it.
        h.get_by_label("first.nii.gz").click();
        h.run();
        assert_eq!(h.state().selected, 0);
        assert_eq!(h.get_all_by_label("opacity").count(), PER_SLIDER);
        h.get_by_label("Controls: first.nii.gz");
    }

    #[test]
    fn each_mode_has_controls_for_every_image() {
        let mut h = harness();
        h.get_by_label("Controls for each image").click();
        h.run();
        assert!(h.state().controls_each);
        assert_eq!(h.get_all_by_label("opacity").count(), 3 * PER_SLIDER);
        for n in ["first.nii.gz", "second.nii.gz", "third.nii.gz"] {
            h.get_by_label(n);
        }
        // and back
        h.get_by_label("Controls for each image").click();
        h.run();
        assert_eq!(h.get_all_by_label("opacity").count(), PER_SLIDER);
    }
}

#[cfg(test)]
mod align_tests {
    use super::*;
    use egui_kittest::kittest::Queryable;
    use egui_kittest::Harness;

    #[test]
    fn alignment_section_only_for_non_reference_and_resamples_fast() {
        let mut a = App::empty();
        a.add_layer(layers::parse_arg(&testdata::head_file("head_ref.nii.gz")).unwrap());
        a.add_layer(layers::parse_arg(&testdata::head_file("head_ref.nii.gz")).unwrap());
        assert_eq!(a.layers.len(), 2, "error: {:?}", a.error);
        assert!(a.layers[0].orig.is_none(), "reference layer must not carry a copy");
        assert!(a.layers[1].orig.is_some());

        // Time a full-size re-resample with a rotation (the cost paid on every drag step).
        let grid = a.grid.clone().unwrap();
        let xf = TransformParams { rotate: [3.0, -2.0, 5.0], translate: [4.0, 0.0, -2.0], ..Default::default() };
        let t = std::time::Instant::now();
        a.layers[1].set_transform(xf, &grid);
        println!("full-size resample: {:?}", t.elapsed());
        assert_eq!(a.layers[1].note.as_deref(), Some("aligned manually"));

        let mut h = Harness::builder()
            .with_size(egui::vec2(1300.0, 1000.0))
            .with_accessibility_check(false)
            .build_ui_state(|ui, a: &mut App| a.draw(ui), a);
        h.run();
        // second image is selected (last loaded): its alignment header is shown (open by default? no)
        h.get_by_label("Align to first image").click();
        h.run();
        for b in ["Save transform...", "Load transform...", "Load reverse transform...", "Apply alignment", "Reset adjustment"] {
            h.get_by_label(b);
        }
        // selecting the reference hides it
        h.get_all_by_label("head_ref.nii.gz").last().unwrap().click();
        h.run();
        assert_eq!(h.state().selected, 0);
        assert!(h.query_by_label("Align to first image").is_none());
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;

    fn app() -> App {
        let mut a = App::empty();
        a.add_layer(layers::parse_arg(&testdata::head_file("head_ref.nii.gz")).unwrap());
        let g = a.grid.clone().unwrap();
        a.layers.push(Layer::mask("m".into(), &g));
        a.selected = 1;
        a
    }

    #[test]
    fn select_is_live_commit_makes_new_parcels_and_undo_works() {
        let mut a = app();
        a.region.method = Method::Ball;
        a.region.radius_mm = 4.0;
        a.cursor = [20, 50, 35];
        a.select_at(2);
        let n1 = a.pending.as_ref().unwrap().idx.len();
        assert!(n1 > 0);
        assert_eq!(a.layers[1].mask_voxels(), 0, "selection must not touch the mask yet");
        // Live: changing the radius recomputes the selection.
        a.region.radius_mm = 8.0;
        a.recompute_pending(false);
        assert!(a.pending.as_ref().unwrap().idx.len() > n1);
        a.region.radius_mm = 4.0;
        a.recompute_pending(false);
        a.commit(false);
        assert!(a.pending.is_none());
        assert_eq!(a.layers[1].mask_voxels(), n1);
        // Second area becomes parcel 2.
        a.cursor = [60, 50, 35];
        a.select_at(2);
        a.commit(false);
        assert_eq!(a.layers[1].mask_voxels(), n1 * 2);
        assert_eq!(a.layers[1].vol.get(20, 50, 35), 1.0);
        assert_eq!(a.layers[1].vol.get(60, 50, 35), 2.0);
        // Remove the second one again, then undo.
        a.select_at(2);
        a.commit(true);
        assert_eq!(a.layers[1].mask_voxels(), n1);
        assert!(a.layers[1].undo_paint());
        assert_eq!(a.layers[1].mask_voxels(), n1 * 2);
    }

    #[test]
    fn mask_is_created_on_demand_and_seed_tolerance_works_on_real_data() {
        let mut a = App::empty();
        a.add_layer(layers::parse_arg(&testdata::head_file("head_ref.nii.gz")).unwrap());
        a.cursor = [40, 50, 35];
        a.region.method = Method::SeedTolerance;
        a.region.tolerance = (a.layers[0].vol.max * 0.05).max(1.0);
        a.select_at(2);
        a.commit(false);
        assert_eq!(a.layers.len(), 2);
        assert!(a.layers[1].is_mask && a.layers[1].mask_voxels() > 0, "{:?}", a.error);
    }

    #[test]
    fn mask_saves_and_reloads() {
        let mut a = app();
        a.region.method = Method::Ball;
        a.cursor = [40, 50, 35];
        a.select_at(2);
        a.commit(false);
        let path = std::env::temp_dir().join("PODO_niftis_mask_test.nii.gz");
        a.layers[1].save_mask(&path).unwrap();
        let back = volume::load(&path).unwrap();
        assert_eq!(back.dims, a.layers[1].vol.dims);
        assert_eq!(back.data.iter().filter(|&&v| v != 0.0).count(), a.layers[1].mask_voxels());
        assert_eq!(back.get(40, 50, 35), 1.0);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn brush_strokes_paint_erase_and_undo_as_one_step() {
        let mut a = app();
        a.brush.radius_mm = 2.0;
        let vs = [2.0, 2.0, 2.0];
        let axes = (0, 1, 2);
        // one stroke along x: continuous, no gaps
        a.brush_stroke([15, 50, 35], axes, vs, false);
        a.brush_stroke([35, 50, 35], axes, vs, false);
        a.brush_last = None;
        assert!((15..=35).all(|x| a.layers[1].vol.get(x, 50, 35) == 1.0));
        let painted = a.layers[1].mask_voxels();
        assert!(painted > 21);
        // second stroke under another label, then erase part of the first
        a.brush.label = 2;
        a.brush_stroke([25, 60, 35], axes, vs, false);
        a.brush_last = None;
        assert_eq!(a.layers[1].vol.get(25, 60, 35), 2.0);
        a.brush_stroke([25, 50, 35], axes, vs, true);
        a.brush_last = None;
        assert_eq!(a.layers[1].vol.get(25, 50, 35), 0.0);
        // each stroke is one undo step
        assert!(a.layers[1].undo_paint());
        assert_eq!(a.layers[1].vol.get(25, 50, 35), 1.0);
        assert!(a.layers[1].undo_paint());
        assert_eq!(a.layers[1].vol.get(25, 60, 35), 0.0);
        assert!(a.layers[1].undo_paint());
        assert_eq!(a.layers[1].mask_voxels(), 0);
        assert!(!a.layers[1].undo_paint());
    }
}
