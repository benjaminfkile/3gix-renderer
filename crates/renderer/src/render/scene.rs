//! Camera-relative scene data: the only place world positions become `f32`.
//!
//! Implements the last sentence of the floating origin device in
//! `space-model.md` section 5: "The GPU only ever sees camera-relative
//! single-precision coordinates." Every position here is computed in `f64`
//! with [`FrameSystem::relative`] from the camera position in its frame, and
//! only the resulting small vector is cast to `f32`.
//!
//! Positions are in root axes, relative to the camera. The rotation from
//! root axes into camera axes is handed over separately as
//! [`Scene::view_rotation`], so later passes can rotate larger vertex sets on
//! the GPU.
//!
//! # Depth
//!
//! The projection is reversed-z with an infinite far plane in `f32` depth:
//! depth is `near / distance`, 1 at the near plane falling toward 0 at
//! infinity, the depth buffer clears to 0, and the depth test keeps the
//! greater value. With the infinite far plane the only plane to choose is the
//! near one, which is set per frame from the nearest visible marker (see
//! [`near_plane`]) and pulled in to the nearest drawn mesh (see
//! [`add_matter`]), so precision is spent where the nearest matter is.
//!
//! # Draw order
//!
//! Opaque meshes first, writing depth; then volumes sorted back to front by
//! the camera distance to the center of their gas box, tested against the
//! mesh depth and writing none; then point sprites of far frames, added on
//! top and tested against the mesh depth (`docs/shading.md`).

use crate::camera::{Camera, FIELD_OF_VIEW_Y};
use crate::extract::SurfaceMesh;
use crate::farfield::Sprite;
use crate::light::PointLight;
use crate::volume::VolumeGrid;
use crate::world::{DrawList, World};
use gx_core::frames::FrameSystem;
use gx_core::units::Vec3;
use std::sync::Arc;

/// The near plane never moves closer than this, meters.
pub const MIN_NEAR: f64 = 1.0e-3;

/// The near plane used when no marker is in front of the camera, meters.
pub const DEFAULT_NEAR: f64 = 1.0;

/// Fraction of the nearest visible marker distance used as the near plane.
pub const NEAR_FRACTION: f64 = 0.5;

/// One frame origin marker, ready for the GPU.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Marker {
    /// The frame this marker sits at.
    pub frame_id: u64,
    /// Frame origin relative to the camera, root axes, meters.
    pub position: [f32; 3],
    /// Billboard diameter in pixels.
    pub size_px: f32,
    /// Linear RGBA color.
    pub color: [f32; 4],
}

/// One end of a line, ready for the GPU.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LineVertex {
    /// Point relative to the camera, root axes, meters.
    pub position: [f32; 3],
    /// Linear RGBA color.
    pub color: [f32; 4],
}

/// One cell's surface mesh placed for drawing.
///
/// The mesh is uploaded once with positions relative to its cell origin
/// (`f32`, cell-sized values). Each frame the cell origin is made relative
/// to the camera in `f64` with [`FrameSystem::relative`] and only that
/// offset is cast to `f32`: the model transform of the draw.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDraw {
    /// The mesh, positions in its frame's coordinates.
    pub mesh: Arc<SurfaceMesh>,
    /// Cell origin relative to the camera, root axes, meters.
    pub offset: [f32; 3],
    /// Columns of the rotation from the frame axes into root axes.
    pub rotation: [[f32; 3]; 3],
    /// Weight of the cell's radiance, 0 to 1: below 1 while its frame fades
    /// in from the far field ([`crate::farfield`]).
    pub weight: f32,
}

/// One cell's volume placed for drawing, the same way as [`SurfaceDraw`].
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeDraw {
    /// The gas and plasma of the cell.
    pub grid: Arc<VolumeGrid>,
    /// Cell origin relative to the camera, root axes, meters.
    pub offset: [f32; 3],
    /// Columns of the rotation from the frame axes into root axes.
    pub rotation: [[f32; 3]; 3],
    /// Weight of the volume's glow and opacity, 0 to 1.
    pub weight: f32,
    /// Distance from the camera to the center of the gas box, meters: the
    /// sort key, farthest first.
    pub distance: f64,
}

/// Everything one render pass draws, in camera-relative `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    /// A marker at every frame origin, in ascending frame id order.
    pub markers: Vec<Marker>,
    /// Pairs of vertices: each frame origin to its parent origin.
    pub lines: Vec<LineVertex>,
    /// Surfaces of the drawn cells, in key order.
    pub surfaces: Vec<SurfaceDraw>,
    /// Volumes of the drawn cells, farthest first (ties in key order).
    pub volumes: Vec<VolumeDraw>,
    /// Point sprites of far frames.
    pub sprites: Vec<Sprite>,
    /// Point lights from hot matter, strongest first.
    pub lights: Vec<PointLight>,
    /// Rotation from root axes into camera axes, column major.
    pub view_rotation: [[f32; 4]; 4],
    /// Near plane distance, meters.
    pub near: f32,
    /// Vertical field of view, radians.
    pub fov_y: f32,
}

/// Marker diameter in pixels for a frame of edge `root_extent` meters: one
/// and a half pixels per decade above a megameter, from 3 to 18 pixels.
pub fn marker_size_px(root_extent: f64) -> f32 {
    let decades = root_extent.max(1.0).log10() - 6.0;
    (3.0 + 1.5 * decades).clamp(3.0, 18.0) as f32
}

/// Marker color by depth in the frame tree. Purely a visual aid: the color
/// says how far down the tree a frame hangs, nothing about what it holds.
pub fn marker_color(tree_depth: usize) -> [f32; 4] {
    const COLORS: [[f32; 4]; 4] = [
        [1.0, 0.85, 0.6, 1.0],
        [0.55, 0.75, 1.0, 1.0],
        [0.6, 1.0, 0.65, 1.0],
        [1.0, 0.6, 0.9, 1.0],
    ];
    COLORS[tree_depth % COLORS.len()]
}

/// Color of the line from a frame to its parent.
pub const LINE_COLOR: [f32; 4] = [0.22, 0.28, 0.4, 1.0];

/// The near plane for a set of camera-axes positions: [`NEAR_FRACTION`] of
/// the distance to the nearest point in front of the camera and inside a
/// cone a little wider than the field of view, at least [`MIN_NEAR`], or
/// [`DEFAULT_NEAR`] when nothing is in front.
pub fn near_plane(points_in_camera_axes: impl IntoIterator<Item = Vec3>, aspect: f64) -> f64 {
    let half_y = 0.5 * FIELD_OF_VIEW_Y;
    let half_x = (half_y.tan() * aspect.max(1.0)).atan();
    let cone = half_x.max(half_y) * 1.5;
    let cos_cone = cone.min(std::f64::consts::FRAC_PI_2).cos();
    let mut nearest = f64::INFINITY;
    for p in points_in_camera_axes {
        let d = p.length();
        if d > 0.0 && -p.z / d >= cos_cone && -p.z > 0.0 {
            nearest = nearest.min(d);
        }
    }
    if nearest.is_finite() {
        (nearest * NEAR_FRACTION).max(MIN_NEAR)
    } else {
        DEFAULT_NEAR
    }
}

fn to_f32(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

/// Builds the scene for the camera: a marker at every frame origin and a
/// line from each frame to its parent, all relative to the camera.
pub fn build_scene(system: &FrameSystem, camera: &Camera, aspect: f64) -> Scene {
    let tree = system.tree();
    let to_camera = camera.root_orientation(system).conjugate();
    let mut markers = Vec::with_capacity(tree.frames().len());
    let mut in_camera_axes = Vec::with_capacity(tree.frames().len());
    for f in tree.frames() {
        let rel = camera.relative(system, Vec3::zero(), f.frame_id);
        in_camera_axes.push(to_camera.rotate(rel));
        markers.push(Marker {
            frame_id: f.frame_id,
            position: to_f32(rel),
            size_px: marker_size_px(f.root_extent.value()),
            color: marker_color(tree.depth(f.frame_id)),
        });
    }
    let mut lines = Vec::new();
    for (i, f) in tree.frames().iter().enumerate() {
        if let Some(parent) = tree.parent(f.frame_id) {
            let p = tree
                .frames()
                .binary_search_by_key(&parent.frame_id, |g| g.frame_id)
                .expect("a parent is in the tree");
            for idx in [i, p] {
                lines.push(LineVertex {
                    position: markers[idx].position,
                    color: LINE_COLOR,
                });
            }
        }
    }
    // Column major rotation matrix from the camera conjugate quaternion.
    let c = |v: Vec3| to_camera.rotate(v);
    let (x, y, z) = (
        c(Vec3::new(1.0, 0.0, 0.0)),
        c(Vec3::new(0.0, 1.0, 0.0)),
        c(Vec3::new(0.0, 0.0, 1.0)),
    );
    let view_rotation = [
        [x.x as f32, x.y as f32, x.z as f32, 0.0],
        [y.x as f32, y.y as f32, y.z as f32, 0.0],
        [z.x as f32, z.y as f32, z.z as f32, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    Scene {
        markers,
        lines,
        surfaces: Vec::new(),
        volumes: Vec::new(),
        sprites: Vec::new(),
        lights: Vec::new(),
        view_rotation,
        near: near_plane(in_camera_axes, aspect) as f32,
        fov_y: FIELD_OF_VIEW_Y as f32,
    }
}

/// Distance from a point to an axis aligned box, 0 inside.
fn box_distance(p: Vec3, lo: [f64; 3], hi: [f64; 3]) -> f64 {
    let gap = |c: f64, l: f64, h: f64| (l - c).max(c - h).max(0.0);
    Vec3::new(
        gap(p.x, lo[0], hi[0]),
        gap(p.y, lo[1], hi[1]),
        gap(p.z, lo[2], hi[2]),
    )
    .length()
}

/// Columns of the rotation from `frame`'s axes into root axes.
fn frame_rotation(system: &FrameSystem, frame: u64) -> [[f32; 3]; 3] {
    let q = system.root_orientation(frame);
    let col = |v: Vec3| {
        let r = q.rotate(v);
        [r.x as f32, r.y as f32, r.z as f32]
    };
    [
        col(Vec3::new(1.0, 0.0, 0.0)),
        col(Vec3::new(0.0, 1.0, 0.0)),
        col(Vec3::new(0.0, 0.0, 1.0)),
    ]
}

/// The camera position in `frame`'s coordinates.
fn camera_in(system: &FrameSystem, camera: &Camera, frame: u64) -> Vec3 {
    system
        .root_orientation(frame)
        .conjugate()
        .rotate(system.relative(camera.position, camera.frame_id, Vec3::zero(), frame))
}

/// Adds the matter of a [`DrawList`] to a scene built by [`build_scene`]:
/// every mesh and volume placed relative to the camera, volumes sorted
/// farthest first, the lights, the sprites, and a near plane pulled in to
/// [`NEAR_FRACTION`] of the distance to the nearest mesh bounds or gas box
/// (at least [`MIN_NEAR`]) when that is closer than the markers.
pub fn add_matter(scene: &mut Scene, system: &FrameSystem, camera: &Camera, draw: &DrawList) {
    let mut nearest = f64::INFINITY;
    for mesh in &draw.meshes {
        let frame = mesh.key.frame_id;
        let offset = camera.relative(system, mesh.origin, frame);
        let cam = camera_in(system, camera, frame);
        nearest = nearest.min(box_distance(cam, mesh.bounds_min, mesh.bounds_max));
        scene.surfaces.push(SurfaceDraw {
            mesh: mesh.clone(),
            offset: to_f32(offset),
            rotation: frame_rotation(system, frame),
            weight: draw.cell_weight(frame),
        });
    }
    for grid in &draw.volumes {
        let frame = grid.key.frame_id;
        let offset = camera.relative(system, grid.origin, frame);
        let cam = camera_in(system, camera, frame);
        nearest = nearest.min(grid.distance_to(cam));
        let center = Vec3::new(
            0.5 * (grid.bounds_min[0] + grid.bounds_max[0]),
            0.5 * (grid.bounds_min[1] + grid.bounds_max[1]),
            0.5 * (grid.bounds_min[2] + grid.bounds_max[2]),
        );
        scene.volumes.push(VolumeDraw {
            grid: grid.clone(),
            offset: to_f32(offset),
            rotation: frame_rotation(system, frame),
            weight: draw.cell_weight(frame),
            distance: (center - cam).length(),
        });
    }
    scene.volumes.sort_by(|a, b| {
        b.distance
            .total_cmp(&a.distance)
            .then(a.grid.key.cmp(&b.grid.key))
    });
    scene.sprites = draw.sprites.clone();
    scene.lights = draw.lights.clone();
    if nearest.is_finite() {
        let near = (nearest * NEAR_FRACTION).max(MIN_NEAR) as f32;
        scene.near = scene.near.min(near);
    }
}

/// The scene for the camera: frame markers plus the world's drawn cells,
/// lights, and far field sprites.
pub fn matter_scene(
    world: &mut World,
    system: &FrameSystem,
    camera: &Camera,
    size: (u32, u32),
    now: f64,
) -> Scene {
    let aspect = f64::from(size.0.max(1)) / f64::from(size.1.max(1));
    let mut scene = build_scene(system, camera, aspect);
    let draw = world.draw_list(system, camera, size, now);
    add_matter(&mut scene, system, camera, &draw);
    scene
}

/// The right handed reversed-z perspective projection with an infinite far
/// plane, column major, for clip depth from 0 to 1: a point at view depth
/// `-z` gets clip `z = near` and `w = z`, so its depth is `near / z`.
pub fn reversed_infinite_projection(fov_y: f32, aspect: f32, near: f32) -> glam::Mat4 {
    let f = 1.0 / (0.5 * fov_y).tan();
    glam::Mat4::from_cols(
        glam::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, f, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.0, -1.0),
        glam::Vec4::new(0.0, 0.0, near, 0.0),
    )
}

impl Scene {
    /// The reversed-z infinite projection times the view rotation, column
    /// major: what the shaders multiply camera-relative positions by.
    pub fn view_projection(&self, aspect: f32) -> glam::Mat4 {
        reversed_infinite_projection(self.fov_y, aspect, self.near)
            * glam::Mat4::from_cols_array_2d(&self.view_rotation)
    }

    /// Projects a camera-relative position to pixel coordinates (origin top
    /// left), or `None` when it is behind the camera.
    pub fn project(&self, position: [f32; 3], width: u32, height: u32) -> Option<(f32, f32)> {
        let clip = self.view_projection(width as f32 / height as f32)
            * glam::Vec4::new(position[0], position[1], position[2], 1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some((
            (ndc.x * 0.5 + 0.5) * width as f32,
            (0.5 - ndc.y * 0.5) * height as f32,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn marker_sizes_are_logarithmic_and_bounded() {
        assert_eq!(marker_size_px(1.0e6), 3.0);
        assert_eq!(marker_size_px(1.0e8), 6.0);
        assert_eq!(marker_size_px(1.0e2), 3.0);
        assert_eq!(marker_size_px(1.0e30), 18.0);
    }

    #[test]
    fn projection_is_reversed_and_infinite() {
        let p = reversed_infinite_projection(1.0, 2.0, 0.5);
        let depth = |z: f32| {
            let c = p * glam::Vec4::new(0.0, 0.0, -z, 1.0);
            c.z / c.w
        };
        assert_eq!(depth(0.5), 1.0);
        assert_eq!(depth(1.0), 0.5);
        assert!(depth(1.0e30) > 0.0 && depth(1.0e30) < 1.0e-29);
    }

    #[test]
    fn near_plane_from_nearest_visible_point() {
        let pts = [
            Vec3::new(0.0, 0.0, -100.0),
            Vec3::new(0.0, 0.0, 10.0), // behind
            Vec3::new(0.0, 0.0, -40.0),
        ];
        assert_eq!(near_plane(pts, 16.0 / 9.0), 20.0);
        assert_eq!(near_plane([Vec3::new(0.0, 0.0, 5.0)], 1.0), DEFAULT_NEAR);
        assert_eq!(near_plane([Vec3::new(0.0, 0.0, -1.0e-9)], 1.0), MIN_NEAR);
    }
}
