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
//! [`near_plane`]), so precision is spent where the nearest matter is.

use crate::camera::{Camera, FIELD_OF_VIEW_Y};
use gx_core::frames::FrameSystem;
use gx_core::units::Vec3;

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

/// Everything one render pass draws, in camera-relative `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    /// A marker at every frame origin, in ascending frame id order.
    pub markers: Vec<Marker>,
    /// Pairs of vertices: each frame origin to its parent origin.
    pub lines: Vec<LineVertex>,
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
        view_rotation,
        near: near_plane(in_camera_axes, aspect) as f32,
        fov_y: FIELD_OF_VIEW_Y as f32,
    }
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
