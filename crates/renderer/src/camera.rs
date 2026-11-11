//! The free camera and the floating origin.
//!
//! Implements the camera half of `space-model.md` section 5 (reference
//! frames with a floating origin): "The camera is positioned relative to the
//! frame it is nearest, and re-parents when it leaves one frame's region for
//! another. The GPU only ever sees camera-relative single-precision
//! coordinates."
//!
//! A [`Camera`] is a frame id, a position in that frame (relative to the
//! frame origin, in the frame axes, meters, `f64`), an orientation relative
//! to the frame axes, and a flight speed. The parent frame is only a
//! precision device: the camera behaves like an observer floating free in
//! space, so a simulation step that moves or turns its frame does not carry
//! it along. [`Camera::hold`] before the step and [`Camera::keep`] after it
//! put it back at the same place and view direction in root coordinates.
//!
//! After every simulation step [`Camera::update_parent`] asks
//! [`FrameSystem::nearest_frame`] for the frame nearest the camera and, when
//! it differs, converts the position and orientation into the new frame with
//! [`FrameSystem::relative`] and the frame orientations, so the camera's
//! place and view direction in space do not change and the image does not
//! jump.
//!
//! Camera axes follow the usual right handed view convention: the camera
//! looks along its local `-z`, `+x` is right, and `+y` is up.

use gx_core::frames::FrameSystem;
use gx_core::registry::FrameTree;
use gx_core::units::{Quat, Vec3};

/// Vertical field of view, radians (60 degrees).
pub const FIELD_OF_VIEW_Y: f64 = std::f64::consts::FRAC_PI_3;

/// Look rotation per pixel of right mouse drag, radians.
pub const LOOK_RADIANS_PER_PIXEL: f64 = 0.003;

/// Factor applied to the set flight speed per mouse wheel notch: about ten
/// notches per factor of ten.
pub const WHEEL_FACTOR: f64 = 1.25;

/// Factor on the set speed while the boost key (Shift) is held.
pub const BOOST_FACTOR: f64 = 10.0;

/// Time constant of the glide from the current flight speed to the set one,
/// seconds: a change of speed is about 99 percent done after five of these.
pub const SPEED_GLIDE_SECONDS: f64 = 0.15;

/// Limits of the flight speed, meters per second.
pub const SPEED_LIMITS: (f64, f64) = (1.0e-3, 1.0e15);

/// The flight speed of a newly placed camera, meters per second.
pub const DEFAULT_SPEED_MPS: f64 = 1.0e11;

/// The keys and mouse movement of one rendered frame.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct FlightInput {
    /// Move along the view direction (W).
    pub forward: bool,
    /// Move against the view direction (S).
    pub back: bool,
    /// Move left (A).
    pub left: bool,
    /// Move right (D).
    pub right: bool,
    /// Move up along the camera's up axis (E).
    pub up: bool,
    /// Move down along the camera's up axis (Q).
    pub down: bool,
    /// Right mouse drag since the last frame, pixels, `+x` right, `+y` down.
    pub look: (f64, f64),
    /// Mouse wheel notches since the last frame, positive away from the
    /// user.
    pub wheel: f64,
    /// Fly at [`BOOST_FACTOR`] times the set speed (Shift held).
    pub boost: bool,
}

/// The free camera: a place and a view direction in one frame.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Camera {
    /// The frame the camera is parented to.
    pub frame_id: u64,
    /// Position relative to the frame origin, in the frame axes, meters.
    pub position: Vec3,
    /// Camera axes relative to the frame axes, a unit quaternion.
    pub orientation: Quat,
    /// The set flight speed, meters per second. Changed only by the
    /// wheel: nothing in space changes it.
    pub speed: f64,
    /// The current flight speed, meters per second: glides toward the set
    /// speed (times [`BOOST_FACTOR`] while boosting) in
    /// [`SPEED_GLIDE_SECONDS`], so a burst of wheel notches does not jump.
    pub cruise: f64,
}

/// The camera's place in space before a simulation step (see
/// [`Camera::hold`] and [`Camera::keep`]).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Hold {
    root_position: Vec3,
    root_orientation: Quat,
}

/// A rotation of `angle` radians about the unit `axis`.
pub fn axis_angle(axis: Vec3, angle: f64) -> Quat {
    let (s, c) = (0.5 * angle).sin_cos();
    Quat::new(axis.x * s, axis.y * s, axis.z * s, c)
}

/// The rotation whose camera looks along `forward` with its up axis as close
/// to `up_hint` as possible, both given in the same axes. Falls back to
/// another hint when the two are parallel.
pub fn look_rotation(forward: Vec3, up_hint: Vec3) -> Quat {
    let f = forward.normalized().unwrap_or(Vec3::new(0.0, 0.0, -1.0));
    let right = f
        .cross(up_hint)
        .normalized()
        .or_else(|| f.cross(Vec3::new(0.0, 1.0, 0.0)).normalized())
        .or_else(|| f.cross(Vec3::new(1.0, 0.0, 0.0)).normalized())
        .expect("a unit vector is not parallel to two orthogonal axes");
    let up = right.cross(f);
    // Columns of the rotation: camera x, y, z in the given axes.
    quat_from_columns(right, up, -f)
}

/// Converts an orthonormal right handed basis (the columns of a rotation
/// matrix) into a unit quaternion.
fn quat_from_columns(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let (m00, m11, m22) = (x.x, y.y, z.z);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat::new((y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, 0.25 * s)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        Quat::new(0.25 * s, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        Quat::new((y.x + x.y) / s, 0.25 * s, (z.y + y.z) / s, (z.x - x.z) / s)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        Quat::new((z.x + x.z) / s, (z.y + y.z) / s, 0.25 * s, (x.y - y.x) / s)
    };
    q.normalized().unwrap_or(Quat::identity())
}

/// The frames the number keys select: every frame id in ascending order,
/// skipping the root. Key `1` selects index 0, key `0` index 9.
pub fn selectable_frames(tree: &FrameTree) -> Vec<u64> {
    tree.frames()
        .iter()
        .filter(|f| !f.is_root())
        .map(|f| f.frame_id)
        .collect()
}

impl Camera {
    /// The camera position in root coordinates. Large values lose
    /// precision; use [`Camera::relative`] for anything drawn.
    pub fn root_position(&self, system: &FrameSystem) -> Vec3 {
        system.from_frame(self.position, self.frame_id)
    }

    /// The camera axes relative to root axes.
    pub fn root_orientation(&self, system: &FrameSystem) -> Quat {
        system.root_orientation(self.frame_id) * self.orientation
    }

    /// The vector from the camera to `point` (a point in `frame_id`), in
    /// root axes: the floating-origin primitive every drawn position goes
    /// through.
    pub fn relative(&self, system: &FrameSystem, point: Vec3, frame_id: u64) -> Vec3 {
        system.relative(point, frame_id, self.position, self.frame_id)
    }

    /// Applies one frame of free flight: look, wheel, then movement in the
    /// camera axes over `dt` seconds of wall-clock time.
    pub fn fly(&mut self, input: &FlightInput, dt: f64) {
        let (dx, dy) = input.look;
        if dx != 0.0 || dy != 0.0 {
            let yaw = axis_angle(Vec3::new(0.0, 1.0, 0.0), -dx * LOOK_RADIANS_PER_PIXEL);
            let pitch = axis_angle(Vec3::new(1.0, 0.0, 0.0), -dy * LOOK_RADIANS_PER_PIXEL);
            self.orientation = (self.orientation * yaw * pitch)
                .normalized()
                .unwrap_or(self.orientation);
        }
        let (lo, hi) = SPEED_LIMITS;
        if input.wheel != 0.0 {
            self.speed = (self.speed * WHEEL_FACTOR.powf(input.wheel)).clamp(lo, hi);
        }
        let target = if input.boost {
            (self.speed * BOOST_FACTOR).min(hi)
        } else {
            self.speed
        };
        // Glide in log space, so every factor of ten takes the same time.
        let k = 1.0 - (-dt / SPEED_GLIDE_SECONDS).exp();
        let (now, to) = (self.cruise.clamp(lo, hi).ln(), target.ln());
        self.cruise = (now + (to - now) * k).exp();
        let axis = |plus: bool, minus: bool| f64::from(u8::from(plus)) - f64::from(u8::from(minus));
        let local = Vec3::new(
            axis(input.right, input.left),
            axis(input.up, input.down),
            axis(input.back, input.forward),
        );
        if let Some(dir) = local.normalized() {
            let step = self.orientation.rotate(dir).scale(self.cruise * dt);
            self.position = self.position + step;
        }
    }

    /// Re-expresses the camera in `frame_id` without moving it: the same
    /// place and view direction in space, a new parent.
    pub fn reparent_to(&mut self, system: &FrameSystem, frame_id: u64) {
        if frame_id == self.frame_id {
            return;
        }
        let new_axes = system.root_orientation(frame_id).conjugate();
        let offset = system.relative(self.position, self.frame_id, Vec3::zero(), frame_id);
        let view = self.root_orientation(system);
        self.position = new_axes.rotate(offset);
        self.orientation = (new_axes * view).normalized().unwrap_or(view);
        self.frame_id = frame_id;
    }

    /// Re-parents the camera to the frame nearest it, if that changed.
    /// Returns `true` when it did. Call after every simulation step.
    pub fn update_parent(&mut self, system: &FrameSystem) -> bool {
        let nearest = system.nearest_frame(self.root_position(system));
        if nearest == self.frame_id {
            return false;
        }
        self.reparent_to(system, nearest);
        true
    }

    /// Records where the camera is in space before a simulation step, for
    /// [`Camera::keep`] after it.
    pub fn hold(&self, system: &FrameSystem) -> Hold {
        Hold {
            root_position: self.root_position(system),
            root_orientation: self.root_orientation(system),
        }
    }

    /// Puts the camera back where `hold` recorded it after a simulation
    /// step moved and turned its frame: the same place and view direction
    /// in root coordinates, the way an observer floating free would see it.
    pub fn keep(&mut self, system: &FrameSystem, hold: Hold) {
        self.position = system.to_frame(hold.root_position, self.frame_id);
        let to_frame = system.root_orientation(self.frame_id).conjugate();
        self.orientation = (to_frame * hold.root_orientation)
            .normalized()
            .unwrap_or(self.orientation);
    }

    /// A camera in `frame_id` at `offset` from the frame origin (root axes)
    /// looking along `forward` (root axes) with `up` (root axes) as the up
    /// hint.
    fn placed(
        system: &FrameSystem,
        frame_id: u64,
        offset: Vec3,
        forward: Vec3,
        up: Vec3,
    ) -> Camera {
        let to_frame = system.root_orientation(frame_id).conjugate();
        let look = look_rotation(forward, up);
        Camera {
            frame_id,
            position: to_frame.rotate(offset),
            orientation: (to_frame * look).normalized().unwrap_or(look),
            speed: DEFAULT_SPEED_MPS,
            cruise: DEFAULT_SPEED_MPS,
        }
    }

    /// The `Home` view: the whole system seen from above the root, along the
    /// root `-z` axis with root `+y` up, far enough that every other frame's
    /// origin and its `root_extent / 8` region fit the vertical field of
    /// view. With no other frame, the root's own `root_extent / 8` region
    /// fills the view.
    pub fn home(system: &FrameSystem) -> Camera {
        let root = system.tree().root();
        let radius = system
            .tree()
            .frames()
            .iter()
            .filter(|f| !f.is_root())
            .map(|f| {
                let r = system
                    .relative(Vec3::zero(), f.frame_id, Vec3::zero(), root.frame_id)
                    .length();
                r + f.root_extent.value() / 8.0
            })
            .fold(None, |acc: Option<f64>, r| {
                Some(acc.map_or(r, |a| a.max(r)))
            })
            .unwrap_or(root.root_extent.value() / 8.0);
        let height = 1.2 * radius / (0.5 * FIELD_OF_VIEW_Y).tan();
        Camera::placed(
            system,
            root.frame_id,
            Vec3::new(0.0, 0.0, height),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
    }

    /// A view of `frame_id` from above, the way [`Camera::home`] sees the
    /// root: parented to the frame, `3 * root_extent / 8` from its origin
    /// along root `+z`, looking along root `-z` with root `+y` up. The
    /// direction is fixed in root axes, so whatever lights the frame shows
    /// up as a lit side and a dark side whose direction follows the light
    /// as simulation time passes. `None` if the id is not in the tree.
    pub fn view_of(system: &FrameSystem, frame_id: u64) -> Option<Camera> {
        let frame = system.tree().get(frame_id)?;
        let distance = 3.0 * frame.root_extent.value() / 8.0;
        Some(Camera::placed(
            system,
            frame_id,
            Vec3::new(0.0, 0.0, distance),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ))
    }

    /// The same camera at `factor` times its distance from its frame
    /// origin, along the same line, with the same view direction.
    pub fn with_distance_scale(self, factor: f64) -> Camera {
        Camera {
            position: self.position.scale(factor),
            ..self
        }
    }

    /// The view the number key with this index selects (see
    /// [`selectable_frames`]), or `None` past the end.
    pub fn view_of_index(system: &FrameSystem, index: usize) -> Option<Camera> {
        let id = *selectable_frames(system.tree()).get(index)?;
        Camera::view_of(system, id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gx_core::registry::{Frame, Registry, ROOT_PARENT};
    use gx_core::units::{Kilograms, Meters, Seconds};

    fn frame(id: u64, parent: u64, extent: f64, position: Vec3, orientation: Quat) -> Frame {
        Frame {
            frame_id: id,
            parent_frame_id: parent,
            root_extent: Meters::new(extent),
            max_depth: 4,
            mass: Kilograms::new(0.0),
            position,
            velocity: Vec3::zero(),
            orientation,
            angular_velocity: Vec3::zero(),
        }
    }

    /// Small offsets, so root coordinates themselves are exact enough to
    /// compare at a micrometer.
    fn system() -> FrameSystem {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let frames = vec![
            {
                // A massless origin with no matter, like a barycenter.
                let mut root = frame(
                    1,
                    ROOT_PARENT,
                    1.0e4,
                    Vec3::zero(),
                    Quat::new(0.0, 0.0, 0.6, 0.8),
                );
                root.max_depth = 0;
                root
            },
            frame(
                5,
                1,
                400.0,
                Vec3::new(1000.0, 0.0, 0.0),
                Quat::new(0.0, 0.0, h, h),
            ),
            frame(
                7,
                5,
                80.0,
                Vec3::new(0.0, 100.0, 0.0),
                Quat::new(h, 0.0, 0.0, h),
            ),
            frame(
                3,
                1,
                200.0,
                Vec3::new(0.0, 0.0, -500.0),
                Quat::new(0.5, 0.5, 0.5, 0.5),
            ),
        ];
        let reg = Registry::new(Seconds::new(0.0), frames).unwrap();
        FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
    }

    fn close(a: Vec3, b: Vec3, tol: f64) -> bool {
        (a - b).length() <= tol
    }

    /// A massless root and one massless frame moving at 1000 m/s along
    /// root +x and spinning about root +z at 0.1 rad/s.
    fn moving_system() -> FrameSystem {
        let mut f = frame(1, ROOT_PARENT, 1.0e5, Vec3::zero(), Quat::identity());
        f.max_depth = 0;
        let mut g = frame(2, 1, 1.0e3, Vec3::new(1.0e4, 0.0, 0.0), Quat::identity());
        g.velocity = Vec3::new(1000.0, 0.0, 0.0);
        g.angular_velocity = Vec3::new(0.0, 0.0, 0.1);
        let reg = Registry::new(Seconds::new(0.0), vec![f, g]).unwrap();
        FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
    }

    fn step(s: &mut FrameSystem, seconds: f64) {
        gx_core::integrate::step(s, Seconds::new(seconds), Default::default());
    }

    #[test]
    fn a_free_camera_stays_put_while_its_frame_moves_and_spins() {
        let mut s = moving_system();
        let mut cam = Camera::view_of(&s, 2).unwrap();
        let place = cam.root_position(&s);
        let v = Vec3::new(0.3, -0.2, 0.9);
        let view = cam.root_orientation(&s).rotate(v);
        for _ in 0..10 {
            let held = cam.hold(&s);
            step(&mut s, 1.0);
            cam.keep(&s, held);
            cam.update_parent(&s);
        }
        // The frame moved 10 km and turned a radian; the camera did not.
        assert!((s.root_position(2).x - 2.0e4).abs() < 1e-6);
        let x = s.root_orientation(2).rotate(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            (x.x - 1.0f64.cos()).abs() < 1e-2,
            "turned a radian, x = {x:?}"
        );
        assert!(close(cam.root_position(&s), place, 1e-6));
        assert!(close(cam.root_orientation(&s).rotate(v), view, 1e-9));
    }

    #[test]
    fn flying_left_moves_what_is_ahead_right_on_screen() {
        use crate::render::scene::build_scene;
        let s = moving_system();
        let mut cam = Camera::view_of(&s, 2).unwrap();
        cam.speed = 100.0;
        cam.cruise = 100.0;
        let x_of_frame_2 = |cam: &Camera| {
            let scene = build_scene(&s, cam, 16.0 / 9.0);
            let m = scene.markers.iter().find(|m| m.frame_id == 2).unwrap();
            scene.project(m.position, 1600, 900).unwrap().0
        };
        let before = x_of_frame_2(&cam);
        assert!((before - 800.0).abs() < 1.0, "frame 2 starts centered");
        let left = FlightInput {
            left: true,
            ..FlightInput::default()
        };
        cam.fly(&left, 0.01);
        assert!(x_of_frame_2(&cam) > before + 1.0, "A slides the view right");
        let right = FlightInput {
            right: true,
            ..FlightInput::default()
        };
        cam.fly(&right, 0.02);
        assert!(x_of_frame_2(&cam) < before - 1.0, "D slides the view left");
    }

    #[test]
    fn look_rotation_points_the_camera() {
        let q = look_rotation(Vec3::new(1.0, 2.0, -0.5), Vec3::new(0.0, 0.0, 1.0));
        let f = Vec3::new(1.0, 2.0, -0.5).normalized().unwrap();
        assert!(close(q.rotate(Vec3::new(0.0, 0.0, -1.0)), f, 1e-12));
        let up = q.rotate(Vec3::new(0.0, 1.0, 0.0));
        assert!(up.z > 0.0 && up.dot(f).abs() < 1e-12);
        // Parallel hint falls back instead of failing.
        let q = look_rotation(Vec3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(close(
            q.rotate(Vec3::new(0.0, 0.0, -1.0)),
            Vec3::new(0.0, 0.0, -1.0),
            1e-12
        ));
    }

    #[test]
    fn reparenting_keeps_the_root_space_point() {
        let s = system();
        let mut cam = Camera {
            frame_id: 7,
            position: Vec3::new(3.0, -40.0, 12.5),
            orientation: axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.7),
            speed: 1.0,
            cruise: 1.0,
        };
        let before = cam.root_position(&s);
        let before_view = cam.root_orientation(&s);
        for target in [3, 1, 5, 7] {
            cam.reparent_to(&s, target);
            assert_eq!(cam.frame_id, target);
            assert!(close(cam.root_position(&s), before, 1e-6), "frame {target}");
            let v = Vec3::new(0.3, -0.2, 0.9);
            assert!(close(
                cam.root_orientation(&s).rotate(v),
                before_view.rotate(v),
                1e-12
            ));
        }
    }

    #[test]
    fn update_parent_follows_the_nearest_frame() {
        let s = system();
        let mut cam = Camera::home(&s);
        assert_eq!(cam.frame_id, 1);
        assert!(!cam.update_parent(&s));
        // Put the camera next to frame 3 while still parented to the root.
        let near3 = s.from_frame(Vec3::new(5.0, 0.0, 0.0), 3);
        cam.position = s.to_frame(near3, 1);
        assert!(cam.update_parent(&s));
        assert_eq!(cam.frame_id, 3);
        assert!(close(cam.position, Vec3::new(5.0, 0.0, 0.0), 1e-9));
    }

    #[test]
    fn home_sees_the_root_from_above() {
        let s = system();
        let cam = Camera::home(&s);
        let to_root = cam.relative(&s, Vec3::zero(), 1);
        let forward = cam.root_orientation(&s).rotate(Vec3::new(0.0, 0.0, -1.0));
        let along = to_root.normalized().unwrap();
        assert!(close(forward, along, 1e-9));
        assert!(close(along, Vec3::new(0.0, 0.0, -1.0), 1e-9));
    }

    #[test]
    fn number_keys_view_frames_in_id_order() {
        let s = system();
        assert_eq!(selectable_frames(s.tree()), vec![3, 5, 7]);
        let cam = Camera::view_of_index(&s, 1).unwrap();
        assert_eq!(cam.frame_id, 5);
        assert!((cam.position.length() - 3.0 * 400.0 / 8.0).abs() < 1e-9);
        // Looking at the frame origin.
        let to_origin = cam.relative(&s, Vec3::zero(), 5).normalized().unwrap();
        let forward = cam.root_orientation(&s).rotate(Vec3::new(0.0, 0.0, -1.0));
        assert!(close(forward, to_origin, 1e-9));
        // Placed above the frame, looking down root -z with root +y up.
        assert!(close(to_origin, Vec3::new(0.0, 0.0, -1.0), 1e-9));
        let up = cam.root_orientation(&s).rotate(Vec3::new(0.0, 1.0, 0.0));
        assert!(close(up, Vec3::new(0.0, 1.0, 0.0), 1e-9));
        assert!(Camera::view_of_index(&s, 3).is_none());
    }

    /// The camera is not affected by anything in space: holding W flies a
    /// straight line at the set speed through a star and past it, no matter
    /// what frame it is parented to, how near matter is, or how the frames
    /// move while the simulation runs.
    #[test]
    fn flight_is_a_straight_line_at_the_set_speed_through_matter() {
        let mut root = frame(1, ROOT_PARENT, 2.0e13, Vec3::zero(), Quat::identity());
        root.max_depth = 0;
        let star_radius = 6.957e8;
        let mut star = frame(
            2,
            1,
            8.0 * star_radius,
            Vec3::new(1.0e9, 0.0, 0.0),
            Quat::identity(),
        );
        star.velocity = Vec3::new(0.0, 4.5e4, 0.0);
        star.angular_velocity = Vec3::new(0.0, 0.0, 1.0e-3);
        let reg = Registry::new(Seconds::new(0.0), vec![root, star]).unwrap();
        let mut s = FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap());
        let start = Vec3::new(-1.0e11, 0.0, 0.0);
        let mut cam = Camera {
            frame_id: 1,
            position: start,
            orientation: look_rotation(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
            speed: 2.0e10,
            cruise: 2.0e10,
        };
        let view = cam.root_orientation(&s);
        let w = FlightInput {
            forward: true,
            ..FlightInput::default()
        };
        // Ten seconds of holding W at 60 frames per second, the simulation
        // running and the camera re-parenting as in the app.
        let dt = 1.0 / 60.0;
        let mut parents = std::collections::BTreeSet::new();
        for _ in 0..600 {
            let held = cam.hold(&s);
            step(&mut s, 100.0 * dt);
            cam.keep(&s, held);
            cam.fly(&w, dt);
            cam.update_parent(&s);
            parents.insert(cam.frame_id);
        }
        assert_eq!(parents.len(), 2, "re-parented to the star and back");
        let flown = cam.root_position(&s) - start;
        assert!(
            close(flown, Vec3::new(2.0e11, 0.0, 0.0), 1.0e2),
            "{flown:?}"
        );
        let v = Vec3::new(0.3, -0.2, 0.9);
        assert!(close(
            cam.root_orientation(&s).rotate(v),
            view.rotate(v),
            1e-9
        ));
    }

    #[test]
    fn flight_moves_along_view_at_the_set_speed() {
        let s = system();
        let mut cam = Camera::view_of(&s, 5).unwrap();
        assert_eq!(cam.speed, DEFAULT_SPEED_MPS);
        cam.speed = 100.0;
        cam.cruise = 100.0;
        let start = cam.position;
        let input = FlightInput {
            forward: true,
            ..FlightInput::default()
        };
        cam.fly(&input, 0.5);
        let moved = cam.position - start;
        assert!((moved.length() - 50.0).abs() < 1e-9);
        assert!(close(
            moved.normalized().unwrap(),
            -start.normalized().unwrap(),
            1e-9
        ));
        // The wheel scales speed geometrically.
        cam.fly(
            &FlightInput {
                wheel: 2.0,
                ..FlightInput::default()
            },
            0.0,
        );
        assert!((cam.speed - 100.0 * WHEEL_FACTOR * WHEEL_FACTOR).abs() < 1e-9);
    }

    #[test]
    fn speed_glides_to_the_set_speed_and_boost_multiplies_it() {
        let s = system();
        let mut cam = Camera::view_of(&s, 5).unwrap();
        cam.speed = 100.0;
        cam.cruise = 100.0;
        let frame = |cam: &mut Camera, input: FlightInput| {
            for _ in 0..60 {
                cam.fly(&input, 1.0 / 60.0);
            }
        };
        // Ten notches in one frame: the set speed jumps, the flight speed
        // follows over the next second instead of at once.
        cam.fly(
            &FlightInput {
                wheel: 10.0,
                ..FlightInput::default()
            },
            1.0 / 60.0,
        );
        let set = 100.0 * WHEEL_FACTOR.powi(10);
        assert!((cam.speed - set).abs() < 1e-9);
        assert!(
            cam.cruise > 100.0 && cam.cruise < 0.5 * set,
            "{}",
            cam.cruise
        );
        frame(&mut cam, FlightInput::default());
        assert!((cam.cruise / set - 1.0).abs() < 1e-2, "{}", cam.cruise);
        // Shift: ten times the set speed while held, back after release.
        let boost = FlightInput {
            boost: true,
            ..FlightInput::default()
        };
        frame(&mut cam, boost);
        assert!((cam.cruise / (BOOST_FACTOR * set) - 1.0).abs() < 1e-2);
        assert!((cam.speed - set).abs() < 1e-9, "boost leaves the set speed");
        frame(&mut cam, FlightInput::default());
        assert!((cam.cruise / set - 1.0).abs() < 1e-2);
    }

    #[test]
    fn look_turns_the_camera() {
        let s = system();
        let mut cam = Camera::home(&s);
        let before = cam.orientation.rotate(Vec3::new(0.0, 0.0, -1.0));
        cam.fly(
            &FlightInput {
                look: (100.0, 0.0),
                ..FlightInput::default()
            },
            0.016,
        );
        let after = cam.orientation.rotate(Vec3::new(0.0, 0.0, -1.0));
        let angle = before.dot(after).clamp(-1.0, 1.0).acos();
        assert!((angle - 100.0 * LOOK_RADIANS_PER_PIXEL).abs() < 1e-9);
    }

    #[test]
    fn distance_scale_keeps_line_and_direction() {
        let s = system();
        let cam = Camera::view_of_index(&s, 0).unwrap();
        let near = cam.with_distance_scale(0.25);
        assert_eq!(near.frame_id, cam.frame_id);
        assert_eq!(near.orientation, cam.orientation);
        assert!((near.position.length() - 0.25 * cam.position.length()).abs() < 1e-9);
        assert!(close(
            near.position.normalized().unwrap(),
            cam.position.normalized().unwrap(),
            1e-12
        ));
    }
}
