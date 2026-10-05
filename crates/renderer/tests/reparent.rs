//! Re-parenting invariance on the conformance frame tree: moving the camera
//! from one frame to another changes neither its place in space nor the
//! image.

use gx_core::frames::FrameSystem;
use gx_core::integrate::{advance, Scheme};
use gx_core::registry::{self, FrameTree};
use gx_core::units::{Seconds, Vec3};
use renderer::camera::{axis_angle, Camera};
use renderer::render::scene::build_scene;

fn tree_system() -> FrameSystem {
    let reg = registry::decode(include_bytes!("data/tree.bin")).unwrap();
    FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
}

#[test]
fn same_point_from_two_frames() {
    let mut system = tree_system();
    // Some time later, so every frame has moved and turned.
    let to = system.time() + Seconds::new(5.0 * 86_400.0);
    advance(&mut system, to, Seconds::new(600.0), Scheme::Yoshida4);
    // A camera 3000 km from frame 31, parented to its parent frame 30.
    let near31 = Vec3::new(1.0e6, -2.0e6, 2.0e6);
    let in30 = system.relative(near31, 31, Vec3::zero(), 30);
    let mut cam = Camera {
        frame_id: 30,
        position: system.root_orientation(30).conjugate().rotate(in30),
        orientation: axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.4),
        speed: 1.0,
    };
    // The vector from frame 31's origin to the camera, both ways.
    let before = system.relative(cam.position, 30, Vec3::zero(), 31);
    assert!(cam.update_parent(&system));
    assert_eq!(cam.frame_id, 31);
    let after = system.relative(cam.position, 31, Vec3::zero(), 31);
    assert!(
        (after - before).length() < 1e-6,
        "moved by {} m",
        (after - before).length()
    );
    assert!((after - near31_root_axes(&system, near31)).length() < 1e-6);
}

fn near31_root_axes(system: &FrameSystem, p: Vec3) -> Vec3 {
    system.root_orientation(31).rotate(p)
}

#[test]
fn image_does_not_jump_when_reparenting() {
    let system = tree_system();
    let aspect = 16.0 / 9.0;
    // Start at the view of frame 30, then express the same camera in the
    // root frame and in frame 31, and compare what the GPU would receive.
    let base = Camera::view_of(&system, 30).unwrap();
    let reference = build_scene(&system, &base, aspect);
    for target in [1, 31, 10, 40] {
        let mut cam = base;
        cam.reparent_to(&system, target);
        let scene = build_scene(&system, &cam, aspect);
        for (a, b) in reference.markers.iter().zip(&scene.markers) {
            let pa = glam::Vec3::from(a.position);
            let pb = glam::Vec3::from(b.position);
            let tol = 1.0e-6 * pa.length().max(1.0);
            assert!(
                (pa - pb).length() <= tol,
                "frame {} to {target}: {pa} vs {pb}",
                a.frame_id
            );
        }
        let ra = glam::Mat4::from_cols_array_2d(&reference.view_rotation);
        let rb = glam::Mat4::from_cols_array_2d(&scene.view_rotation);
        assert!(
            ra.abs_diff_eq(rb, 1.0e-6),
            "view rotation changed for {target}"
        );
        assert_eq!(reference.near, scene.near);
        // Re-parenting back lands on the start.
        cam.reparent_to(&system, 30);
        assert!((cam.position - base.position).length() < 1.0e-6 * base.position.length());
    }
}
