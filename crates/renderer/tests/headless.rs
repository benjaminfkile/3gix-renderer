//! Headless integration test: renders the conformance frame tree on
//! whatever adapter wgpu finds (here, the software Vulkan driver) and checks
//! the pixels.
//!
//! `tests/data/` holds the core library's valid registry conformance vectors
//! (`conformance/registry/valid/`), copied unchanged.

use gx_core::frames::FrameSystem;
use gx_core::registry::{self, FrameTree};
use renderer::camera::Camera;
use renderer::render::headless::{Headless, Image};
use renderer::render::overlay::OverlayInfo;
use renderer::render::scene::build_scene;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;

fn tree_system() -> FrameSystem {
    let bytes = include_bytes!("data/tree.bin");
    let reg = registry::decode(bytes).expect("conformance registry decodes");
    FrameSystem::from_tree(FrameTree::from_registries(&[reg]).expect("valid union"))
}

fn lit(p: [u8; 4]) -> bool {
    p[0] > 0 || p[1] > 0 || p[2] > 0
}

fn lit_in(image: &Image, x0: u32, y0: u32, x1: u32, y1: u32) -> usize {
    let mut n = 0;
    for y in y0..y1.min(image.height) {
        for x in x0..x1.min(image.width) {
            n += usize::from(lit(image.pixel(x, y)));
        }
    }
    n
}

#[test]
fn home_view_renders_markers_and_overlay_deterministically() {
    let system = tree_system();
    let camera = Camera::home(&system);
    let scene = build_scene(&system, &camera, f64::from(WIDTH) / f64::from(HEIGHT));
    let overlay = OverlayInfo {
        sim_time: system.time(),
        time_scale: 1.0,
        paused: false,
        camera_frame: camera.frame_id,
        camera_distance: camera.position.length(),
        frame_count: system.tree().frames().len(),
        fps: None,
        sim_lag: false,
        matter: Default::default(),
    }
    .text();

    let mut headless = Headless::new(WIDTH, HEIGHT).expect("a wgpu adapter, software is fine");
    let first = headless.render(&scene, Some(&overlay)).unwrap();
    let second = headless.render(&scene, Some(&overlay)).unwrap();
    let bare = headless.render(&scene, None).unwrap();
    assert_eq!((first.width, first.height), (WIDTH, HEIGHT));
    assert_eq!(first.rgba.len(), (WIDTH * HEIGHT * 4) as usize);

    // A marker pixel near the projected root origin.
    let root = scene
        .markers
        .iter()
        .find(|m| m.frame_id == system.tree().root().frame_id)
        .unwrap();
    let (rx, ry) = scene
        .project(root.position, WIDTH, HEIGHT)
        .expect("root in front");
    assert!((rx - WIDTH as f32 / 2.0).abs() < 1.0 && (ry - HEIGHT as f32 / 2.0).abs() < 1.0);
    let (cx, cy) = (rx.round() as u32, ry.round() as u32);
    let r = (root.size_px / 2.0).ceil() as u32 + 1;
    assert!(
        lit_in(&first, cx - r, cy - r, cx + r + 1, cy + r + 1) > 0,
        "no marker pixel near the projected root at ({rx}, {ry})"
    );
    // Every frame origin projects inside the image from Home.
    for m in &scene.markers {
        let (x, y) = scene.project(m.position, WIDTH, HEIGHT).expect("in front");
        assert!(
            x >= 0.0 && x < WIDTH as f32 && y >= 0.0 && y < HEIGHT as f32,
            "frame {}",
            m.frame_id
        );
    }

    // The overlay region, top left, has text pixels the bare render lacks.
    let with_text = lit_in(&first, 0, 0, 320, 120);
    let without = lit_in(&bare, 0, 0, 320, 120);
    assert!(
        with_text > without + 200,
        "overlay pixels: {with_text} vs {without}"
    );

    // Same state, same pixels.
    assert!(first == second, "two renders of the same state differ");

    if let Ok(dir) = std::env::var("GX_TEST_OUTPUT_DIR") {
        first
            .write_png(&std::path::Path::new(&dir).join("headless-home.png"))
            .unwrap();
    }
}

#[test]
fn single_root_and_frame_views_render() {
    let reg = registry::decode(include_bytes!("data/single-root.bin")).unwrap();
    let system = FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap());
    let camera = Camera::home(&system);
    let scene = build_scene(&system, &camera, f64::from(WIDTH) / f64::from(HEIGHT));
    assert_eq!(scene.markers.len(), 1);
    assert!(scene.lines.is_empty());
    let mut headless = Headless::new(WIDTH, HEIGHT).unwrap();
    let image = headless.render(&scene, None).unwrap();
    let (cx, cy) = (WIDTH / 2, HEIGHT / 2);
    assert!(lit_in(&image, cx - 4, cy - 4, cx + 5, cy + 5) > 0);

    // A number key view of every frame of the tree renders its frame at the
    // center of the image.
    let system = tree_system();
    for index in 0..5 {
        let camera = Camera::view_of_index(&system, index).unwrap();
        let scene = build_scene(&system, &camera, f64::from(WIDTH) / f64::from(HEIGHT));
        let image = headless.render(&scene, None).unwrap();
        assert!(
            lit_in(&image, cx - 4, cy - 4, cx + 5, cy + 5) > 0,
            "view of index {index}"
        );
    }
}
