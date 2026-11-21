//! Surface tiles in the far view (`rendering.md` section 7 item 8).
//!
//! A massive frame with surface tiles, seen from 10 and from 100 root
//! extents: both beyond the cells' reach
//! ([`renderer::farfield::CELL_REACH_EXTENTS`]). In views tall enough for
//! it to project to 8 pixels or more it is drawn from its level-0 tiles
//! alone; between 2 and 8 pixels its tiles fade out and its sprite fades in
//! with the weights of the cells; below 2 pixels it is a sprite only.

use gx_core::frames::FrameSystem;
use gx_core::key::SurfaceKey;
use gx_core::matter::{encode, Compression, Sample, Samples, Section, State};
use gx_core::registry::{Frame, FrameTree, Registry, ROOT_PARENT};
use gx_core::surface::{SurfaceTile, HEIGHT_SIDE, PARENT_SIDE, VERTEX_COUNT};
use gx_core::units::{Attenuation, Density, Kelvin, Kilograms, Meters, Quat, Ratio, Seconds, Vec3};
use renderer::app::matter_scene;
use renderer::camera::{look_rotation, Camera, FIELD_OF_VIEW_Y};
use renderer::farfield::{
    cells_out_of_reach, depth_zero_key, matter_radius, sprite_weight, CELLS_ONLY_FROM_PX,
    SPRITE_ONLY_BELOW_PX,
};
use renderer::seen::Seen;
use renderer::stream::{decode_cell, FetchOutcome};
use renderer::world::World;

const ROOT: u64 = 1;
const BODY: u64 = 2;
/// The body's root extent, meters: its cube holds the reference ellipsoid.
const EXTENT: f64 = 1.6e7;
/// Equatorial and polar radii of the tiles' reference ellipsoid, meters.
const RADII: (f64, f64) = (6.0e6, 5.98e6);

fn frame(id: u64, parent: u64, extent: f64, mass: f64, position: Vec3) -> Frame {
    Frame {
        frame_id: id,
        parent_frame_id: parent,
        root_extent: Meters::new(extent),
        max_depth: 4,
        mass: Kilograms::new(mass),
        position,
        velocity: Vec3::zero(),
        orientation: Quat::identity(),
        angular_velocity: Vec3::zero(),
        j2: 0.0,
        reference_radius: Meters::new(0.0),
        principal_moments: Vec3::zero(),
    }
}

/// A massless root around the camera and the body `distance` meters ahead
/// along `+y`.
fn system(distance: f64) -> FrameSystem {
    let frames = vec![
        frame(ROOT, ROOT_PARENT, 1.0e12, 0.0, Vec3::zero()),
        frame(BODY, ROOT, EXTENT, 6.0e24, Vec3::new(0.0, distance, 0.0)),
    ];
    let reg = Registry::new(Seconds::new(0.0), frames).unwrap();
    FrameSystem::from_tree(FrameTree::from_registries(&[reg]).unwrap())
}

/// At the root origin looking along `+y`, `+z` up.
fn camera() -> Camera {
    Camera {
        frame_id: ROOT,
        position: Vec3::zero(),
        orientation: look_rotation(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)),
        speed: 1.0,
        cruise: 1.0,
    }
}

/// The body's depth-0 cell: hot matter filling its middle, so its sprite
/// has light of its own.
fn depth_zero() -> Section {
    let key = depth_zero_key(BODY);
    let g = key.geometry(Meters::new(EXTENT));
    let hot = Sample {
        density: Density::new(1.0),
        state: State::Plasma,
        temperature: Kelvin::new(6000.0),
        albedo: [Ratio::new(0.0); 3],
        roughness: Ratio::new(1.0),
        attenuation: Attenuation::new(1.0),
        radiating_temperature: Kelvin::new(6000.0),
        occupancy: Ratio::new(1.0),
    };
    let samples = Samples::from_fn(4, |x, y, z| {
        if [x, y, z].iter().all(|v| (1..3).contains(v)) {
            hot
        } else {
            Sample::VACUUM
        }
    });
    Section::new(key, g.origin, g.edge, 4, samples).unwrap()
}

/// A smooth tile of `key` with a uniform material, encoded as the hub
/// sends it.
fn tile(key: &SurfaceKey) -> Vec<u8> {
    let tile = SurfaceTile {
        key: *key,
        radius_equatorial: RADII.0,
        radius_polar: RADII.1,
        height_floor: -500.0,
        heights: vec![0.0; HEIGHT_SIDE * HEIGHT_SIDE],
        parent_heights: vec![0.0; PARENT_SIDE * PARENT_SIDE],
        albedo: vec![[40, 90, 160]; VERTEX_COUNT],
        roughness: vec![200; VERTEX_COUNT],
        temperature: vec![290.0; VERTEX_COUNT],
    };
    gx_core::surface::encode(&tile, Compression::None)
}

/// A view tall enough, at `distance`, for the body's matter to project to
/// about `px` pixels.
fn view_for(px: f64, distance: f64) -> (u32, u32) {
    let radius = matter_radius(&depth_zero());
    let height = (px * 2.0 * (0.5 * FIELD_OF_VIEW_Y).tan() * distance / radius).ceil() as u32;
    (height, height)
}

struct Drawn {
    /// Projected size of the body's matter, pixels.
    px: f64,
    /// Tiles of the body drawn.
    tiles: usize,
    /// Every drawn tile is level 0.
    level_zero: bool,
    /// Weight of the body's tiles in the scene.
    tile_weight: Option<f32>,
    /// The body has a sprite.
    sprite: bool,
}

/// Streams the body's depth-0 cell and its tiles for `view` and returns
/// what the scene draws of it.
fn draw(system: &FrameSystem, view: (u32, u32)) -> (Drawn, World) {
    let camera = camera();
    let mut world = World::new(0);
    let cell = vec![encode(&depth_zero(), Compression::None)];
    for round in 0..16 {
        let now = f64::from(round);
        world.select(system, &camera, view, now, true);
        world.select_surfaces(system, &camera, view, now);
        for key in world.take_requests(now) {
            let outcome = if key == depth_zero_key(BODY) {
                decode_cell(&key, &cell, Meters::new(EXTENT))
            } else {
                FetchOutcome::Decoded(None)
            };
            world.complete(key, outcome, now);
        }
        for key in world.take_surface_requests() {
            let sections: Vec<Vec<u8>> = if key.level == 0 {
                vec![tile(&key)]
            } else {
                Vec::new()
            };
            let decoded = renderer::tiles::decode_tile(&key, &sections, &[0], &Default::default());
            world.complete_surface(key, decoded, false);
        }
        world.finish_extraction();
    }
    assert!(world.surfaces().has_surface(BODY));
    let seen = Seen::instant(system, &camera);
    let list = world.draw_list(&seen, view, 99.0);
    let scene = matter_scene(&mut world, &seen, view, 1.0);
    let far = renderer::farfield::far_frames(system, &camera, f64::from(view.1), |_| {
        Some(matter_radius(&depth_zero()))
    });
    let px = far
        .iter()
        .find(|f| f.frame_id == BODY)
        .map_or(f64::INFINITY, |f| f.projected_px);
    let body_tiles: Vec<_> = list
        .tiles
        .iter()
        .filter(|t| t.geometry.key.frame_id == BODY)
        .collect();
    let drawn = Drawn {
        px,
        tiles: body_tiles.len(),
        level_zero: body_tiles.iter().all(|t| t.geometry.key.level == 0),
        tile_weight: scene
            .tiles
            .iter()
            .find(|t| t.tile.geometry.key.frame_id == BODY)
            .map(|t| t.weight),
        sprite: list.sprites.iter().any(|s| s.frame_id == BODY),
    };
    assert_eq!(
        scene.tiles.len(),
        list.tiles.len(),
        "the scene draws the list's tiles"
    );
    (drawn, world)
}

#[test]
fn tiles_are_drawn_from_far_away_and_hand_over_to_the_sprite() {
    for extents in [10.0, 100.0] {
        let distance = extents * EXTENT;
        let system = system(distance);
        // Beyond the cells' reach at both distances.
        let body = *system.tree().get(BODY).unwrap();
        assert!(cells_out_of_reach(&system, &camera(), &body));
        for target in [24.0, 5.0, 1.0] {
            let view = view_for(target, distance);
            let (d, world) = draw(&system, view);
            println!(
                "{extents} extents, view {}: {:.2} px, {} tiles, weight {:?}, sprite {}",
                view.1, d.px, d.tiles, d.tile_weight, d.sprite
            );
            assert!((d.px - target).abs() < 0.1 * target, "{} px", d.px);
            // No cell of the body is drawn: it has a surface.
            assert!(world
                .shown_keys()
                .iter()
                .all(|k| k.frame_id != BODY || k.depth == 0));
            if d.px >= CELLS_ONLY_FROM_PX {
                assert!(d.tiles > 0 && d.level_zero, "level-0 tiles drawn");
                assert_eq!(d.tile_weight, Some(1.0));
                assert!(!d.sprite, "no sprite at {} px", d.px);
            } else if d.px >= SPRITE_ONLY_BELOW_PX {
                // The cells' transition: the same weights.
                let w = sprite_weight(d.px);
                assert!(w > 0.0 && w < 1.0);
                assert!(d.tiles > 0 && d.level_zero);
                let tw = d.tile_weight.unwrap();
                assert!((f64::from(tw) - (1.0 - w)).abs() < 1e-6, "{tw} {w}");
                assert!(d.sprite, "the sprite fades in");
            } else {
                assert_eq!(d.tiles, 0, "no tiles below 2 px");
                assert_eq!(d.tile_weight, None);
                assert!(d.sprite, "a sprite below 2 px");
            }
        }
    }
}
