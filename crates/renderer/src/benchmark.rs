//! The benchmark start (`rendering.md` section 5.1): the frame and
//! direction the benchmark views B1 to B3 are taken on, picked by rule from
//! whatever universe the hub serves, so the views name no frame or place.
//!
//! The rule: the frame is the lowest-id frame that has surface tiles (any
//! of its six level-0 tiles is not empty, section 7 item 1); among its
//! level [`BENCHMARK_LEVEL`] tiles, the one with the largest difference
//! between its highest and lowest height ([`height_span`]), ties broken by the
//! smallest tile key; the direction is that tile's centre
//! ([`tile_centre`], `gx_core::surface` geometry). [`format_start`] writes
//! it as `--start-on` reads it, with 9 decimals.
//!
//! [`print_benchmark_start`] is `gx-renderer --print-benchmark-start`: a
//! developer tool for `scripts/perf-budget.sh`, native only and not part of
//! the API. It connects to the configured hub, waits for the registry and
//! the tiles it needs within [`PICK_DEADLINE`], and returns the line. The
//! same tiles give the same line on every run: nothing in it depends on
//! arrival order.

use crate::config::Config;
use crate::stream::{FetchEvent, Fetcher};
use anyhow::{anyhow, Result};
use gx_core::frames::FrameSystem;
use gx_core::key::SurfaceKey;
use gx_core::surface::{vertex_direction, SurfaceTile, Tile, TILE_QUADS};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

/// The surface level whose tiles the rule compares.
pub const BENCHMARK_LEVEL: u8 = 6;

/// Longest wait for the registry and every tile the pick needs.
pub const PICK_DEADLINE: Duration = Duration::from_secs(120);

/// The message of a universe in which no frame has surface tiles.
pub const NO_SURFACE: &str = "no frame has surface tiles";

/// The difference between the highest and the lowest height of `tile`'s
/// own vertices (`0 ..= TILE_QUADS` both ways, the border left to its
/// neighbors), meters.
pub fn height_span(tile: &SurfaceTile) -> f64 {
    let q = TILE_QUADS as i64;
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for j in 0..=q {
        for i in 0..=q {
            let h = tile.height(i, j);
            lo = lo.min(h);
            hi = hi.max(h);
        }
    }
    f64::from(hi) - f64::from(lo)
}

/// The tile the rule picks among `tiles` (key and [`height_span`]): the largest
/// span, ties broken by the smallest key. `None` for no tiles.
pub fn roughest(tiles: impl IntoIterator<Item = (SurfaceKey, f64)>) -> Option<SurfaceKey> {
    tiles
        .into_iter()
        .fold(None, |best: Option<(SurfaceKey, f64)>, (k, r)| match best {
            Some((bk, br)) if br > r || (br == r && bk < k) => Some((bk, br)),
            _ => Some((k, r)),
        })
        .map(|(k, _)| k)
}

/// The unit direction from the frame origin through the centre of `key`,
/// its middle vertex.
pub fn tile_centre(key: &SurfaceKey) -> [f64; 3] {
    let m = (TILE_QUADS / 2) as i64;
    // Adding zero turns a negative zero into zero, so it prints as one.
    vertex_direction(key, m, m).map(|x| x + 0.0)
}

/// `<frame>:<x>,<y>,<z>` with 9 decimals, as `--start-on` reads it.
pub fn format_start(frame_id: u64, d: [f64; 3]) -> String {
    format!("{frame_id}:{:.9},{:.9},{:.9}", d[0], d[1], d[2])
}

/// The six level-0 tiles of `frame_id`.
fn level_zero(frame_id: u64) -> impl Iterator<Item = SurfaceKey> {
    (0..6).map(move |face| SurfaceKey {
        frame_id,
        face,
        level: 0,
        x: 0,
        y: 0,
    })
}

/// Every tile of `face` of `frame_id` at `level`.
fn face_tiles(frame_id: u64, face: u8, level: u8) -> impl Iterator<Item = SurfaceKey> {
    let n = 1u32 << level;
    (0..n).flat_map(move |y| {
        (0..n).map(move |x| SurfaceKey {
            frame_id,
            face,
            level,
            x,
            y,
        })
    })
}

/// The base surface of an arrived tile: the first non-overlay layer's
/// non-empty tile, decoded, or `None` when every one is empty.
fn base_tile(
    key: &SurfaceKey,
    ev: Result<crate::tiles::DecodedTile, String>,
) -> Result<Option<SurfaceTile>> {
    let decoded = ev.map_err(|e| anyhow!("surface tile {key} is invalid: {e}"))?;
    let Some(bytes) = decoded.sections.first() else {
        return Ok(None);
    };
    match gx_core::surface::decode(key, bytes) {
        Ok(Tile::Surface(t)) => Ok(Some(t)),
        Ok(Tile::Empty(_)) => Ok(None),
        Err(e) => Err(anyhow!(
            "surface tile {key} is invalid: code {}: {}",
            e.code,
            e.reason
        )),
    }
}

/// Requests `keys` and waits until each has arrived or `deadline` passes,
/// returning each key's base tile ([`base_tile`]).
fn fetch_tiles(
    fetcher: &mut Fetcher,
    keys: &BTreeSet<SurfaceKey>,
    deadline: Instant,
    what: &str,
) -> Result<BTreeMap<SurfaceKey, Option<SurfaceTile>>> {
    for k in keys {
        fetcher.request_surface(*k);
    }
    fetcher.flush();
    let mut out = BTreeMap::new();
    while out.len() < keys.len() {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(anyhow!(
                "{what}: {} of {} surface tiles arrived within {} s",
                out.len(),
                keys.len(),
                PICK_DEADLINE.as_secs()
            ));
        }
        for e in fetcher.wait(left.min(Duration::from_millis(200))) {
            if let FetchEvent::Surface { key, outcome, .. } = e {
                // The first arrival has the base; later ones add overlays.
                if keys.contains(&key) && !out.contains_key(&key) {
                    out.insert(key, base_tile(&key, outcome)?);
                }
            }
        }
    }
    Ok(out)
}

/// The rule of the module docs over the universe `fetcher` serves, with
/// `system` its registry: fetches the level-0 tiles of every frame, then
/// the level [`BENCHMARK_LEVEL`] tiles of the first frame with a surface
/// on each face whose level-0 tile is not empty, and returns
/// [`format_start`] of the pick. Fails with [`NO_SURFACE`] when no frame
/// has surface tiles, and with a message when a tile is invalid or the
/// tiles have not all arrived by `deadline`.
pub fn benchmark_start(
    fetcher: &mut Fetcher,
    system: &FrameSystem,
    deadline: Instant,
) -> Result<String> {
    // Overlays give material only (section 7 item 6): hidden, they are not
    // fetched.
    let overlays = fetcher.client().build().overlay_layers();
    fetcher.set_hidden_layers(overlays);
    let frames: BTreeSet<u64> = system.tree().frames().iter().map(|f| f.frame_id).collect();
    let roots: BTreeSet<SurfaceKey> = frames.iter().flat_map(|&f| level_zero(f)).collect();
    let roots = fetch_tiles(fetcher, &roots, deadline, "level 0")?;
    let Some(frame_id) = frames
        .iter()
        .copied()
        .find(|&f| level_zero(f).any(|k| roots[&k].is_some()))
    else {
        return Err(anyhow!(NO_SURFACE));
    };
    let keys: BTreeSet<SurfaceKey> = level_zero(frame_id)
        .filter(|k| roots[k].is_some())
        .flat_map(|k| face_tiles(frame_id, k.face, BENCHMARK_LEVEL))
        .collect();
    let tiles = fetch_tiles(
        fetcher,
        &keys,
        deadline,
        &format!("frame {frame_id} level {BENCHMARK_LEVEL}"),
    )?;
    let pick = roughest(
        tiles
            .iter()
            .filter_map(|(k, t)| t.as_ref().map(|t| (*k, height_span(t)))),
    )
    .ok_or_else(|| anyhow!("frame {frame_id} has no surface tile at level {BENCHMARK_LEVEL}"))?;
    Ok(format_start(frame_id, tile_centre(&pick)))
}

/// `gx-renderer --print-benchmark-start`: connects to the hub of `config`
/// with its cache and workers, and returns [`benchmark_start`]'s line,
/// everything within [`PICK_DEADLINE`].
pub fn print_benchmark_start(config: &Config) -> Result<String> {
    let deadline = Instant::now() + PICK_DEADLINE;
    let mut c = crate::api::create::connect(config, PICK_DEADLINE)?;
    benchmark_start(&mut c.fetcher, &c.system, deadline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(face: u8, x: u32, y: u32) -> SurfaceKey {
        SurfaceKey::new(3, face, BENCHMARK_LEVEL, x, y).unwrap()
    }

    #[test]
    fn the_roughest_tile_wins_and_ties_go_to_the_smallest_key() {
        let tiles = [
            (key(2, 5, 1), 40.0),
            (key(1, 9, 9), 70.0),
            (key(0, 3, 4), 70.0),
            (key(0, 0, 0), 10.0),
        ];
        assert_eq!(roughest(tiles), Some(key(0, 3, 4)));
        assert_eq!(roughest(tiles.iter().rev().copied()), Some(key(0, 3, 4)));
        assert_eq!(roughest([]), None);
    }

    #[test]
    fn the_line_has_nine_decimals_and_no_negative_zero() {
        let k = key(4, 0, 63);
        let d = tile_centre(&k);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-12);
        assert_eq!(
            format_start(7, [0.5, -0.25, 0.0]),
            "7:0.500000000,-0.250000000,0.000000000"
        );
        assert!(!format_start(7, tile_centre(&key(0, 31, 31))).contains("-0.000000000"));
    }
}
