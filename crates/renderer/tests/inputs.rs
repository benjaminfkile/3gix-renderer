//! Layers that read other layers, against the mock hub
//! (`client-compute.md` sections 2.8, 10.3, and 10.4).
//!
//! The space has the shared scene's registry layer (`frames`), the core
//! library's version 1 example module on every frame (`matter`), and the
//! version 2 example module (`reader`) on the root frame, which reads the
//! root frame's record (a frames slot) and `matter` (a layer slot): for a
//! cell the same cell and its parent. A third layer (`reader2`) runs the
//! same version 2 module over `reader` instead of `matter`. Their
//! addresses are version 2, and their uploads and checks carry
//! `X-Inputs-Sha256`.

mod common;

use gx_core::key::CellKey;
use renderer::config::ApiKey;
use renderer::hub::HubClient;
use renderer::mock_hub::{
    example_matter_layer, registry_layer, MockHub, MockHubOptions, MockLayer, MOCK_API_KEY,
    MOCK_SPACE_ID,
};
use renderer::protocol::{Attachment, InputSlot, Routes};
use renderer::stream::{FetchEvent, FetchOutcome, Fetcher, FetcherOptions};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

const EXAMPLE_MODULE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/example-module/module.wasm"
));

const READER_MODULE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../testdata/example-module-v2/module.wasm"
));

/// The frame record of `frame_id` in the test registry.
fn record(frame_id: u64) -> Vec<u8> {
    let r = common::registry();
    let f = r
        .frames()
        .iter()
        .find(|f| f.frame_id == frame_id)
        .expect("a frame of the registry");
    gx_core::frame_record::encode(&gx_core::frame_record::FrameRecord::from_registry_frame(f))
}

/// A version 2 layer on the root frame up to depth 1 reading the root
/// frame's record and the layer `reads`.
fn reader_layer(id: &str, reads: &str) -> MockLayer {
    MockLayer {
        id: id.into(),
        module: READER_MODULE.to_vec(),
        manifest: None,
        attachment: Attachment::Frames(vec![common::ROOT]),
        exclusive: false,
        max_depth: [(common::ROOT.to_string(), 1)].into(),
        span_seconds: 0,
        uses_origin: false,
        stored: None,
        shared: true,
        inputs: vec![
            InputSlot::Frames {
                layer: "frames".into(),
                frames: vec![common::ROOT],
            },
            InputSlot::Layer(reads.into()),
        ],
    }
}

/// The space with `readers` reader layers chained after `matter`: 1 for
/// the two-layer chain, 2 for the three-layer one.
fn space(readers: usize) -> MockHubOptions {
    let (matter, tiles) = example_matter_layer(
        "matter",
        EXAMPLE_MODULE,
        &[
            (common::ROOT, common::ROOT_EXTENT),
            (common::CHILD, common::CHILD_EXTENT),
        ],
        1,
        20.0,
        6000.0,
    );
    let mut layers = vec![registry_layer("frames", &common::registry_bytes()), matter];
    let mut reads = "matter".to_string();
    for i in 0..readers {
        let id = if i == 0 {
            "reader".to_string()
        } else {
            format!("reader{}", i + 1)
        };
        layers.push(reader_layer(&id, &reads));
        reads = id;
    }
    let mut options = MockHubOptions::new(layers, tiles);
    options.frame_records = vec![
        ("frames".into(), common::ROOT, record(common::ROOT)),
        ("frames".into(), common::CHILD, record(common::CHILD)),
    ];
    options
}

/// The root frame's root cell and one of its children.
fn cells() -> Vec<CellKey> {
    vec![
        CellKey::new(common::ROOT, 0, 0, 0, 0).unwrap(),
        CellKey::new(common::ROOT, 1, 1, 0, 1).unwrap(),
    ]
}

fn connect(rt: &tokio::runtime::Runtime, hub: &MockHub, threads: usize) -> Fetcher {
    let client = rt
        .block_on(HubClient::open(
            Routes::new(&hub.url(), None, MOCK_SPACE_ID),
            ApiKey::new(MOCK_API_KEY),
            None,
        ))
        .expect("the mock hub answers");
    let mut fetcher = Fetcher::new(
        rt.handle().clone(),
        Arc::new(client),
        FetcherOptions {
            cache: None,
            worker_threads: threads,
            ..FetcherOptions::default()
        },
    )
    .unwrap();
    fetcher.load_registry(Duration::from_secs(60)).unwrap();
    fetcher
}

/// Requests `cells` and waits until each has arrived with matter, then
/// until every upload is answered.
fn fetch(fetcher: &mut Fetcher, cells: &[CellKey]) {
    for c in cells {
        fetcher.request(*c, gx_core::units::Meters::new(common::ROOT_EXTENT));
    }
    let mut left: BTreeSet<CellKey> = cells.iter().copied().collect();
    let start = Instant::now();
    while !left.is_empty() {
        assert!(start.elapsed() < Duration::from_secs(120), "cells hang");
        for e in fetcher.wait(Duration::from_millis(50)) {
            if let FetchEvent::Completed { key, outcome, .. } = e {
                match outcome {
                    FetchOutcome::Decoded(Some(_)) => {}
                    other => panic!("{key}: {other:?}"),
                }
                left.remove(&key);
            }
        }
    }
    while fetcher.stats().worker.uploads_pending > 0 {
        assert!(start.elapsed() < Duration::from_secs(120), "uploads hang");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Every layer's section of every cell is computed once, even though the
/// root cell's `matter` section is wanted by the view and read by both
/// readers, and the next client streams them all.
fn chain(readers: usize) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space(readers))).unwrap();
    let mut a = connect(&rt, &hub, 2);
    fetch(&mut a, &cells());
    // Every layer's section of each cell, and of the registry key, which
    // goes to every layer (the `frames` layer included).
    let keys = cells().len() + 1;
    let sections = (keys * (1 + readers) + 1) as u64;
    let w = a.stats().worker;
    assert_eq!(w.sections_computed, sections, "each computed once: {w:?}");
    let h = hub.counts();
    assert_eq!(h.uploads_stored, sections, "{h:?}");
    assert_eq!(
        h.uploads_matched + h.uploads_refused,
        0,
        "offered once: {h:?}"
    );
    assert_eq!(
        h.inputs_headers,
        (keys * readers) as u64,
        "every reader upload names its inputs: {h:?}"
    );
    let stored = hub.sections();

    let mut b = connect(&rt, &hub, 2);
    fetch(&mut b, &cells());
    assert_eq!(b.stats().worker.module_runs, 0, "the next client streams");
    assert_eq!(hub.sections(), stored);
}

#[test]
fn a_two_layer_chain_computes_each_section_once() {
    chain(1);
}

#[test]
fn a_three_layer_chain_resolves_inputs_recursively() {
    chain(2);
}

/// A hub that holds other inputs as true answers `409`: counted, never
/// retried, and the cells still draw.
#[test]
fn a_409_is_counted_and_not_retried() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space(1))).unwrap();
    hub.refuse_inputs();
    let mut a = connect(&rt, &hub, 2);
    fetch(&mut a, &cells());
    let h = hub.counts();
    let w = a.stats().worker;
    // The reader's sections of both cells and of the registry key.
    let keys = cells().len() as u64 + 1;
    assert_eq!(h.inputs_refused, keys, "{h:?}");
    assert_eq!(w.inputs_conflicts, h.inputs_refused, "{w:?}");
    assert_eq!(w.uploads_failed, 0, "{w:?}");
    assert_eq!(h.uploads_stored, keys + 1, "matter and frames only: {h:?}");
}

/// A downloaded section of a layer with inputs is checked by recomputing
/// it with its inputs resolved the same way (`client-compute.md` sections
/// 2.6 and 10.3), and the report names its inputs (10.4).
#[test]
fn checks_of_a_layer_with_inputs_resolve_their_inputs() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let hub = rt.block_on(MockHub::start(space(1))).unwrap();
    let mut a = connect(&rt, &hub, 2);
    fetch(&mut a, &cells());
    let before = hub.counts();
    hub.set_check_rate(Some(1.0));
    let mut b = connect(&rt, &hub, 2);
    fetch(&mut b, &cells());
    let start = Instant::now();
    loop {
        let w = b.stats().worker;
        let answered = w.checks_matched
            + w.checks_disputed
            + w.checks_failed
            + w.checks_skipped
            + w.inputs_conflicts;
        if w.checks_queued > 0 && answered >= w.checks_queued {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "checks hang: {w:?}"
        );
        b.fetcher_wait();
    }
    let w = b.stats().worker;
    let h = hub.counts();
    assert_eq!(w.checks_matched, w.checks_queued, "{w:?}");
    assert_eq!(h.checks_disputed, 0, "{h:?}");
    let reader_checks = h.inputs_headers - before.inputs_headers;
    assert!(reader_checks >= 1, "a reader section was checked: {h:?}");
}

trait Wait {
    fn fetcher_wait(&mut self);
}

impl Wait for Fetcher {
    fn fetcher_wait(&mut self) {
        let _ = self.wait(Duration::from_millis(20));
    }
}
