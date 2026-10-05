//! Mock hub test: the client loads the registry through a `202`, a
//! `chunkReady` push, and a `200`, and ends up with a valid frame system;
//! matter cells stream through the same cycle into the cell cache.

mod common;

use gx_core::container::encode_chunk;
use renderer::config::ApiKey;
use renderer::hub::{load_registry, ChunkFetch, HubClient, ReadyPolicy};
use renderer::mock_hub::{MockHub, MockHubOptions, MOCK_API_KEY, MOCK_BUILD_ID, MOCK_SPACE_ID};
use renderer::stream::{CellCache, CellState, FetchEvent, Fetcher, SelectedCell};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn registry_chunk() -> Vec<u8> {
    let tree = include_bytes!("data/tree.bin");
    let empty = include_bytes!("data/empty.bin");
    // Two layers: one declares the frames, one declares none.
    encode_chunk(&[tree, empty], &["layer-a", "layer-b"])
}

/// A policy that would fail the test if the client fell back to polling:
/// the push must arrive long before the fallback.
fn push_only() -> ReadyPolicy {
    ReadyPolicy {
        fallback_after: Duration::from_secs(30),
        poll_interval: Duration::from_secs(30),
        deadline: Duration::from_secs(20),
    }
}

#[tokio::test]
async fn registry_loads_through_202_push_and_200() {
    let hub = MockHub::start(MockHubOptions::new(registry_chunk()))
        .await
        .unwrap();
    let client = HubClient::new(
        &hub.url(),
        ApiKey::new(MOCK_API_KEY),
        MOCK_SPACE_ID,
        MOCK_BUILD_ID,
    )
    .unwrap();
    let system = load_registry(&client, push_only()).await.unwrap();
    assert_eq!(system.tree().frames().len(), 6);
    assert_eq!(system.tree().root().frame_id, 1);
    assert_eq!(hub.registry_requests(), 2, "one 202 and one 200");
    assert_eq!(hub.ready_pushes(), 1);

    // The 200 body is never requested again in this session.
    assert!(matches!(
        client.fetch_chunk("registry").await,
        ChunkFetch::Ready(_)
    ));
    assert_eq!(hub.registry_requests(), 2);
}

#[tokio::test]
async fn active_build_is_resolved_from_the_listing() {
    let hub = MockHub::start(MockHubOptions::new(registry_chunk()))
        .await
        .unwrap();
    let client = HubClient::new(&hub.url(), ApiKey::new(MOCK_API_KEY), MOCK_SPACE_ID, "").unwrap();
    assert_eq!(client.active_build().await.unwrap(), MOCK_BUILD_ID);
}

#[tokio::test]
async fn polling_fallback_without_a_socket() {
    let mut options = MockHubOptions::new(registry_chunk());
    options.pending_responses = 2;
    let hub = MockHub::start(options).await.unwrap();
    let client = HubClient::new(
        &hub.url(),
        ApiKey::new(MOCK_API_KEY),
        MOCK_SPACE_ID,
        MOCK_BUILD_ID,
    )
    .unwrap();
    let policy = ReadyPolicy {
        fallback_after: Duration::from_millis(100),
        poll_interval: Duration::from_millis(50),
        deadline: Duration::from_secs(10),
    };
    let bytes = client
        .wait_for_chunk("registry", policy, None)
        .await
        .unwrap();
    assert_eq!(&bytes[..], &registry_chunk()[..]);
    assert_eq!(hub.registry_requests(), 3);
}

#[tokio::test]
async fn statuses_and_errors() {
    let hub = MockHub::start(MockHubOptions::new(registry_chunk()))
        .await
        .unwrap();
    let client = HubClient::new(
        &hub.url(),
        ApiKey::new(MOCK_API_KEY),
        MOCK_SPACE_ID,
        MOCK_BUILD_ID,
    )
    .unwrap();
    assert_eq!(client.fetch_chunk("1-0-0-0-0").await, ChunkFetch::NotFound);
    let wrong = HubClient::new(
        &hub.url(),
        ApiKey::new("wrong"),
        MOCK_SPACE_ID,
        MOCK_BUILD_ID,
    )
    .unwrap();
    assert!(matches!(
        wrong.fetch_chunk("registry").await,
        ChunkFetch::Error(_)
    ));
}

#[tokio::test]
async fn never_ready_times_out_and_invalid_registry_is_rejected() {
    let mut options = MockHubOptions::new(registry_chunk());
    options.pending_responses = u32::MAX;
    let hub = MockHub::start(options).await.unwrap();
    let client = HubClient::new(
        &hub.url(),
        ApiKey::new(MOCK_API_KEY),
        MOCK_SPACE_ID,
        MOCK_BUILD_ID,
    )
    .unwrap();
    let policy = ReadyPolicy {
        fallback_after: Duration::from_millis(50),
        poll_interval: Duration::from_millis(50),
        deadline: Duration::from_millis(400),
    };
    let e = load_registry(&client, policy)
        .await
        .unwrap_err()
        .to_string();
    assert!(e.contains("did not become ready"), "{e}");

    let bad = encode_chunk(
        &[
            include_bytes!("data/tree.bin"),
            include_bytes!("data/tree.bin"),
        ],
        &["a", "b"],
    );
    let mut options = MockHubOptions::new(bad);
    options.pending_responses = 0;
    let hub = MockHub::start(options).await.unwrap();
    let client = HubClient::new(
        &hub.url(),
        ApiKey::new(MOCK_API_KEY),
        MOCK_SPACE_ID,
        MOCK_BUILD_ID,
    )
    .unwrap();
    let e = load_registry(&client, push_only())
        .await
        .unwrap_err()
        .to_string();
    assert!(e.starts_with("registry is invalid"), "{e}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cells_stream_through_202_push_and_200() {
    let chunks = common::chunks();
    let mut options = MockHubOptions::new(common::registry_chunk());
    for (key, chunk) in &chunks {
        options = options.with_cell(key.to_string(), chunk.clone());
    }
    let hub = MockHub::start(options).await.unwrap();
    let client = Arc::new(
        HubClient::new(
            &hub.url(),
            ApiKey::new(MOCK_API_KEY),
            MOCK_SPACE_ID,
            MOCK_BUILD_ID,
        )
        .unwrap(),
    );
    let system = load_registry(&client, push_only()).await.unwrap();
    assert_eq!(system.tree().frames().len(), 2);

    let keys = [common::hot_key(), common::solid_key(), common::empty_key()];
    let handle = tokio::runtime::Handle::current();
    let (cache, fetcher) = tokio::task::spawn_blocking(move || {
        let mut fetcher = Fetcher::new(handle, client);
        let mut cache = CellCache::new();
        let start = Instant::now();
        let selection = keys
            .iter()
            .map(|&key| SelectedCell {
                key,
                projected_px: 100.0,
            })
            .collect();
        cache.set_selection(selection, 0.0);
        // Well inside the 10 s before the cache would poll: the pushes must
        // bring the cells in.
        while start.elapsed() < Duration::from_secs(8) {
            let now = start.elapsed().as_secs_f64();
            for key in cache.take_requests(now) {
                let extent = system.tree().get(key.frame_id).unwrap().root_extent;
                fetcher.request(key, extent);
            }
            for event in fetcher.wait(Duration::from_millis(20)) {
                match event {
                    FetchEvent::Completed { key, outcome, .. } => cache.complete(key, outcome, now),
                    FetchEvent::ChunkReady(key) => {
                        cache.notify_ready(&key);
                    }
                }
            }
            let done = keys.iter().all(|k| {
                matches!(
                    cache.state(k),
                    Some(CellState::Ready(_) | CellState::Empty | CellState::Gone)
                )
            });
            if done {
                break;
            }
        }
        (cache, fetcher)
    })
    .await
    .unwrap();

    let hot = cache.state(&common::hot_key());
    let solid = cache.state(&common::solid_key());
    assert!(matches!(hot, Some(CellState::Ready(_))), "{hot:?}");
    assert!(matches!(solid, Some(CellState::Ready(_))), "{solid:?}");
    assert_eq!(cache.state(&common::empty_key()), Some(&CellState::Empty));
    if let Some(CellState::Ready(cell)) = hot {
        assert!(cell.emitter.is_some(), "the hot cell carries an emitter");
    }
    if let Some(CellState::Ready(cell)) = solid {
        assert!(cell.emitter.is_none(), "300 K does not emit light");
        assert_eq!(*cell.section, common::solid_section());
    }
    for key in keys {
        assert_eq!(
            hub.chunk_requests(&key.to_string()),
            2,
            "{key}: one 202, one 200"
        );
    }
    // One push for the registry and one per cell.
    assert_eq!(hub.ready_pushes(), 4);
    let total: usize = chunks.values().map(Vec::len).sum();
    assert_eq!(fetcher.bytes_fetched(), total as u64);
    assert!(fetcher.last_round_trip().is_some());
    let counts = cache.counts();
    assert_eq!((counts.selected, counts.ready, counts.pending), (3, 3, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn headless_run_streams_cells_from_the_hub() {
    let mut options = MockHubOptions::new(common::registry_chunk());
    for (key, chunk) in common::chunks() {
        options = options.with_cell(key.to_string(), chunk);
    }
    let hub = MockHub::start(options).await.unwrap();
    let client = Arc::new(
        HubClient::new(
            &hub.url(),
            ApiKey::new(MOCK_API_KEY),
            MOCK_SPACE_ID,
            MOCK_BUILD_ID,
        )
        .unwrap(),
    );
    let system = load_registry(&client, push_only()).await.unwrap();
    let handle = tokio::runtime::Handle::current();
    let image = tokio::task::spawn_blocking(move || {
        let mut fetcher = Fetcher::new(handle, client);
        let mut sim = renderer::sim::Simulation::new(
            system,
            renderer::sim::SimClock::new(gx_core::units::Seconds::new(0.0), 1.0),
        );
        renderer::app::render_headless_frame(&mut sim, Some(&mut fetcher), 320, 180)
    })
    .await
    .unwrap()
    .expect("a wgpu adapter, software is fine");
    // Both depth 0 cells were fetched through a 202 and a 200.
    assert_eq!(hub.chunk_requests(&common::hot_key().to_string()), 2);
    assert_eq!(hub.chunk_requests(&common::solid_key().to_string()), 2);
    // The hot blob sits at the root origin, the center of the Home view.
    let p = image.pixel(160, 90);
    assert!(p[0] > 150 && p[1] > 150 && p[2] > 150, "{p:?}");
}
