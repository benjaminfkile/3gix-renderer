//! Mock hub test: the client loads the registry through a `202`, a
//! `chunkReady` push, and a `200`, and ends up with a valid frame system.

use gx_core::container::encode_chunk;
use renderer::config::ApiKey;
use renderer::hub::{load_registry, ChunkFetch, HubClient, ReadyPolicy};
use renderer::mock_hub::{MockHub, MockHubOptions, MOCK_API_KEY, MOCK_BUILD_ID, MOCK_SPACE_ID};
use std::time::Duration;

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
