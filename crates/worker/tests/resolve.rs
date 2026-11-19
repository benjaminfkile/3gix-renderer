//! The shared input resolver (`client-compute.md` sections 2.8, 10.3, and
//! 10.4) against a fake platform: two and three layer chains, one flight
//! per address, `409`, and checks. Both builds drive this same code.

use gx_core::address::{section_address, section_address_v2, SlotKind, SlotValue};
use gx_core::key::ChunkKey;
use gx_core::matter::{encode, Compression, Section};
use gx_core::module_io::{inputs_sha256, ModuleRequest};
use gx_core::sha256::sha256;
use gx_core::units::Meters;
use gx_worker::resolve::block_on;
use gx_worker::{
    CheckOutcome, ComputeJob, Computed, InputDirection, InputResolver, Platform, UploadOutcome,
    WorkerError,
};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

const KEY: &str = "1-0-0-0-0";

fn key() -> ChunkKey {
    KEY.parse().unwrap()
}

/// A valid section for the test key: an empty cell section.
fn section_bytes() -> Vec<u8> {
    let ChunkKey::Cell(cell) = key() else {
        unreachable!()
    };
    let g = cell.geometry(Meters::new(1000.0));
    encode(
        &Section::empty(cell, g.origin, g.edge).unwrap(),
        Compression::None,
    )
}

/// The job of `layer` at the test key. `reads` names the layer its one
/// layer slot reads, if any.
fn job(layer: &str, reads: Option<&str>) -> ComputeJob {
    let request = ModuleRequest {
        origin_utc_ms: 0,
        span_seconds: 0,
        span_index: 0,
        key: key(),
    };
    let module = sha256(layer.as_bytes());
    let slots: Vec<SlotValue> = reads
        .map(|r| SlotValue {
            kind: SlotKind::Layer,
            value: sha256(r.as_bytes()),
        })
        .into_iter()
        .collect();
    let address = if slots.is_empty() {
        section_address(&module, None, 0, 0, 0, &request.key)
    } else {
        section_address_v2(&module, None, 0, 0, 0, &slots, &request.key).unwrap()
    };
    ComputeJob {
        address,
        layer: layer.into(),
        share: true,
        module,
        manifest: None,
        request,
        frames_per_slot: vec![0; slots.len()],
        slots,
        frames: Vec::new(),
        inputs_sha256: None,
    }
}

/// Yields once, so a concurrent future gets polled in between.
struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            return Poll::Ready(());
        }
        self.0 = true;
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

/// Runs two futures together on this thread.
fn join<A: Future, B: Future>(a: A, b: B) -> (A::Output, B::Output) {
    let mut a = Box::pin(a);
    let mut b = Box::pin(b);
    let (mut ra, mut rb) = (None, None);
    block_on(std::future::poll_fn(|cx| {
        if ra.is_none() {
            if let Poll::Ready(v) = a.as_mut().poll(cx) {
                ra = Some(v);
            }
        }
        if rb.is_none() {
            if let Poll::Ready(v) = b.as_mut().poll(cx) {
                rb = Some(v);
            }
        }
        if ra.is_some() && rb.is_some() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }));
    (ra.unwrap(), rb.unwrap())
}

/// Layers by id: the layer each reads (none for a base layer) and what
/// directions say for it. Counts every call.
#[derive(Default)]
struct Fake {
    reads: BTreeMap<String, String>,
    directions: BTreeMap<String, InputDirection>,
    streams: RefCell<BTreeMap<[u8; 32], usize>>,
    computes: RefCell<BTreeMap<String, usize>>,
    offered: RefCell<Vec<ComputeJob>>,
    cache: RefCell<BTreeMap<[u8; 32], Vec<u8>>>,
    raised: RefCell<Vec<f64>>,
    inputs_computed: Cell<usize>,
}

impl Fake {
    fn chain(layers: &[&str], direction: InputDirection) -> Fake {
        let mut f = Fake::default();
        for w in layers.windows(2) {
            f.reads.insert(w[1].into(), w[0].into());
        }
        for l in layers {
            f.directions.insert((*l).into(), direction.clone());
        }
        f
    }

    fn job(&self, layer: &str) -> ComputeJob {
        job(layer, self.reads.get(layer).map(String::as_str))
    }
}

impl Platform for Fake {
    fn input_job(
        &self,
        reader: &ComputeJob,
        slot: u32,
        key: ChunkKey,
    ) -> Result<ComputeJob, WorkerError> {
        assert_eq!(slot, 0);
        assert_eq!(key, self::key());
        Ok(self.job(&self.reads[&reader.layer]))
    }

    async fn direct(&self, job: &ComputeJob) -> Result<InputDirection, WorkerError> {
        Ok(self.directions[&job.layer].clone())
    }

    async fn stream(&self, address: &[u8; 32], _: &[u8; 32]) -> Result<Vec<u8>, WorkerError> {
        *self.streams.borrow_mut().entry(*address).or_default() += 1;
        YieldOnce(false).await;
        Ok(section_bytes())
    }

    async fn input_list(&self, job: &ComputeJob) -> Result<Vec<(u32, ChunkKey)>, WorkerError> {
        Ok(match self.reads.get(&job.layer) {
            Some(_) => vec![(0, key())],
            None => Vec::new(),
        })
    }

    async fn compile(
        &self,
        job: &ComputeJob,
        inputs: &[Arc<[u8]>],
    ) -> Result<Computed, WorkerError> {
        assert_eq!(inputs.len(), job.slots.len());
        *self
            .computes
            .borrow_mut()
            .entry(job.layer.clone())
            .or_default() += 1;
        YieldOnce(false).await;
        Ok(Computed {
            bytes: Arc::from(section_bytes()),
            section: None,
            inputs_sha256: None,
        })
    }

    async fn cache_get(&self, address: &[u8; 32]) -> Option<Vec<u8>> {
        self.cache.borrow().get(address).cloned()
    }

    async fn cache_put(&self, address: &[u8; 32], bytes: &Arc<[u8]>) {
        self.cache.borrow_mut().insert(*address, bytes.to_vec());
    }

    fn offer(&self, job: &ComputeJob, _: Arc<[u8]>) {
        self.offered.borrow_mut().push(job.clone());
    }

    fn input_computed(&self) {
        self.inputs_computed.set(self.inputs_computed.get() + 1);
    }

    fn raise(&self, _: &ChunkKey, _: &[u8; 32], priority: f64) {
        self.raised.borrow_mut().push(priority);
    }
}

fn ready() -> InputDirection {
    InputDirection::Ready {
        sha256: sha256(&section_bytes()),
    }
}

/// Two layers, the input stored: streamed once, cached, and the inputs
/// hash is over exactly that input.
#[test]
fn two_layer_chain_streams_its_input() {
    let fake = Fake::chain(&["base", "reader"], ready());
    let r = InputResolver::new();
    let out = block_on(r.compute(&fake, &fake.job("reader"), 1.0)).unwrap();
    let input = section_bytes();
    assert_eq!(out.inputs_sha256, Some(inputs_sha256(&[&input])));
    assert_eq!(fake.streams.borrow().values().sum::<usize>(), 1);
    assert!(fake.cache.borrow().contains_key(&fake.job("base").address));
    assert_eq!(fake.computes.borrow().get("base"), None);
    // Again: from memory, nothing streamed.
    block_on(r.compute(&fake, &fake.job("reader"), 1.0)).unwrap();
    assert_eq!(fake.streams.borrow().values().sum::<usize>(), 1);
    // A new session with the same cache: from the cache.
    let r = InputResolver::new();
    block_on(r.compute(&fake, &fake.job("reader"), 1.0)).unwrap();
    assert_eq!(fake.streams.borrow().values().sum::<usize>(), 1);
}

/// Three layers, nothing stored: the middle and base inputs are computed
/// recursively, each offered once with its own inputs hash, base first.
#[test]
fn three_layer_chain_computes_and_offers_missing_inputs() {
    let fake = Fake::chain(
        &["base", "middle", "top"],
        InputDirection::Compute { share: true },
    );
    let r = InputResolver::new();
    let out = block_on(r.compute(&fake, &fake.job("top"), 1.0)).unwrap();
    let input = section_bytes();
    assert_eq!(out.inputs_sha256, Some(inputs_sha256(&[&input])));
    let offered = fake.offered.borrow();
    let layers: Vec<&str> = offered.iter().map(|j| j.layer.as_str()).collect();
    assert_eq!(layers, ["base", "middle"]);
    assert_eq!(offered[0].inputs_sha256, None);
    assert_eq!(offered[1].inputs_sha256, Some(inputs_sha256(&[&input])));
    assert_eq!(fake.inputs_computed.get(), 2);
    for l in ["base", "middle", "top"] {
        assert_eq!(fake.computes.borrow()[l], 1, "{l}");
    }
}

/// A `local` input is computed and kept, never offered.
#[test]
fn local_inputs_are_never_offered() {
    let fake = Fake::chain(&["base", "top"], InputDirection::Compute { share: false });
    block_on(InputResolver::new().compute(&fake, &fake.job("top"), 1.0)).unwrap();
    assert!(fake.offered.borrow().is_empty());
}

/// Two readers of one input at once: one stream, and the waiter raised
/// the work to its priority. The wanted compute of the input's own
/// address is handed the same bytes and not computed again.
#[test]
fn one_flight_per_address() {
    let mut fake = Fake::chain(&["base", "reader"], ready());
    fake.reads.insert("other".into(), "base".into());
    fake.directions.insert("other".into(), ready());
    let r = InputResolver::new();
    let (a, b) = join(
        r.run_wanted(&fake, &fake.job("reader"), 1.0),
        r.run_wanted(&fake, &fake.job("other"), 5.0),
    );
    assert!(a.unwrap().1 && b.unwrap().1);
    assert_eq!(
        fake.streams.borrow().values().copied().collect::<Vec<_>>(),
        [1]
    );
    assert_eq!(*fake.raised.borrow(), [5.0]);
    // The base's own wanted compute takes the streamed bytes.
    let (base, fresh) = block_on(r.run_wanted(&fake, &fake.job("base"), 1.0)).unwrap();
    assert!(!fresh);
    assert_eq!(&*base.bytes, &section_bytes()[..]);
    assert_eq!(fake.computes.borrow().get("base"), None);

    // The same for a computed input: two wanted computes of one address
    // run it once.
    let fake = Fake::chain(&["base"], InputDirection::Compute { share: true });
    let r = InputResolver::new();
    let (a, b) = join(
        r.run_wanted(&fake, &fake.job("base"), 1.0),
        r.run_wanted(&fake, &fake.job("base"), 2.0),
    );
    assert!(a.unwrap().1);
    assert!(!b.unwrap().1);
    assert_eq!(fake.computes.borrow()["base"], 1);
}

/// A platform's own stream (the renderer's) shares the flight: an input
/// streamed under it is not downloaded a second time.
#[test]
fn a_renderer_stream_shares_the_flight() {
    let fake = Fake::chain(&["base", "reader"], ready());
    let r = InputResolver::new();
    let base = fake.job("base").address;
    let renderer = async {
        let _f = r.flights().acquire(&base, 1.0, &|_| {}).await;
        let bytes = fake.stream(&base, &[0; 32]).await.unwrap();
        r.remember(&base, Arc::from(bytes));
    };
    let (_, out) = join(renderer, r.compute(&fake, &fake.job("reader"), 1.0));
    out.unwrap();
    assert_eq!(fake.streams.borrow()[&base], 1);
}

/// A check recomputes through the same resolution and gets the same inputs
/// hash, so its report carries the right `X-Inputs-Sha256`; `409` (the hub
/// lacks the inputs) is its own outcome for uploads and checks.
#[test]
fn checks_and_409() {
    let fake = Fake::chain(&["base", "reader"], InputDirection::Compute { share: true });
    let r = InputResolver::new();
    let (wanted, _) = block_on(r.run_wanted(&fake, &fake.job("reader"), 1.0)).unwrap();
    let check = block_on(InputResolver::new().compute(&fake, &fake.job("reader"), 0.0)).unwrap();
    assert_eq!(check.inputs_sha256, wanted.inputs_sha256);
    assert!(check.inputs_sha256.is_some());
    assert_eq!(UploadOutcome::from_status(409), UploadOutcome::Conflict);
    assert_eq!(CheckOutcome::from_status(409), CheckOutcome::Conflict);
}

/// An input that does not hash to what directions said fails, and so does
/// a module naming a slot that is not a layer slot.
#[test]
fn bad_inputs_fail() {
    let fake = Fake::chain(
        &["base", "reader"],
        InputDirection::Ready { sha256: [9; 32] },
    );
    assert!(block_on(InputResolver::new().compute(&fake, &fake.job("reader"), 1.0)).is_err());
    let fake = Fake::chain(&["base"], ready());
    let mut j = fake.job("base");
    j.layer = "base".into();
    let mut fake = fake;
    fake.reads.insert("base".into(), "base".into());
    // `base` now names an input, but its job has no layer slot.
    assert!(block_on(InputResolver::new().compute(&fake, &j, 1.0)).is_err());
}
