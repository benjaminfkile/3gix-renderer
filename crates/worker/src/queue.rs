//! The worker's work list (`client-compute.md` section 2.8): the order in
//! which waiting work starts, shared by the native pool
//! ([`crate::WorkerPool`]) and the browser fetcher.
//!
//! Work comes in three classes, strictly ordered ([`Lane`]): wanted
//! computes, then uploads, then checks. A lower class starts only when no
//! higher class is waiting. Within the compute lane the job of the
//! highest priority starts first (ties in the order they were queued).
//! When the wanted set changes the driver calls
//! [`ComputeQueue::reprioritize`]: every waiting job takes its key's new
//! priority, and a job whose key is no longer wanted is removed before it
//! starts. A job already taken from the queue is not affected: it finishes
//! and its section is cached.
//!
//! Uploads never hold a compute thread. A native pool thread that takes an
//! upload only hands its request to the fetcher's async runtime and is free
//! at once; at most [`crate::MAX_UPLOADS_IN_FLIGHT`] are in flight, and
//! while that many are the upload lane gives nothing and keeps the rest
//! (bounded, oldest dropped). In the browser uploads are async `fetch`
//! calls on the page and never occupy the Web Worker.

use gx_core::key::ChunkKey;

/// The three classes of work, highest first.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lane {
    /// Computing a section for a wanted key.
    Compute,
    /// Offering a computed `missing` section to the hub
    /// ([`crate::UploadLane`]).
    Upload,
    /// Checking a downloaded section by computing it again.
    Check,
}

impl Lane {
    /// Every lane, highest first.
    pub const ALL: [Lane; 3] = [Lane::Compute, Lane::Upload, Lane::Check];
}

struct Waiting<T> {
    key: ChunkKey,
    priority: f64,
    seq: u64,
    item: T,
}

/// Compute jobs waiting to start, each under its key and priority. Taken
/// highest priority first, ties first queued first.
pub struct ComputeQueue<T> {
    waiting: Vec<Waiting<T>>,
    next_seq: u64,
}

impl<T> Default for ComputeQueue<T> {
    fn default() -> ComputeQueue<T> {
        ComputeQueue {
            waiting: Vec::new(),
            next_seq: 0,
        }
    }
}

impl<T> ComputeQueue<T> {
    /// An empty queue.
    pub fn new() -> ComputeQueue<T> {
        ComputeQueue::default()
    }

    /// Queues `item` for `key` at `priority` (larger starts sooner).
    pub fn push(&mut self, key: ChunkKey, priority: f64, item: T) {
        self.waiting.push(Waiting {
            key,
            priority,
            seq: self.next_seq,
            item,
        });
        self.next_seq += 1;
    }

    /// Takes the waiting job of the highest priority, the first queued
    /// among equals.
    pub fn pop(&mut self) -> Option<(ChunkKey, T)> {
        let best = self
            .waiting
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| b.priority.total_cmp(&a.priority).then(a.seq.cmp(&b.seq)))
            .map(|(i, _)| i)?;
        let w = self.waiting.swap_remove(best);
        Some((w.key, w.item))
    }

    /// Gives every waiting job its key's priority from `priority`; a job
    /// whose key gets `None` (no longer wanted) is removed and returned,
    /// in the order it was queued.
    pub fn reprioritize(&mut self, priority: impl Fn(&ChunkKey) -> Option<f64>) -> Vec<T> {
        let mut dropped = Vec::new();
        let mut kept = Vec::with_capacity(self.waiting.len());
        let mut all = std::mem::take(&mut self.waiting);
        all.sort_by_key(|w| w.seq);
        for mut w in all {
            match priority(&w.key) {
                Some(p) => {
                    w.priority = p;
                    kept.push(w);
                }
                None => dropped.push(w.item),
            }
        }
        self.waiting = kept;
        dropped
    }

    /// Jobs waiting.
    pub fn len(&self) -> usize {
        self.waiting.len()
    }

    /// `true` when no job waits.
    pub fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u64) -> ChunkKey {
        format!("1-6-{i}-0-0").parse().unwrap()
    }

    #[test]
    fn highest_priority_first_ties_in_queued_order() {
        let mut q = ComputeQueue::new();
        q.push(key(0), 1.0, "a");
        q.push(key(1), 5.0, "b");
        q.push(key(2), 1.0, "c");
        q.push(key(3), 5.0, "d");
        let order: Vec<&str> = std::iter::from_fn(|| q.pop().map(|(_, s)| s)).collect();
        assert_eq!(order, ["b", "d", "a", "c"]);
    }

    #[test]
    fn reprioritizing_reorders_and_drops_unwanted_keys() {
        let mut q = ComputeQueue::new();
        for i in 0..4 {
            q.push(key(i), i as f64, i);
        }
        let dropped = q.reprioritize(|k| match k {
            k if *k == key(0) => Some(10.0),
            k if *k == key(2) => Some(1.0),
            _ => None,
        });
        assert_eq!(dropped, [1, 3]);
        assert_eq!(q.pop().map(|(_, i)| i), Some(0));
        assert_eq!(q.pop().map(|(_, i)| i), Some(2));
        assert!(q.is_empty());
    }

    #[test]
    fn lanes_are_ordered() {
        assert!(Lane::Compute < Lane::Upload && Lane::Upload < Lane::Check);
        assert_eq!(Lane::ALL[0], Lane::Compute);
    }
}
