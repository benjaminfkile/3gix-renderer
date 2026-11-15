//! The browser's local cache (`client-compute.md` section 6): modules,
//! manifests, source tiles, and sections in the Cache API, under the same
//! keys as the native cache directory ([`gx_worker::DiskCache`]): the kind's
//! route segment and the hash or address.
//!
//! Everything in it is immutable, so it is evicted by size alone, least
//! recently used first. The Cache API keeps no sizes, so the cache keeps an
//! index (kind, name, and length, in use order) in one more entry, written
//! at most every [`INDEX_FLUSH_SECONDS`]. On open, entries the index does
//! not list (a page closed before the index was written) are deleted, so
//! the size limit holds across sessions.
//!
//! A hit never touches the network. Every read is checked by the caller
//! (hashes, or the validator for sections) exactly as natively.

use super::hub::now_seconds;
use gx_worker::BlobKind;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

/// The Cache API cache the client uses.
pub const CACHE_NAME: &str = "gx-renderer";

/// Path prefix of every key: `/gx-cache/{segment}/{name}`, resolved
/// against the page's origin.
pub const KEY_PREFIX: &str = "/gx-cache";

/// Default size limit, bytes: 20 GB, as natively. The browser's own quota
/// may be lower; a refused write is logged and the bytes are not cached.
pub const DEFAULT_MAX_BYTES: u64 = 20_000_000_000;

/// Least seconds between two writes of the index.
pub const INDEX_FLUSH_SECONDS: f64 = 5.0;

fn index_url() -> String {
    format!("{KEY_PREFIX}/index.json")
}

fn url(kind: BlobKind, name: &str) -> String {
    format!("{KEY_PREFIX}/{}/{name}", kind.segment())
}

fn kind_of(segment: &str) -> Option<BlobKind> {
    BlobKind::ALL.into_iter().find(|k| k.segment() == segment)
}

/// Lengths and use order of everything stored.
#[derive(Default)]
struct Index {
    /// Kind and name to length and last use tick.
    entries: BTreeMap<(BlobKind, String), (u64, u64)>,
    /// Use tick to kind and name, oldest first.
    order: BTreeMap<u64, (BlobKind, String)>,
    tick: u64,
    total: u64,
}

impl Index {
    fn touch(&mut self, key: &(BlobKind, String)) {
        if let Some((_, t)) = self.entries.get_mut(key) {
            self.order.remove(t);
            self.tick += 1;
            *t = self.tick;
            self.order.insert(self.tick, key.clone());
        }
    }

    fn insert(&mut self, key: (BlobKind, String), size: u64) {
        self.remove(&key);
        self.tick += 1;
        self.entries.insert(key.clone(), (size, self.tick));
        self.order.insert(self.tick, key);
        self.total += size;
    }

    fn remove(&mut self, key: &(BlobKind, String)) -> bool {
        match self.entries.remove(key) {
            Some((size, t)) => {
                self.order.remove(&t);
                self.total -= size;
                true
            }
            None => false,
        }
    }

    /// `[segment, name, length]` per entry, oldest first.
    fn to_json(&self) -> Vec<u8> {
        let list: Vec<(&str, &str, u64)> = self
            .order
            .values()
            .map(|(k, n)| (k.segment(), n.as_str(), self.entries[&(*k, n.clone())].0))
            .collect();
        serde_json::to_vec(&list).unwrap_or_default()
    }

    fn from_json(bytes: &[u8]) -> Index {
        let mut index = Index::default();
        let list: Vec<(String, String, u64)> = serde_json::from_slice(bytes).unwrap_or_default();
        for (segment, name, size) in list {
            if let Some(kind) = kind_of(&segment) {
                if gx_worker::job::is_hex_name(&name) {
                    index.insert((kind, name), size);
                }
            }
        }
        index
    }
}

/// The browser's local cache.
pub struct WebCache {
    cache: web_sys::Cache,
    index: RefCell<Index>,
    max_bytes: u64,
    dirty: Cell<bool>,
    flushed_at: Cell<f64>,
}

fn js_text(e: JsValue) -> String {
    e.as_string()
        .or_else(|| {
            e.dyn_ref::<js_sys::Error>()
                .map(|e| String::from(e.message()))
        })
        .unwrap_or_else(|| format!("{e:?}"))
}

async fn response_bytes(value: JsValue) -> Option<Vec<u8>> {
    let resp: web_sys::Response = value.dyn_into().ok()?;
    let buffer = JsFuture::from(resp.array_buffer().ok()?).await.ok()?;
    Some(js_sys::Uint8Array::new(&buffer).to_vec())
}

impl WebCache {
    /// Opens the cache with a size limit of `max_bytes`, or `None` when the
    /// page has no Cache API (it needs a secure context: `https` or
    /// `localhost`).
    pub async fn open(max_bytes: u64) -> Option<WebCache> {
        let storage = web_sys::window()?.caches().ok()?;
        let cache: web_sys::Cache = JsFuture::from(storage.open(CACHE_NAME))
            .await
            .ok()?
            .dyn_into()
            .ok()?;
        let index = match JsFuture::from(cache.match_with_str(&index_url())).await {
            Ok(v) if !v.is_undefined() => response_bytes(v)
                .await
                .map(|b| Index::from_json(&b))
                .unwrap_or_default(),
            _ => Index::default(),
        };
        let web = WebCache {
            cache,
            index: RefCell::new(index),
            max_bytes,
            dirty: Cell::new(false),
            flushed_at: Cell::new(now_seconds()),
        };
        web.drop_unlisted().await;
        web.evict(None).await;
        Some(web)
    }

    /// Deletes stored entries the index does not list.
    async fn drop_unlisted(&self) {
        let Ok(keys) = JsFuture::from(self.cache.keys()).await else {
            return;
        };
        let keys: js_sys::Array = keys.unchecked_into();
        let index_path = index_url();
        for request in keys.iter() {
            let Ok(request) = request.dyn_into::<web_sys::Request>() else {
                continue;
            };
            let full = request.url();
            let Some(at) = full.find(KEY_PREFIX) else {
                continue;
            };
            let path = &full[at..];
            if path == index_path {
                continue;
            }
            let listed = path[KEY_PREFIX.len()..]
                .trim_start_matches('/')
                .split_once('/')
                .and_then(|(s, n)| Some((kind_of(s)?, n.to_string())))
                .is_some_and(|k| self.index.borrow().entries.contains_key(&k));
            if !listed {
                let _ = JsFuture::from(self.cache.delete_with_str(&full)).await;
            }
        }
    }

    /// Total length of everything stored, bytes.
    pub fn total_bytes(&self) -> u64 {
        self.index.borrow().total
    }

    /// The bytes stored under `kind` and `name`, or `None`.
    pub async fn get(&self, kind: BlobKind, name: &str) -> Option<Vec<u8>> {
        let key = (kind, name.to_string());
        if !self.index.borrow().entries.contains_key(&key) {
            return None;
        }
        let found = JsFuture::from(self.cache.match_with_str(&url(kind, name)))
            .await
            .ok()
            .filter(|v| !v.is_undefined());
        match found {
            Some(v) => {
                let bytes = response_bytes(v).await;
                if bytes.is_some() {
                    self.index.borrow_mut().touch(&key);
                    self.dirty.set(true);
                }
                bytes
            }
            None => {
                self.index.borrow_mut().remove(&key);
                self.dirty.set(true);
                None
            }
        }
    }

    /// Stores `bytes` under `kind` and `name`, then evicts the least
    /// recently used entries until the cache is within its limit.
    pub async fn put(&self, kind: BlobKind, name: &str, bytes: &[u8]) -> Result<(), String> {
        let size = bytes.len() as u64;
        if size > self.max_bytes {
            return Ok(());
        }
        let mut body = bytes.to_vec();
        let resp = web_sys::Response::new_with_opt_u8_array(Some(&mut body)).map_err(js_text)?;
        JsFuture::from(self.cache.put_with_str(&url(kind, name), &resp))
            .await
            .map_err(js_text)?;
        let key = (kind, name.to_string());
        self.index.borrow_mut().insert(key.clone(), size);
        self.dirty.set(true);
        self.evict(Some(&key)).await;
        Ok(())
    }

    /// Deletes the entry under `kind` and `name`, for bytes that failed
    /// their check.
    pub async fn remove(&self, kind: BlobKind, name: &str) {
        let removed = self.index.borrow_mut().remove(&(kind, name.to_string()));
        if removed {
            self.dirty.set(true);
        }
        let _ = JsFuture::from(self.cache.delete_with_str(&url(kind, name))).await;
    }

    async fn evict(&self, keep: Option<&(BlobKind, String)>) {
        loop {
            let victim = {
                let mut index = self.index.borrow_mut();
                if index.total <= self.max_bytes {
                    return;
                }
                let Some(k) = index.order.values().find(|k| Some(*k) != keep).cloned() else {
                    return;
                };
                index.remove(&k);
                k
            };
            self.dirty.set(true);
            let _ = JsFuture::from(self.cache.delete_with_str(&url(victim.0, &victim.1))).await;
        }
    }

    /// Writes the index if it changed and [`INDEX_FLUSH_SECONDS`] have
    /// passed since the last write.
    pub async fn flush_index(&self) {
        let now = now_seconds();
        if !self.dirty.get() || now - self.flushed_at.get() < INDEX_FLUSH_SECONDS {
            return;
        }
        self.dirty.set(false);
        self.flushed_at.set(now);
        let mut body = self.index.borrow().to_json();
        let Ok(resp) = web_sys::Response::new_with_opt_u8_array(Some(&mut body)) else {
            return;
        };
        if let Err(e) = JsFuture::from(self.cache.put_with_str(&index_url(), &resp)).await {
            self.dirty.set(true);
            web_sys::console::warn_1(
                &format!("gx-renderer: writing the cache index: {}", js_text(e)).into(),
            );
        }
    }
}
