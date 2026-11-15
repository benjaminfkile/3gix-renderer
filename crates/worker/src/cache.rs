//! The native client's local cache: modules, source manifests, source
//! tiles, and sections on disk, keyed by hash or address
//! (`client-compute.md` section 6).
//!
//! Everything in the cache is immutable for its name, so a hit never
//! touches the network and the cache is evicted by size alone: when a write
//! takes the total past the limit, the least recently used entries go
//! first (ties broken by kind and name, so eviction is the same on every
//! run). The browser keeps the same keys in the Cache API instead (the
//! renderer's `web` module).
//!
//! Layout under the root: one directory per [`BlobKind`] (its route
//! segment), one file per entry named by its 64 hex characters, and `tmp/`
//! for writes in progress, which are renamed into place so a reader never
//! sees a partial file. Only 64-character lowercase hex names are accepted,
//! so a name can never leave its directory.

use crate::job::{is_hex_name, BlobKind};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The default size limit: 20 GB (`20e9` bytes).
pub const DEFAULT_MAX_BYTES: u64 = 20_000_000_000;

/// The directory under the platform cache directory the client uses.
pub const CACHE_DIR_NAME: &str = "gx-renderer";

/// The platform cache directory joined with [`CACHE_DIR_NAME`]:
/// `$XDG_CACHE_HOME` or `~/.cache` on Linux and other Unix systems,
/// `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows. `None` when
/// none of those is set.
pub fn default_dir() -> Option<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = if cfg!(windows) {
        env("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Caches"))
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache")))
    };
    base.map(|b| b.join(CACHE_DIR_NAME))
}

#[derive(Copy, Clone, Debug)]
struct Entry {
    size: u64,
    stamp: u64,
}

#[derive(Default)]
struct Index {
    entries: BTreeMap<(BlobKind, String), Entry>,
    /// Use order: stamp to entry key.
    order: BTreeMap<u64, (BlobKind, String)>,
    total: u64,
    next_stamp: u64,
}

impl Index {
    fn touch(&mut self, key: &(BlobKind, String)) {
        let stamp = self.next_stamp;
        if let Some(e) = self.entries.get_mut(key) {
            self.order.remove(&e.stamp);
            e.stamp = stamp;
            self.order.insert(stamp, key.clone());
            self.next_stamp += 1;
        }
    }

    fn insert(&mut self, key: (BlobKind, String), size: u64) {
        self.remove(&key);
        let stamp = self.next_stamp;
        self.next_stamp += 1;
        self.total += size;
        self.order.insert(stamp, key.clone());
        self.entries.insert(key, Entry { size, stamp });
    }

    fn remove(&mut self, key: &(BlobKind, String)) -> bool {
        match self.entries.remove(key) {
            Some(e) => {
                self.order.remove(&e.stamp);
                self.total -= e.size;
                true
            }
            None => false,
        }
    }
}

/// A size-limited cache directory. Safe to share between threads.
pub struct DiskCache {
    root: PathBuf,
    max_bytes: u64,
    index: Mutex<Index>,
    tmp_counter: AtomicU64,
}

impl std::fmt::Debug for DiskCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskCache")
            .field("root", &self.root)
            .field("max_bytes", &self.max_bytes)
            .finish()
    }
}

impl DiskCache {
    /// Opens (creating if needed) the cache at `root` with a limit of
    /// `max_bytes`, indexing what is already there in the order the files
    /// were last used, and evicting down to the limit.
    pub fn open(root: impl Into<PathBuf>, max_bytes: u64) -> io::Result<DiskCache> {
        let root = root.into();
        let mut found: Vec<(std::time::SystemTime, BlobKind, String, u64)> = Vec::new();
        for kind in BlobKind::ALL {
            let dir = root.join(kind.segment());
            fs::create_dir_all(&dir)?;
            for item in fs::read_dir(&dir)? {
                let item = item?;
                let Some(name) = item.file_name().to_str().map(str::to_string) else {
                    continue;
                };
                let meta = item.metadata()?;
                if !meta.is_file() || !is_hex_name(&name) {
                    continue;
                }
                let used = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                found.push((used, kind, name, meta.len()));
            }
        }
        let tmp = root.join("tmp");
        if tmp.exists() {
            // Writes a previous run never finished.
            fs::remove_dir_all(&tmp)?;
        }
        fs::create_dir_all(&tmp)?;
        found.sort();
        let mut index = Index::default();
        for (_, kind, name, size) in found {
            index.insert((kind, name), size);
        }
        let cache = DiskCache {
            root,
            max_bytes,
            index: Mutex::new(index),
            tmp_counter: AtomicU64::new(0),
        };
        cache.evict(None);
        Ok(cache)
    }

    /// The cache directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The size limit, bytes.
    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    /// Bytes held now.
    pub fn total_bytes(&self) -> u64 {
        self.index.lock().expect("cache index").total
    }

    /// Entries held now.
    pub fn len(&self) -> usize {
        self.index.lock().expect("cache index").entries.len()
    }

    /// `true` when the cache holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn path(&self, kind: BlobKind, name: &str) -> PathBuf {
        self.root.join(kind.segment()).join(name)
    }

    /// `true` when the cache holds `name` of `kind`.
    pub fn contains(&self, kind: BlobKind, name: &str) -> bool {
        self.index
            .lock()
            .expect("cache index")
            .entries
            .contains_key(&(kind, name.to_string()))
    }

    /// The bytes of `name`, if cached. Marks the entry as used.
    pub fn get(&self, kind: BlobKind, name: &str) -> Option<Vec<u8>> {
        if !is_hex_name(name) {
            return None;
        }
        let key = (kind, name.to_string());
        if !self
            .index
            .lock()
            .expect("cache index")
            .entries
            .contains_key(&key)
        {
            return None;
        }
        let path = self.path(kind, name);
        match fs::read(&path) {
            Ok(bytes) => {
                self.index.lock().expect("cache index").touch(&key);
                // Best effort: the file time orders entries on the next open.
                if let Ok(f) = fs::File::options().write(true).open(&path) {
                    let _ = f.set_modified(std::time::SystemTime::now());
                }
                Some(bytes)
            }
            Err(_) => {
                self.index.lock().expect("cache index").remove(&key);
                None
            }
        }
    }

    /// Stores `bytes` as `name` of `kind`, then evicts down to the limit.
    /// Storing a name already held changes nothing: the bytes behind a name
    /// never change. An entry larger than the whole limit is not stored.
    pub fn put(&self, kind: BlobKind, name: &str, bytes: &[u8]) -> io::Result<()> {
        if !is_hex_name(name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cache names are 64 lowercase hex characters",
            ));
        }
        if self.contains(kind, name) || bytes.len() as u64 > self.max_bytes {
            return Ok(());
        }
        let n = self.tmp_counter.fetch_add(1, Ordering::Relaxed);
        let tmp = self
            .root
            .join("tmp")
            .join(format!("{name}.{}.{n}", std::process::id()));
        fs::write(&tmp, bytes)?;
        let path = self.path(kind, name);
        if let Err(e) = fs::rename(&tmp, &path) {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        let key = (kind, name.to_string());
        self.index
            .lock()
            .expect("cache index")
            .insert(key.clone(), bytes.len() as u64);
        self.evict(Some(&key));
        Ok(())
    }

    /// Drops `name` of `kind`, for bytes that turned out to be damaged.
    pub fn remove(&self, kind: BlobKind, name: &str) {
        if !is_hex_name(name) {
            return;
        }
        if self
            .index
            .lock()
            .expect("cache index")
            .remove(&(kind, name.to_string()))
        {
            let _ = fs::remove_file(self.path(kind, name));
        }
    }

    /// Removes the least recently used entries until the total is within
    /// the limit, sparing `keep` (the entry just written).
    fn evict(&self, keep: Option<&(BlobKind, String)>) {
        let mut doomed = Vec::new();
        {
            let mut index = self.index.lock().expect("cache index");
            while index.total > self.max_bytes {
                let victim = index.order.values().find(|k| Some(*k) != keep).cloned();
                let Some(victim) = victim else {
                    break;
                };
                index.remove(&victim);
                doomed.push(victim);
            }
        }
        for (kind, name) in doomed {
            let _ = fs::remove_file(self.path(kind, &name));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn name(i: u8) -> String {
        format!("{i:02x}").repeat(32)
    }

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("gx-worker-cache-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn put_get_and_reopen() {
        let root = dir("reopen");
        let c = DiskCache::open(&root, 1000).unwrap();
        assert!(c.get(BlobKind::Section, &name(1)).is_none());
        c.put(BlobKind::Section, &name(1), b"hello").unwrap();
        c.put(BlobKind::Module, &name(1), b"other").unwrap();
        assert_eq!(c.get(BlobKind::Section, &name(1)).unwrap(), b"hello");
        assert_eq!(c.get(BlobKind::Module, &name(1)).unwrap(), b"other");
        assert_eq!(c.total_bytes(), 10);
        assert!(c.put(BlobKind::Section, "../../x", b"no").is_err());
        drop(c);
        let c = DiskCache::open(&root, 1000).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c.get(BlobKind::Section, &name(1)).unwrap(), b"hello");
        c.remove(BlobKind::Section, &name(1));
        assert!(c.get(BlobKind::Section, &name(1)).is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn evicts_least_recently_used_by_size() {
        let root = dir("evict");
        let c = DiskCache::open(&root, 30).unwrap();
        for i in 0..3 {
            c.put(BlobKind::Source, &name(i), &[i; 10]).unwrap();
        }
        // Use the first, so the second is the oldest.
        assert!(c.get(BlobKind::Source, &name(0)).is_some());
        c.put(BlobKind::Source, &name(3), &[3; 10]).unwrap();
        assert!(c.contains(BlobKind::Source, &name(0)));
        assert!(!c.contains(BlobKind::Source, &name(1)));
        assert!(!root.join("source").join(name(1)).exists());
        assert_eq!(c.total_bytes(), 30);
        // Larger than the whole limit: not stored.
        c.put(BlobKind::Source, &name(9), &[0; 31]).unwrap();
        assert!(!c.contains(BlobKind::Source, &name(9)));
        // Reopening with a smaller limit evicts at once.
        drop(c);
        let c = DiskCache::open(&root, 15).unwrap();
        assert_eq!(c.len(), 1);
        let _ = fs::remove_dir_all(&root);
    }
}
