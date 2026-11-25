//! One memory budget in bytes (`rendering.md` sections 3 and 8 item 7,
//! `docs/architecture.md`, "Memory budget"): the bytes held by built and
//! compact surface tiles with their overlays, cached cells' decoded
//! samples, and extracted meshes, against one limit chosen once at start
//! by the platform layer. No API option sets it; the caches stay inside.

/// The native default: 2 GiB. A desktop has memory to spare and freed
/// memory goes back to the system, so the budget only keeps the caches
/// from growing without end on a long flight.
pub const NATIVE_BUDGET_BYTES: u64 = 2 << 30;

/// The browser default: 512 MiB. WebAssembly memory never shrinks, so the
/// highest point while loading is what the page keeps for its whole life;
/// this keeps it well inside what a desktop browser tab is given.
pub const BROWSER_BUDGET_BYTES: u64 = 512 << 20;

/// The browser default on a touch-first or small-memory device: 192 MiB.
/// Phones and small tablets end a tab that holds much more, and their
/// views are small enough that the drawn tiles and cells fit well inside.
pub const SMALL_DEVICE_BUDGET_BYTES: u64 = 192 << 20;

/// Asking ahead resumes once the held bytes fall below this fraction of
/// the budget (`docs/architecture.md`, "Memory budget").
pub const RESUME_FRACTION: f64 = 0.9;

/// Whether asking ahead is held after a pass that left `held` bytes, given
/// whether it was held before: held while over `budget`, and held until
/// the bytes fall below [`RESUME_FRACTION`] of it.
pub fn hold_ahead(held: u64, budget: u64, was: bool) -> bool {
    if held > budget {
        true
    } else if (held as f64) < RESUME_FRACTION * budget as f64 {
        false
    } else {
        was
    }
}

/// One step of the detail factor ([`Detail`]): each step down scales
/// every split distance by this. Chosen so a step changes the drawn
/// bytes by about a third (the drawn area at each level goes with the
/// square of the split distance), small enough to end near the budget
/// and large enough to settle in a few passes.
pub const DETAIL_STEP: f64 = 0.8;

/// The most steps the detail factor goes down: `0.8^16`, about 1/35,
/// leaves the level-0 tiles and few more, the least the view draws.
pub const DETAIL_STEPS: u8 = 16;

/// Seconds held bytes stay below [`RESUME_FRACTION`] of the budget before
/// the detail factor goes up a step, and the longest a step down waits
/// for the step before it to take effect.
pub const DETAIL_WAIT_SECONDS: f64 = 2.0;

/// The detail factor (`rendering.md` 3.5, 8 item 7): a number no greater
/// than 1 that scales every surface tile's split distance, so when the
/// tiles the view wants at full detail do not fit the memory budget the
/// selection draws fewer, coarser tiles instead of eviction dropping what
/// the selection asks for again.
///
/// It goes down a step after an eviction pass that left the held bytes
/// over the budget, once the step before has taken effect (the
/// selection's builds are done) or [`DETAIL_WAIT_SECONDS`] have passed.
/// It goes up a step only after the held bytes stay below
/// [`RESUME_FRACTION`] of the budget for [`DETAIL_WAIT_SECONDS`], and
/// never back to a step found over the budget for the same camera and
/// budget, so a still view settles instead of swinging between two.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Detail {
    step: u8,
    /// The finest step found over the budget since the camera or the
    /// budget last changed.
    ceiling: Option<u8>,
    /// Since when held bytes have been below [`RESUME_FRACTION`] of the
    /// budget.
    below_since: Option<f64>,
    /// When the step last changed, on the selection's clock.
    changed_at: f64,
}

impl Detail {
    /// The factor, `DETAIL_STEP^step`: 1 at full detail.
    pub fn factor(&self) -> f64 {
        DETAIL_STEP.powi(i32::from(self.step))
    }

    /// Steps below full detail.
    pub fn step(&self) -> u8 {
        self.step
    }

    /// Forgets the steps found over the budget: the camera or the budget
    /// changed, so what fits may have changed.
    pub fn forget(&mut self) {
        self.ceiling = None;
    }

    /// Takes the result of one eviction pass at `now`, seconds: `held`
    /// bytes against `budget`, and whether the selection's builds and
    /// requests are all done (`settled`). Returns whether the step
    /// changed.
    pub fn after_pass(&mut self, held: u64, budget: u64, settled: bool, now: f64) -> bool {
        if held > budget {
            self.below_since = None;
            let due = settled || now - self.changed_at >= DETAIL_WAIT_SECONDS;
            if due && self.step < DETAIL_STEPS {
                self.ceiling = Some(self.ceiling.map_or(self.step, |c| c.max(self.step)));
                self.step += 1;
                self.changed_at = now;
                return true;
            }
        } else if (held as f64) < RESUME_FRACTION * budget as f64 {
            let since = *self.below_since.get_or_insert(now);
            let allowed = self.ceiling.is_none_or(|c| self.step > c + 1);
            if self.step > 0 && allowed && now - since >= DETAIL_WAIT_SECONDS {
                self.step -= 1;
                self.changed_at = now;
                self.below_since = Some(now);
                return true;
            }
        } else {
            self.below_since = None;
        }
        false
    }
}

/// The browser's budget from the two device facts: whether the primary
/// pointer is coarse (`matchMedia("(pointer: coarse)")` matches) and the
/// device memory in gigabytes (`navigator.deviceMemory`, `None` where the
/// browser does not say).
pub fn browser_budget(coarse_pointer: bool, device_memory_gb: Option<f64>) -> u64 {
    if coarse_pointer || device_memory_gb.is_some_and(|gb| gb <= 4.0) {
        SMALL_DEVICE_BUDGET_BYTES
    } else {
        BROWSER_BUDGET_BYTES
    }
}

/// The budget for the platform this runs on, read once at start: the
/// native default, or in the browser [`browser_budget`] of the page's
/// device facts.
pub fn platform_budget() -> u64 {
    #[cfg(all(target_arch = "wasm32", feature = "web"))]
    {
        let (coarse, memory) = browser_facts();
        browser_budget(coarse, memory)
    }
    #[cfg(all(target_arch = "wasm32", not(feature = "web")))]
    {
        BROWSER_BUDGET_BYTES
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        NATIVE_BUDGET_BYTES
    }
}

/// Whether the pointer is coarse and the device memory, gigabytes, read
/// from the global scope (a window or a worker; a worker has no
/// `matchMedia` and counts as not coarse).
#[cfg(all(target_arch = "wasm32", feature = "web"))]
fn browser_facts() -> (bool, Option<f64>) {
    use wasm_bindgen::{JsCast, JsValue};
    let global = js_sys::global();
    let get = |o: &JsValue, name: &str| js_sys::Reflect::get(o, &JsValue::from_str(name)).ok();
    let memory = get(&global, "navigator")
        .and_then(|n| get(&n, "deviceMemory"))
        .and_then(|m| m.as_f64());
    let coarse = get(&global, "matchMedia")
        .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
        .and_then(|f| {
            f.call1(&global, &JsValue::from_str("(pointer: coarse)"))
                .ok()
        })
        .and_then(|q| get(&q, "matches"))
        .and_then(|m| m.as_bool())
        .unwrap_or(false);
    (coarse, memory)
}

/// What one eviction pass did ([`crate::world::World::evict`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Evicted {
    /// Built surface tiles made compact.
    pub compacted: usize,
    /// Compact surface tiles dropped.
    pub tiles: usize,
    /// Cells dropped.
    pub cells: usize,
    /// Meshes dropped.
    pub meshes: usize,
}

/// One item eviction may drop over the budget: when it was last used,
/// its level (deepest first on a tie), and its bytes.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Candidate<K> {
    /// Last use, seconds of the selection's clock.
    pub last_used: f64,
    /// Tile level or cell depth.
    pub level: u8,
    /// The item.
    pub key: K,
    /// Bytes it holds.
    pub bytes: u64,
}

/// Sorts `items` least recently used first, the deepest level first on a
/// tie, then by key, so the order is the same on every run.
pub fn sort_candidates<K: Ord>(items: &mut [Candidate<K>]) {
    items.sort_by(|a, b| {
        a.last_used
            .total_cmp(&b.last_used)
            .then(b.level.cmp(&a.level))
            .then_with(|| a.key.cmp(&b.key))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_two_gib_native_and_512_or_192_mib_in_the_browser() {
        assert_eq!(NATIVE_BUDGET_BYTES, 2 * 1024 * 1024 * 1024);
        #[cfg(not(target_arch = "wasm32"))]
        assert_eq!(platform_budget(), NATIVE_BUDGET_BYTES);
        assert_eq!(browser_budget(false, None), 512 * 1024 * 1024);
        assert_eq!(browser_budget(false, Some(8.0)), 512 * 1024 * 1024);
        assert_eq!(browser_budget(true, None), 192 * 1024 * 1024);
        assert_eq!(browser_budget(true, Some(16.0)), 192 * 1024 * 1024);
        assert_eq!(browser_budget(false, Some(4.0)), 192 * 1024 * 1024);
        assert_eq!(browser_budget(false, Some(2.0)), 192 * 1024 * 1024);
    }

    #[test]
    fn asking_ahead_stops_over_the_budget_and_resumes_below_ninety_percent() {
        let b = 1000;
        assert!(!hold_ahead(1000, b, false));
        assert!(hold_ahead(1001, b, false));
        assert!(
            hold_ahead(950, b, true),
            "still held between 90 and 100 percent"
        );
        assert!(hold_ahead(900, b, true), "still held at 90 percent");
        assert!(!hold_ahead(899, b, true));
        assert!(!hold_ahead(950, b, false));
    }

    #[test]
    fn detail_goes_down_over_the_budget_and_up_only_below_resume_for_a_while() {
        let b = 1000;
        let mut d = Detail::default();
        assert_eq!(d.factor(), 1.0);
        assert!(
            !d.after_pass(1000, b, true, 0.0),
            "at the budget: no change"
        );
        assert!(d.after_pass(1001, b, true, 0.1));
        assert_eq!(d.step(), 1);
        assert!(
            !d.after_pass(1001, b, false, 0.2),
            "waits for the step to take effect"
        );
        assert!(d.after_pass(1001, b, false, 0.1 + DETAIL_WAIT_SECONDS));
        assert_eq!(d.step(), 2);
        // Below the resume fraction: up only after the wait, and never
        // back to a step found over the budget.
        assert!(!d.after_pass(800, b, true, 3.0));
        assert!(!d.after_pass(800, b, true, 3.0 + DETAIL_WAIT_SECONDS));
        assert_eq!(d.step(), 2);
        d.forget();
        assert!(d.after_pass(800, b, true, 6.0), "below since 3 s");
        assert_eq!(d.step(), 1);
        // Between the resume fraction and the budget: the wait restarts.
        assert!(!d.after_pass(950, b, true, 20.0));
        assert!(!d.after_pass(800, b, true, 21.0));
        assert!(!d.after_pass(800, b, true, 22.0));
        assert!(d.after_pass(800, b, true, 21.0 + DETAIL_WAIT_SECONDS));
        assert_eq!(d.factor(), 1.0);
    }

    #[test]
    fn candidates_go_oldest_first_then_deepest() {
        let c = |last_used, level, key| Candidate {
            last_used,
            level,
            key,
            bytes: 1,
        };
        let mut v = vec![c(2.0, 1, 0), c(1.0, 1, 1), c(1.0, 3, 2), c(1.0, 3, 3)];
        sort_candidates(&mut v);
        let keys: Vec<u32> = v.iter().map(|c| c.key).collect();
        assert_eq!(keys, vec![2, 3, 1, 0]);
    }
}
