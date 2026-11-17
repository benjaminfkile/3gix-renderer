//! Where the selection runs (`rendering.md` sections 3.4 and 4).
//!
//! Natively the selection ([`crate::stream::select_all`]) runs on its own
//! thread, so its cost never lands on the frame thread: the frame thread
//! hands it a job (the frames, the camera, the view, and the cell states
//! that changed since the last job) and later picks up the finished
//! selection with the layout extraction needs ([`SelectionLayout`]) and the
//! cells whose place in it changed. One job is in flight at a time; what
//! changes meanwhile goes with the next one. The selection thread keeps its
//! own copy of every cell's state, kept current from those changes, so it
//! never reads the frame thread's cache.
//!
//! In the browser, and wherever a [`crate::world::World`] is made without
//! worker threads, the same work runs inline, at once, on the calling
//! thread.

use crate::camera::Camera;
use crate::stream::{select_all, CellMap, CellSet, CellState, SelectedCell, Selection};
use gx_core::frames::FrameSystem;
use gx_core::key::CellKey;
use std::collections::BTreeMap;

/// The latest drawn selection as extraction needs it: the selected cells,
/// and every cell with a selected cell strictly below it.
#[derive(Clone, Debug, Default)]
pub struct SelectionLayout {
    /// The cells of the drawn selection.
    pub selected: CellSet,
    /// Every strict ancestor of a selected cell.
    pub above_selected: CellSet,
}

impl SelectionLayout {
    /// The layout of a drawn selection.
    pub fn of(selection: &[SelectedCell]) -> SelectionLayout {
        let selected: CellSet = selection.iter().map(|c| c.key).collect();
        let mut above_selected = CellSet::default();
        for k in &selected {
            let mut up = k.parent();
            while let Some(p) = up {
                if !above_selected.insert(p) {
                    break;
                }
                up = p.parent();
            }
        }
        SelectionLayout {
            selected,
            above_selected,
        }
    }

    /// The keys whose selected or above-selected status differs between
    /// `self` and `other`, in key order.
    pub fn changes(&self, other: &SelectionLayout) -> Vec<CellKey> {
        let diff = |a: &CellSet, b: &CellSet, out: &mut Vec<CellKey>| {
            out.extend(a.iter().filter(|k| !b.contains(k)));
            out.extend(b.iter().filter(|k| !a.contains(k)));
        };
        let mut out = Vec::new();
        diff(&self.selected, &other.selected, &mut out);
        diff(&self.above_selected, &other.above_selected, &mut out);
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// One selection to make.
pub struct SelectJob {
    /// The frames as drawn.
    pub system: FrameSystem,
    /// The camera.
    pub camera: Camera,
    /// The view's width and height, pixels.
    pub view_px: (f64, f64),
    /// Cell states that changed since the previous job, in the order they
    /// changed: the new state, or `None` for a cell that left the cache.
    pub changes: Vec<(CellKey, Option<CellState>)>,
}

/// A finished selection.
pub struct Selected {
    /// The selection.
    pub selection: Selection,
    /// When the drawn selection differs from the previous one: its layout
    /// and the cells whose selected or above-selected status changed
    /// ([`SelectionLayout::changes`]).
    pub layout: Option<(SelectionLayout, Vec<CellKey>)>,
}

/// What the selection keeps from one job to the next.
#[derive(Default)]
pub struct SelectorState {
    states: CellMap<CellState>,
    budgets: BTreeMap<u64, usize>,
    drawn: Vec<CellKey>,
    layout: SelectionLayout,
}

impl SelectorState {
    /// Applies the job's changes and selects.
    pub fn run(&mut self, job: SelectJob) -> Selected {
        for (key, state) in job.changes {
            match state {
                Some(s) => {
                    self.states.insert(key, s);
                }
                None => {
                    self.states.remove(&key);
                }
            }
        }
        let selection = select_all(
            &job.system,
            &job.camera,
            job.view_px,
            &self.states,
            &mut self.budgets,
        );
        let same = selection.drawn.len() == self.drawn.len()
            && selection
                .drawn
                .iter()
                .zip(&self.drawn)
                .all(|(a, b)| a.key == *b);
        let layout = if same {
            None
        } else {
            self.drawn = selection.drawn.iter().map(|c| c.key).collect();
            let layout = SelectionLayout::of(&selection.drawn);
            let changes = layout.changes(&self.layout);
            self.layout = layout.clone();
            Some((layout, changes))
        };
        Selected { selection, layout }
    }
}

/// Runs selections inline or on the selection thread (module docs).
pub enum Selector {
    /// On the calling thread.
    Inline(Box<SelectorState>),
    /// On the selection thread.
    #[cfg(not(target_arch = "wasm32"))]
    Thread {
        /// Jobs to the thread.
        jobs: std::sync::mpsc::Sender<SelectJob>,
        /// Finished selections from it.
        done: std::sync::mpsc::Receiver<Selected>,
        /// A job is in flight.
        busy: bool,
    },
}

impl Selector {
    /// A selector on its own thread when `threaded` is set and the platform
    /// has threads, else inline.
    pub fn new(threaded: bool) -> Selector {
        #[cfg(not(target_arch = "wasm32"))]
        if threaded {
            let (jobs, rx) = std::sync::mpsc::channel::<SelectJob>();
            let (tx, done) = std::sync::mpsc::channel();
            let spawned = std::thread::Builder::new()
                .name("selection".into())
                .spawn(move || {
                    let mut state = SelectorState::default();
                    while let Ok(job) = rx.recv() {
                        if tx.send(state.run(job)).is_err() {
                            break;
                        }
                    }
                });
            match spawned {
                Ok(_) => {
                    return Selector::Thread {
                        jobs,
                        done,
                        busy: false,
                    }
                }
                Err(e) => tracing::warn!("selecting inline: {e}"),
            }
        }
        let _ = threaded;
        Selector::Inline(Box::default())
    }

    /// A job is in flight.
    pub fn busy(&self) -> bool {
        match self {
            Selector::Inline(_) => false,
            #[cfg(not(target_arch = "wasm32"))]
            Selector::Thread { busy, .. } => *busy,
        }
    }

    /// Starts `job`. Inline, the selection is made at once and returned;
    /// on the thread, it is picked up later ([`Selector::try_finish`],
    /// [`Selector::wait`]). Must not be called while [`Selector::busy`].
    pub fn start(&mut self, job: SelectJob) -> Option<Selected> {
        match self {
            Selector::Inline(state) => Some(state.run(job)),
            #[cfg(not(target_arch = "wasm32"))]
            Selector::Thread { jobs, busy, .. } => {
                debug_assert!(!*busy, "one selection job at a time");
                *busy = jobs.send(job).is_ok();
                None
            }
        }
    }

    /// The finished selection, if the job in flight is done.
    pub fn try_finish(&mut self) -> Option<Selected> {
        match self {
            Selector::Inline(_) => None,
            #[cfg(not(target_arch = "wasm32"))]
            Selector::Thread { done, busy, .. } => {
                if !*busy {
                    return None;
                }
                match done.try_recv() {
                    Ok(s) => {
                        *busy = false;
                        Some(s)
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => None,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        *busy = false;
                        None
                    }
                }
            }
        }
    }

    /// Waits for the job in flight, if any, and returns its selection.
    pub fn wait(&mut self) -> Option<Selected> {
        match self {
            Selector::Inline(_) => None,
            #[cfg(not(target_arch = "wasm32"))]
            Selector::Thread { done, busy, .. } => {
                if !*busy {
                    return None;
                }
                *busy = false;
                done.recv().ok()
            }
        }
    }
}
