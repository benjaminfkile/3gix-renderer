//! Showing and hiding layers, shared by the native and browser builds
//! (`rendering.md` section 7 item 7 and section 6 rule 5).
//!
//! The viewer starts with the build's `shown` defaults, or the
//! `hide-layers` option instead ([`crate::options::hidden_layer_indices`]).
//! `L` opens a chooser on the overlay listing the build's layers by number,
//! shown or hidden; a number key toggles that layer. A toggle takes effect
//! on the next frame ([`crate::world::World::set_hidden_layers`]): a hidden
//! overlay is not composited, a hidden plain layer's cells and surface
//! tiles are not drawn. Nothing computed changes, and input resolution is
//! unaffected.

use std::collections::BTreeSet;

/// The build's layer ids in layer order and the hidden ones among them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayerView {
    /// The build's layer ids, in layer order.
    pub ids: Vec<String>,
    /// The hidden layers, by index into `ids`.
    pub hidden: BTreeSet<usize>,
}

impl LayerView {
    /// The view of layers `ids` starting with `hidden` hidden.
    pub fn new(ids: Vec<String>, hidden: BTreeSet<usize>) -> LayerView {
        LayerView { ids, hidden }
    }

    /// The view a client starts with for `build`: `hide` (the
    /// `hide-layers` option) when given, else the build's `shown`
    /// defaults. Ids not in the build are reported (logged) and ignored,
    /// and returned.
    pub fn for_build(
        build: &crate::protocol::BuildDescription,
        hide: Option<&[String]>,
    ) -> (LayerView, Vec<String>) {
        let ids: Vec<String> = build.layers.iter().map(|l| l.layer.clone()).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        let (hidden, unknown) =
            crate::options::hidden_layer_indices(&refs, hide, build.initially_hidden_layers());
        for id in &unknown {
            tracing::warn!("hide-layers: the build has no layer {id:?}; ignored");
        }
        (LayerView::new(ids, hidden), unknown)
    }

    /// Toggles the layer at `index` (`0` for key `1`). Returns `false`
    /// when there is no such layer.
    pub fn toggle(&mut self, index: usize) -> bool {
        if index >= self.ids.len() {
            return false;
        }
        if !self.hidden.remove(&index) {
            self.hidden.insert(index);
        }
        true
    }

    /// The overlay line of the open chooser: every layer by number with
    /// shown or hidden.
    pub fn chooser_line(&self) -> String {
        if self.ids.is_empty() {
            return "layers: none in this build  (L closes)".to_string();
        }
        let choices: Vec<String> = (self.ids.iter().enumerate())
            .take(10)
            .map(|(i, id)| {
                let state = if self.hidden.contains(&i) {
                    "hidden"
                } else {
                    "shown"
                };
                format!("{}:{id} {state}", (i + 1) % 10)
            })
            .collect();
        format!("layers: {}  (L closes)", choices.join("  "))
    }

    /// The overlay line naming the hidden layers, `None` while every
    /// layer is shown.
    pub fn hidden_line(&self) -> Option<String> {
        if self.hidden.is_empty() {
            return None;
        }
        let names: Vec<&str> = (self.hidden.iter())
            .filter_map(|&i| self.ids.get(i).map(String::as_str))
            .collect();
        Some(format!("hidden layers: {}", names.join(", ")))
    }

    /// Applies the toggles picked since the last frame
    /// ([`crate::controls::Controls::take_layer_toggles`]). Returns
    /// `true` when the hidden set changed, so the caller passes it on.
    pub fn apply(&mut self, toggles: &[usize]) -> bool {
        let mut changed = false;
        for &i in toggles {
            changed |= self.toggle(i);
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> LayerView {
        LayerView::new(vec!["ground".into(), "stain".into()], [1].into())
    }

    #[test]
    fn the_chooser_lists_layers_by_number() {
        let mut v = view();
        assert_eq!(
            v.chooser_line(),
            "layers: 1:ground shown  2:stain hidden  (L closes)"
        );
        assert_eq!(v.hidden_line().as_deref(), Some("hidden layers: stain"));
        assert!(v.apply(&[1, 0]));
        assert_eq!(v.hidden, [0].into());
        assert_eq!(v.hidden_line().as_deref(), Some("hidden layers: ground"));
        assert!(!v.apply(&[5]));
        assert!(v.apply(&[0]));
        assert_eq!(v.hidden_line(), None);
    }
}
