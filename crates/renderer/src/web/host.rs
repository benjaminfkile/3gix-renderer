//! The page's one event loop.
//!
//! `winit` allows one event loop per page, and never a second one, even
//! after the first has stopped. A page may create a renderer, destroy it,
//! and create another (a component unmounted and mounted again, React's
//! strict mode, a new connection), so the loop is created with the first
//! renderer and kept for the life of the page. Each later renderer is
//! handed to it: the loop opens the renderer's canvas window, routes each
//! window event to the renderer that owns the window, and drops a renderer
//! once it is done (`docs/api.md` section 3, `destroy`).

use super::app::WebApp;
use std::cell::RefCell;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::platform::web::EventLoopExtWebSys;
use winit::window::WindowId;

/// Wakes the loop: a renderer is waiting to be opened.
#[derive(Debug, Clone, Copy)]
pub struct Arrived;

thread_local! {
    /// Renderers handed over and not yet opened by the loop.
    static WAITING: RefCell<Vec<WebApp>> = const { RefCell::new(Vec::new()) };
    /// The running loop's proxy; `None` until the first renderer.
    static PROXY: RefCell<Option<EventLoopProxy<Arrived>>> = const { RefCell::new(None) };
}

/// Hands `app` to the page's loop, creating the loop with the first
/// renderer of the page. Never called from inside the loop's own
/// handlers, so nothing here is borrowed twice.
pub fn run(app: WebApp) -> Result<(), String> {
    WAITING.with(|w| w.borrow_mut().push(app));
    let proxy = PROXY.with(|p| p.borrow().clone());
    if let Some(proxy) = proxy {
        return proxy
            .send_event(Arrived)
            .map_err(|e| format!("waking the event loop: {e}"));
    }
    let event_loop = EventLoop::<Arrived>::with_user_event()
        .build()
        .map_err(|e| e.to_string())?;
    // Wait, not Poll: on the web Poll is a busy loop that starves the page.
    // Each frame requests the next redraw, which the browser delivers once
    // per display refresh.
    event_loop.set_control_flow(ControlFlow::Wait);
    PROXY.with(|p| *p.borrow_mut() = Some(event_loop.create_proxy()));
    event_loop.spawn_app(Host { apps: Vec::new() });
    Ok(())
}

/// The loop's handler: every live renderer of the page.
struct Host {
    apps: Vec<WebApp>,
}

impl Host {
    /// Opens the renderers handed over since the last call.
    fn open_waiting(&mut self, event_loop: &ActiveEventLoop) {
        let waiting = WAITING.with(|w| std::mem::take(&mut *w.borrow_mut()));
        for mut app in waiting {
            app.open(event_loop);
            self.apps.push(app);
        }
    }
}

impl ApplicationHandler<Arrived> for Host {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.open_waiting(event_loop);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, _event: Arrived) {
        self.open_waiting(event_loop);
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if let Some(app) = self.apps.iter_mut().find(|a| a.window_id() == Some(id)) {
            app.window_event(event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.open_waiting(event_loop);
        self.apps.retain_mut(|app| app.after_events());
    }
}
