use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::{Window, WindowId}};

pub mod renderer;
pub mod math;

#[derive(Default)]
struct CialloEngineApp {
    window: Option<Window>,
    renderer: Option<renderer::CialloRenderer>,
}

impl ApplicationHandler for CialloEngineApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match event_loop.create_window(Window::default_attributes()) {
            Ok(window) => {
                self.renderer = Some(renderer::CialloRenderer::new());
                self.window = Some(window);
            }
            Err(err) => {
                log::error!("Failed to create window: {err}");
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                // Draw.
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.render();
                }
                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw in
                // applications which do not always need to. Applications that redraw continuously
                // can render here instead.
                // self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = CialloEngineApp::default();

    let _ = event_loop.run_app(&mut app);
    // ...
}
