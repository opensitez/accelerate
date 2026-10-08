//! Shared surface operations and native presentation, independent of DOM and layout.
//!
//! CPU targets always have a software implementation. Native window presentation
//! is optional: callers retain their software presenter when it is unavailable.
//! The current native backend uses Core Animation, not a Metal command encoder.

mod diagnostics;
pub mod fonts;
mod layer;
pub mod paint;
mod picture;
mod presentation;
pub mod software;
mod surface;
pub mod video;

#[cfg(any(target_os = "linux", target_os = "windows", feature = "backend-check"))]
#[cfg_attr(not(any(target_os = "linux", target_os = "windows")), allow(dead_code))]
mod gl;
#[cfg(any(target_os = "linux", feature = "backend-check"))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
mod linux;
#[cfg(any(target_os = "windows", feature = "backend-check"))]
#[cfg_attr(not(target_os = "windows"), allow(dead_code))]
mod windows;

#[cfg(target_os = "macos")]
mod macos;

pub use diagnostics::{Diagnostics, Phase};
pub use layer::{CroppedLayerFrame, ForegroundLayer, LayerFrame};
pub use picture::{MAX_PICTURE_BYTES, MAX_PICTURES, PictureLayer, draw_pictures};
pub use presentation::{Backend, FallbackReason, HostWindow, Preference, WindowSurface};
pub use software::draw_surface;
pub use surface::{InvalidRgbaSurface, RgbaSurface};

pub use layer::tinted_pixels;
