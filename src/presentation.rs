use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;
use tiny_skia::Pixmap;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Preference {
    #[default]
    Auto,
    Software,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Software,
    CoreAnimation,
    Wgl,
    Egl,
}

pub trait HostWindow: HasWindowHandle + HasDisplayHandle {}
impl<T: HasWindowHandle + HasDisplayHandle> HostWindow for T {}

#[cfg(target_os = "windows")]
type NativeContext = crate::windows::NativeContext;
#[cfg(target_os = "linux")]
type NativeContext = crate::linux::NativeContext;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallbackReason {
    SoftwareRequested,
    Disabled,
    Unavailable,
    PresentationFailed,
}

/// Optional native presentation. `false` from presentation requests software fallback.
/// This does not claim that rasterization or video decoding runs on the GPU.
pub struct WindowSurface {
    #[cfg(target_os = "macos")]
    native: Option<super::macos::VideoLayer>,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    native_gl: Option<crate::gl::GlSurface<NativeContext>>,
    fallback: Option<FallbackReason>,
    // Retain the host window until after its native layers have been removed.
    _window: Option<Arc<dyn HostWindow>>,
}

impl WindowSurface {
    pub fn new<W: HostWindow + 'static>(window: Arc<W>, preference: Preference) -> Self {
        if preference == Preference::Software {
            return Self::software();
        }
        if std::env::var_os("ACCELERATE_DISABLE_GPU").is_some()
            || std::env::var_os("WEBACCEL_DISABLE_GPU").is_some()
        {
            return Self::fallback(FallbackReason::Disabled);
        }
        #[cfg(target_os = "macos")]
        if let Some(native) = super::macos::VideoLayer::new(window.as_ref()) {
            return Self {
                native: Some(native),
                fallback: None,
                _window: Some(window),
            };
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(native_gl) =
            NativeContext::new(window.as_ref(), window.as_ref()).and_then(crate::gl::GlSurface::new)
        {
            return Self {
                native_gl: Some(native_gl),
                fallback: None,
                _window: Some(window),
            };
        }
        let _ = window;
        Self::fallback(FallbackReason::Unavailable)
    }

    pub fn software() -> Self {
        Self::fallback(FallbackReason::SoftwareRequested)
    }

    fn fallback(reason: FallbackReason) -> Self {
        Self {
            #[cfg(target_os = "macos")]
            native: None,
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            native_gl: None,
            fallback: Some(reason),
            _window: None,
        }
    }

    pub fn with_diagnostics(self, diagnostics: super::Diagnostics) -> Self {
        #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
        {
            let mut surface = self;
            #[cfg(target_os = "macos")]
            if let Some(native) = surface.native.as_mut() {
                native.diagnostics = diagnostics;
            }
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            if let Some(native) = surface.native_gl.as_mut() {
                native.diagnostics = diagnostics;
            }
            surface
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            let _ = diagnostics;
            self
        }
    }

    pub fn backend(&self) -> Backend {
        #[cfg(target_os = "macos")]
        if self.native.is_some() {
            return Backend::CoreAnimation;
        }
        #[cfg(target_os = "windows")]
        if self.native_gl.is_some() {
            return Backend::Wgl;
        }
        #[cfg(target_os = "linux")]
        if self.native_gl.is_some() {
            return Backend::Egl;
        }
        Backend::Software
    }

    pub fn fallback_reason(&self) -> Option<FallbackReason> {
        self.fallback
    }

    pub fn supports_layers(&self) -> bool {
        self.backend() != Backend::Software
    }

    /// Physical framebuffer size and pixels per logical host coordinate.
    pub fn resize(&mut self, width: u32, height: u32, scale: f32) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(native) = self.native_gl.as_mut() {
            native.resize(width, height, scale);
        }
        let _ = (width, height, scale);
    }

    pub fn present_layer(&mut self, frame: Option<&super::LayerFrame>) -> bool {
        #[cfg(target_os = "macos")]
        if let Some(native) = self.native.as_mut() {
            native.present(frame);
            return true;
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(native) = self.native_gl.as_mut() {
            if native.present_layer(frame) {
                return true;
            }
            self.native_gl = None;
            self.fallback = Some(FallbackReason::PresentationFailed);
        }
        let _ = frame;
        false
    }

    pub fn present_surface(&mut self, surface: &crate::RgbaSurface, destination: [f32; 4]) -> bool {
        self.present_layer(Some(&surface.layer_frame(destination)))
    }

    pub fn present_cropped_layer(&mut self, layer: Option<&crate::CroppedLayerFrame>) -> bool {
        self.present_masked_cropped_layer(layer, None)
    }

    /// The mask's alpha covers the full logical destination, not the source crop.
    /// Reuse its immutable pixels across crop-only updates to avoid native uploads.
    pub fn present_masked_cropped_layer(
        &mut self,
        layer: Option<&crate::CroppedLayerFrame>,
        mask: Option<&crate::RgbaSurface>,
    ) -> bool {
        if layer.is_some_and(|layer| !layer.is_valid()) || !crate::layer::valid_mask(mask) {
            return false;
        }
        #[cfg(target_os = "macos")]
        if let Some(native) = self.native.as_mut() {
            if native.present_masked_cropped_layer(layer, mask) {
                return true;
            }
            native.present(None);
            return false;
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(native) = self.native_gl.as_mut() {
            if native.present_masked_cropped_layer(layer, mask) {
                return true;
            }
            self.native_gl = None;
            self.fallback = Some(FallbackReason::PresentationFailed);
        }
        let _ = (layer, mask);
        false
    }

    pub fn present_framebuffer(&mut self, pixels: &Arc<Pixmap>) -> bool {
        #[cfg(target_os = "macos")]
        if let Some(native) = self.native.as_ref() {
            if native.present_framebuffer(pixels) {
                return true;
            }
            self.native = None;
            self.fallback = Some(FallbackReason::PresentationFailed);
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(native) = self.native_gl.as_mut() {
            if native.present_framebuffer(pixels) {
                return true;
            }
            self.native_gl = None;
            self.fallback = Some(FallbackReason::PresentationFailed);
        }
        let _ = pixels;
        false
    }

    /// Submit a framebuffer and ordered pictures in one native commit/swap.
    /// Use this instead of two presentation calls when both inputs change.
    /// False requests software fallback and disables the failed native presenter.
    pub fn present_framebuffer_with_pictures(
        &mut self,
        pixels: &Arc<Pixmap>,
        pictures: &[crate::PictureLayer],
    ) -> bool {
        #[cfg(target_os = "macos")]
        if let Some(native) = self.native.as_mut() {
            if native.present_framebuffer_with_pictures(pixels, pictures) {
                return true;
            }
            self.native = None;
            self.fallback = Some(FallbackReason::PresentationFailed);
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(native) = self.native_gl.as_mut() {
            if native.present_framebuffer_with_pictures(pixels, pictures) {
                return true;
            }
            self.native_gl = None;
            self.fallback = Some(FallbackReason::PresentationFailed);
        }
        let _ = (pixels, pictures);
        false
    }

    /// Retain and composite ordered pictures above framebuffer/video layers.
    /// Transform/opacity/clip-only changes reuse immutable image uploads.
    /// An empty slice removes the pictures. False requests the caller's software
    /// path; no GPU readback or font rasterization is performed here.
    pub fn present_pictures(&mut self, pictures: &[crate::PictureLayer]) -> bool {
        if !crate::picture::valid_picture_storage(pictures) {
            #[cfg(target_os = "macos")]
            if let Some(native) = self.native.as_mut() {
                native.clear_pictures();
            }
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            if let Some(native) = self.native_gl.as_mut() {
                native.present_pictures(&[]);
            }
            return false;
        }
        #[cfg(target_os = "macos")]
        if let Some(native) = self.native.as_mut() {
            if native.present_pictures(pictures) {
                return true;
            }
            native.clear_pictures();
            return false;
        }
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(native) = self.native_gl.as_mut() {
            if native.present_pictures(pictures) {
                return true;
            }
            self.native_gl = None;
            self.fallback = Some(FallbackReason::PresentationFailed);
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_fallback_is_available_without_a_window_or_device() {
        let mut surface = WindowSurface::software();
        assert_eq!(surface.backend(), Backend::Software);
        assert_eq!(
            surface.fallback_reason(),
            Some(FallbackReason::SoftwareRequested)
        );
        assert!(!surface.supports_layers());
        assert!(!surface.present_layer(None));
        assert!(!surface.present_framebuffer(&Arc::new(Pixmap::new(1, 1).unwrap())));
        let pixels = crate::RgbaSurface::new(Arc::new(vec![255; 4]), 1, 1).unwrap();
        let layer = crate::CroppedLayerFrame {
            frame: pixels.layer_frame([0., 0., 1., 1.]),
            crop: [0., 0., 1., 1.],
        };
        assert!(!surface.present_masked_cropped_layer(Some(&layer), Some(&pixels)));
        assert!(!surface.present_cropped_layer(Some(&layer)));
    }
}
