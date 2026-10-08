use crate::LayerFrame;
use std::sync::Arc;
use tiny_skia::{Mask, Pixmap, PixmapPaint, PixmapRef, Transform};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidRgbaSurface;

impl std::fmt::Display for InvalidRgbaSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RGBA storage does not match its nonzero dimensions")
    }
}
impl std::error::Error for InvalidRgbaSurface {}

/// Immutable, tightly packed premultiplied RGBA shared by decoders and presenters.
/// Construction and cloning do not initialize a GPU or copy pixel data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbaSurface {
    pixels: Arc<Vec<u8>>,
    width: u32,
    height: u32,
    presentation_size: Option<(u32, u32)>,
}

impl RgbaSurface {
    pub fn new(pixels: Arc<Vec<u8>>, width: u32, height: u32) -> Result<Self, InvalidRgbaSurface> {
        if !crate::software::rgba_size_matches(pixels.len(), width, height)
            || PixmapRef::from_bytes(&pixels, width, height).is_none()
        {
            return Err(InvalidRgbaSurface);
        }
        Ok(Self {
            pixels,
            width,
            height,
            presentation_size: None,
        })
    }

    pub fn with_presentation_size(mut self, size: Option<(u32, u32)>) -> Self {
        self.presentation_size = size.filter(|&(w, h)| w != 0 && h != 0);
        self
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    pub fn display_dimensions(&self) -> (u32, u32) {
        self.presentation_size.unwrap_or(self.dimensions())
    }
    pub fn pixels(&self) -> &Arc<Vec<u8>> {
        &self.pixels
    }
    pub fn into_pixels(self) -> Arc<Vec<u8>> {
        self.pixels
    }
    pub fn as_pixmap(&self) -> PixmapRef<'_> {
        PixmapRef::from_bytes(&self.pixels, self.width, self.height)
            .expect("validated RGBA surface")
    }

    pub fn layer_frame(&self, destination: [f32; 4]) -> LayerFrame {
        LayerFrame {
            rgba: self.pixels.clone(),
            source_width: self.width,
            source_height: self.height,
            presentation_size: self.presentation_size,
            x: destination[0],
            y: destination[1],
            width: destination[2],
            height: destination[3],
            cover: false,
            corner_radius: 0.,
            tint: None,
            foreground: None,
        }
    }

    pub fn draw(
        &self,
        target: &mut Pixmap,
        x: i32,
        y: i32,
        paint: &PixmapPaint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        crate::draw_surface(target, self.as_pixmap(), x, y, paint, transform, mask);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn media_layer_and_software_draw_share_pixels_without_copying() {
        let pixels = Arc::new(vec![40, 80, 120, 255]);
        let surface = RgbaSurface::new(pixels.clone(), 1, 1)
            .unwrap()
            .with_presentation_size(Some((2, 1)));
        let frame = surface.layer_frame([5., 6., 8., 4.]);
        assert!(Arc::ptr_eq(&pixels, &frame.rgba));
        assert_eq!((frame.source_width, frame.source_height), (1, 1));
        assert_eq!(frame.presentation_size, Some((2, 1)));
        let mut target = Pixmap::new(2, 2).unwrap();
        surface.draw(
            &mut target,
            1,
            1,
            &PixmapPaint::default(),
            Transform::identity(),
            None,
        );
        assert_eq!(&target.data()[12..16], pixels.as_slice());
        assert!(Arc::ptr_eq(&pixels, &surface.into_pixels()));
    }
    #[test]
    fn malformed_and_overflowing_storage_is_rejected() {
        for (bytes, width, height) in [(0, 0, 0), (3, 1, 1), (0, u32::MAX, u32::MAX)] {
            assert!(RgbaSurface::new(Arc::new(vec![0; bytes]), width, height).is_err());
        }
    }
}
