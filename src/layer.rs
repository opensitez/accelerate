use std::sync::Arc;

/// A retained texture sampled through a normalized source rectangle.
#[derive(Clone)]
pub struct CroppedLayerFrame {
    pub frame: LayerFrame,
    pub crop: [f32; 4],
}

impl CroppedLayerFrame {
    pub fn is_valid(&self) -> bool {
        let [x, y, width, height] = self.crop;
        self.crop.iter().all(|value| value.is_finite())
            && x >= 0.0
            && y >= 0.0
            && width > 0.0
            && height > 0.0
            && x + width <= 1.000001
            && y + height <= 1.000001
            && [
                self.frame.x,
                self.frame.y,
                self.frame.width,
                self.frame.height,
            ]
            .iter()
            .all(|value| value.is_finite())
            && self.frame.width > 0.0
            && self.frame.height > 0.0
            && self.frame.corner_radius.is_finite()
            && super::software::rgba_size_matches(
                self.frame.rgba.len(),
                self.frame.source_width,
                self.frame.source_height,
            )
    }

    /// Software composition and capture use the same source crop as native layers.
    pub fn draw_software(&self, target: &mut tiny_skia::Pixmap, scale: f32) -> bool {
        self.draw_software_masked(target, scale, None)
    }

    /// Alpha coverage spans the entire destination independently of the source
    /// crop. RGB and presentation size are ignored. Sampling is bilinear with
    /// clamped edges; cloning a mask retains its immutable pixel allocation.
    /// Destination-physical-pixel dimensions avoid resampling at the given scale.
    pub fn draw_software_masked(
        &self,
        target: &mut tiny_skia::Pixmap,
        scale: f32,
        coverage: Option<&crate::RgbaSurface>,
    ) -> bool {
        if !self.is_valid() || !valid_mask(coverage) || !scale.is_finite() || scale <= 0.0 {
            return false;
        }
        let frame = &self.frame;
        let Some(rect) = tiny_skia::Rect::from_xywh(
            frame.x * scale,
            frame.y * scale,
            frame.width * scale,
            frame.height * scale,
        ) else {
            return false;
        };
        let Some(mut mask) = tiny_skia::Mask::new(target.width(), target.height()) else {
            return false;
        };
        let copied = coverage.is_some_and(|coverage| copy_aligned_mask(&mut mask, coverage, rect));
        if !copied {
            mask.fill_path(
                &tiny_skia::PathBuilder::from_rect(rect),
                tiny_skia::FillRule::Winding,
                false,
                tiny_skia::Transform::identity(),
            );
        }
        if let Some(coverage) = coverage.filter(|_| !copied) {
            let (width, height) = coverage.dimensions();
            for (index, alpha) in mask.data_mut().iter_mut().enumerate() {
                if *alpha == 0 {
                    continue;
                }
                let px = (index % target.width() as usize) as f32 + 0.5;
                let py = (index / target.width() as usize) as f32 + 0.5;
                let x = ((px - rect.x()) / rect.width() * width as f32 - 0.5)
                    .clamp(0.0, (width - 1) as f32);
                let y = ((py - rect.y()) / rect.height() * height as f32 - 0.5)
                    .clamp(0.0, (height - 1) as f32);
                let x0 = x.floor() as u32;
                let y0 = y.floor() as u32;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);
                let sample = |x: u32, y: u32| {
                    coverage.pixels()[(y as usize * width as usize + x as usize) * 4 + 3] as f32
                };
                let top = sample(x0, y0) + (sample(x1, y0) - sample(x0, y0)) * (x - x0 as f32);
                let bottom = sample(x0, y1) + (sample(x1, y1) - sample(x0, y1)) * (x - x0 as f32);
                *alpha = (top + (bottom - top) * (y - y0 as f32)).round() as u8;
            }
        }
        let [x, y, width, height] = self.crop;
        let sx = rect.width() / (width * frame.source_width as f32);
        let sy = rect.height() / (height * frame.source_height as f32);
        let transform = tiny_skia::Transform::from_row(
            sx,
            0.0,
            0.0,
            sy,
            rect.x() - x * frame.source_width as f32 * sx,
            rect.y() - y * frame.source_height as f32 * sy,
        );
        if !transform.is_finite() {
            return false;
        }
        let Some(source) =
            tiny_skia::PixmapRef::from_bytes(&frame.rgba, frame.source_width, frame.source_height)
        else {
            return false;
        };
        crate::draw_surface(
            target,
            source,
            0,
            0,
            &tiny_skia::PixmapPaint::default(),
            transform,
            Some(&mask),
        );
        true
    }
}

// A physical-size mask already contains destination coverage. Extract only the
// visible alpha rows, avoiding both rectangle rasterization and resampling.
fn copy_aligned_mask(
    mask: &mut tiny_skia::Mask,
    coverage: &crate::RgbaSurface,
    rect: tiny_skia::Rect,
) -> bool {
    let (width, height) = coverage.dimensions();
    if rect.x().fract() != 0.0
        || rect.y().fract() != 0.0
        || rect.width() as f64 != width as f64
        || rect.height() as f64 != height as f64
    {
        return false;
    }
    let target_width = mask.width() as usize;
    let target_height = mask.height() as usize;
    let x = rect.x() as f64;
    let y = rect.y() as f64;
    let left = x.clamp(0.0, target_width as f64) as usize;
    let top = y.clamp(0.0, target_height as f64) as usize;
    let right = (x + width as f64).clamp(0.0, target_width as f64) as usize;
    let bottom = (y + height as f64).clamp(0.0, target_height as f64) as usize;
    if left >= right || top >= bottom {
        return true;
    }
    let source_x = (left as f64 - x) as usize;
    let source_y = (top as f64 - y) as usize;
    let row_width = right - left;
    for (row, target_y) in (top..bottom).enumerate() {
        let source_start = ((source_y + row) * width as usize + source_x) * 4;
        let source = &coverage.pixels()[source_start..source_start + row_width * 4];
        let destination =
            &mut mask.data_mut()[target_y * target_width + left..target_y * target_width + right];
        for (alpha, pixel) in destination.iter_mut().zip(source.chunks_exact(4)) {
            *alpha = pixel[3];
        }
    }
    true
}

#[cfg(test)]
mod crop_tests {
    use super::*;

    #[test]
    fn crop_keeps_shared_pixels_and_does_not_paint_outside_destination() {
        let pixels = Arc::new(vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255,
        ]);
        let surface = crate::RgbaSurface::new(pixels.clone(), 2, 2).unwrap();
        let layer = CroppedLayerFrame {
            frame: surface.layer_frame([1.0, 1.0, 2.0, 1.0]),
            crop: [0.0, 0.5, 1.0, 0.5],
        };
        assert!(Arc::ptr_eq(&pixels, &layer.frame.rgba));
        let mut target = tiny_skia::Pixmap::new(4, 3).unwrap();
        assert!(layer.draw_software(&mut target, 1.0));
        assert_eq!(target.pixel(1, 1).unwrap().blue(), 255);
        assert_eq!(target.pixel(2, 1).unwrap().red(), 255);
        assert_eq!(target.pixel(0, 1).unwrap().alpha(), 0);
        assert_eq!(target.pixel(1, 2).unwrap().alpha(), 0);
        let mut bad = layer;
        bad.crop[1] = f32::NAN;
        let before = target.clone();
        assert!(!bad.draw_software(&mut target, 1.0));
        assert_eq!(target.data(), before.data());
    }

    #[test]
    fn destination_mask_alpha_is_independent_of_source_crop() {
        let source = crate::RgbaSurface::new(
            Arc::new(vec![
                255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255,
            ]),
            2,
            2,
        )
        .unwrap();
        let mask =
            crate::RgbaSurface::new(Arc::new(vec![0, 0, 0, 0, 0, 128, 0, 128]), 2, 1).unwrap();
        let mut layer = CroppedLayerFrame {
            frame: source.layer_frame([1.0, 1.0, 2.0, 1.0]),
            crop: [0.0, 0.0, 1.0, 0.5],
        };
        for (crop_y, expected) in [(0.0, [128, 0, 0, 128]), (0.5, [0, 0, 128, 128])] {
            layer.crop[1] = crop_y;
            let mut target = tiny_skia::Pixmap::new(4, 3).unwrap();
            assert!(layer.draw_software_masked(&mut target, 1.0, Some(&mask)));
            assert_eq!(target.pixel(1, 1).unwrap().alpha(), 0);
            assert_eq!(&target.data()[(1 * 4 + 2) * 4..(1 * 4 + 3) * 4], &expected);
            assert_eq!(target.pixel(0, 1).unwrap().alpha(), 0);
            assert_eq!(target.pixel(2, 2).unwrap().alpha(), 0);
        }
    }

    #[test]
    fn destination_mask_scales_bilinearly_and_clamps_edges() {
        let source = crate::RgbaSurface::new(Arc::new(vec![255; 4]), 1, 1).unwrap();
        let mask =
            crate::RgbaSurface::new(Arc::new(vec![0, 0, 0, 0, 255, 255, 255, 255]), 2, 1).unwrap();
        let layer = CroppedLayerFrame {
            frame: source.layer_frame([0.0, 0.0, 2.0, 1.0]),
            crop: [0.0, 0.0, 1.0, 1.0],
        };
        let mut target = tiny_skia::Pixmap::new(4, 2).unwrap();
        assert!(layer.draw_software_masked(&mut target, 2.0, Some(&mask)));
        for y in 0..2 {
            for (x, alpha) in [0, 64, 191, 255].into_iter().enumerate() {
                assert_eq!(target.pixel(x as u32, y).unwrap().alpha(), alpha);
            }
        }
        let mut ordinary = tiny_skia::Pixmap::new(4, 2).unwrap();
        let mut unmasked = ordinary.clone();
        assert!(layer.draw_software(&mut ordinary, 2.0));
        assert!(layer.draw_software_masked(&mut unmasked, 2.0, None));
        assert_eq!(ordinary.data(), unmasked.data());
    }

    #[test]
    fn aligned_physical_mask_copies_clipped_alpha_rows_at_both_scales() {
        let pixels = Arc::new(vec![
            0, 0, 0, 0, 64, 64, 64, 64, 128, 128, 128, 128, 191, 191, 191, 191, 255, 255, 255, 255,
            32, 32, 32, 32,
        ]);
        let coverage = crate::RgbaSurface::new(pixels.clone(), 3, 2).unwrap();
        let source = crate::RgbaSurface::new(Arc::new(vec![255; 4]), 1, 1).unwrap();
        for scale in [1.0, 2.0] {
            let layer = CroppedLayerFrame {
                frame: source.layer_frame([-1.0 / scale, 1.0 / scale, 3.0 / scale, 2.0 / scale]),
                crop: [0.0, 0.0, 1.0, 1.0],
            };
            let mut target = tiny_skia::Pixmap::new(3, 2).unwrap();
            assert!(layer.draw_software_masked(&mut target, scale, Some(&coverage)));
            assert_eq!(target.pixel(0, 1).unwrap().alpha(), 64);
            assert_eq!(target.pixel(1, 1).unwrap().alpha(), 128);
            assert_eq!(target.pixel(2, 1).unwrap().alpha(), 0);
            assert_eq!(target.pixel(0, 0).unwrap().alpha(), 0);
            assert!(Arc::ptr_eq(&pixels, coverage.pixels()));
        }
        let mut mask = tiny_skia::Mask::new(3, 2).unwrap();
        assert!(copy_aligned_mask(
            &mut mask,
            &coverage,
            tiny_skia::Rect::from_xywh(1.0, -1.0, 3.0, 2.0).unwrap()
        ));
        assert_eq!(mask.data(), &[0, 191, 255, 0, 0, 0]);
        assert!(!copy_aligned_mask(
            &mut mask,
            &coverage,
            tiny_skia::Rect::from_xywh(0.5, 0.0, 3.0, 2.0).unwrap()
        ));
        assert!(!copy_aligned_mask(
            &mut mask,
            &coverage,
            tiny_skia::Rect::from_xywh(0.0, 0.0, 6.0, 4.0).unwrap()
        ));
    }

    #[test]
    fn physical_mask_rounding_matches_reference_for_every_alpha() {
        let mut mask_bytes = Vec::new();
        for alpha in 0..=255u8 {
            mask_bytes.extend_from_slice(&[alpha; 4]);
        }
        let coverage = crate::RgbaSurface::new(Arc::new(mask_bytes), 256, 1).unwrap();
        for color in [[255, 0, 0, 255], [37, 91, 173, 255], [31, 63, 95, 127]] {
            let source = crate::RgbaSurface::new(Arc::new(color.repeat(256)), 256, 1).unwrap();
            for scale in [1.0, 2.0] {
                let layer = CroppedLayerFrame {
                    frame: source.layer_frame([
                        1.0 / scale,
                        1.0 / scale,
                        256.0 / scale,
                        1.0 / scale,
                    ]),
                    crop: [0.0, 0.0, 1.0, 1.0],
                };
                let mut actual = tiny_skia::Pixmap::new(258, 3).unwrap();
                actual.fill(tiny_skia::Color::from_rgba8(113, 149, 197, 255));
                let mut expected = actual.clone();
                let mut reference_mask = tiny_skia::Mask::new(258, 3).unwrap();
                for alpha in 0..=255usize {
                    reference_mask.data_mut()[258 + 1 + alpha] = alpha as u8;
                }
                expected.draw_pixmap(
                    1,
                    1,
                    source.as_pixmap(),
                    &tiny_skia::PixmapPaint::default(),
                    tiny_skia::Transform::identity(),
                    Some(&reference_mask),
                );
                assert!(layer.draw_software_masked(&mut actual, scale, Some(&coverage)));
                assert_eq!(
                    actual.data(),
                    expected.data(),
                    "color={color:?}, scale={scale}"
                );
            }
        }
    }

    #[test]
    fn rounded_mask_pixels_match_direct_tiny_skia_clip_exactly() {
        for scale in [1.0, 2.0] {
            let size = (32.0 * scale) as u32;
            let mut path = tiny_skia::PathBuilder::new();
            path.push_circle(size as f32 / 2.0, size as f32 / 2.0, size as f32 / 2.0);
            let path = path.finish().unwrap();
            let mut reference_mask = tiny_skia::Mask::new(size, size).unwrap();
            reference_mask.fill_path(
                &path,
                tiny_skia::FillRule::Winding,
                true,
                tiny_skia::Transform::identity(),
            );
            let mut mask_pixels = tiny_skia::Pixmap::new(size, size).unwrap();
            let rect = tiny_skia::Rect::from_xywh(0.0, 0.0, size as f32, size as f32).unwrap();
            let mut white = tiny_skia::Paint::default();
            white.set_color_rgba8(255, 255, 255, 255);
            mask_pixels.fill_rect(
                rect,
                &white,
                tiny_skia::Transform::identity(),
                Some(&reference_mask),
            );
            let coverage =
                crate::RgbaSurface::new(Arc::new(mask_pixels.take()), size, size).unwrap();
            assert_eq!(
                coverage
                    .pixels()
                    .chunks_exact(4)
                    .map(|pixel| pixel[3])
                    .collect::<Vec<_>>(),
                reference_mask.data()
            );
            let mut red = tiny_skia::Pixmap::new(size, size).unwrap();
            red.fill(tiny_skia::Color::from_rgba8(255, 0, 0, 255));
            let source = crate::RgbaSurface::new(Arc::new(red.take()), size, size).unwrap();
            let layer = CroppedLayerFrame {
                frame: source.layer_frame([0.0, 0.0, 32.0, 32.0]),
                crop: [0.0, 0.0, 1.0, 1.0],
            };
            let mut expected = tiny_skia::Pixmap::new(size, size).unwrap();
            expected.fill(tiny_skia::Color::WHITE);
            let mut actual = expected.clone();
            let mut paint = tiny_skia::Paint::default();
            paint.set_color_rgba8(255, 0, 0, 255);
            expected.fill_rect(
                rect,
                &paint,
                tiny_skia::Transform::identity(),
                Some(&reference_mask),
            );
            assert!(layer.draw_software_masked(&mut actual, scale, Some(&coverage)));
            assert_eq!(actual.data(), expected.data(), "scale={scale}");
        }
    }

    #[test]
    fn invalid_mask_storage_and_nonfinite_geometry_are_rejected() {
        assert!(crate::RgbaSurface::new(Arc::new(vec![0; 3]), 1, 1).is_err());
        assert!(crate::RgbaSurface::new(Arc::new(vec![]), 0, 1).is_err());
        let source = crate::RgbaSurface::new(Arc::new(vec![255; 4]), 1, 1).unwrap();
        let mut layer = CroppedLayerFrame {
            frame: source.layer_frame([0.0, 0.0, 1.0, 1.0]),
            crop: [0.0, 0.0, 1.0, 1.0],
        };
        let mut target = tiny_skia::Pixmap::new(1, 1).unwrap();
        for value in [f32::NAN, f32::INFINITY] {
            layer.frame.x = value;
            assert!(!layer.draw_software_masked(&mut target, 1.0, Some(&source)));
            assert_eq!(target.data(), &[0; 4]);
        }
    }
}

pub(crate) fn valid_mask(mask: Option<&crate::RgbaSurface>) -> bool {
    mask.is_none_or(|mask| {
        let (width, height) = mask.dimensions();
        super::software::rgba_size_matches(mask.pixels().len(), width, height)
            && width <= i32::MAX as u32
            && height <= i32::MAX as u32
    })
}

#[derive(Clone)]
pub struct ForegroundLayer {
    pub rgba: Arc<Vec<u8>>,
    pub source_width: u32,
    pub source_height: u32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Retained premultiplied RGBA pixels with a logical-window destination.
#[derive(Clone)]
pub struct LayerFrame {
    pub rgba: Arc<Vec<u8>>,
    pub source_width: u32,
    pub source_height: u32,
    pub presentation_size: Option<(u32, u32)>,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub cover: bool,
    pub corner_radius: f32,
    pub tint: Option<[u8; 4]>,
    pub foreground: Option<ForegroundLayer>,
}

impl LayerFrame {
    /// Normalized centered source crop, shared by native layers and capture.
    pub fn contents_rect(&self) -> (f64, f64, f64, f64) {
        let (width, height) = self
            .presentation_size
            .filter(|&(width, height)| width > 0 && height > 0)
            .unwrap_or((self.source_width, self.source_height));
        if !self.cover
            || width == 0
            || height == 0
            || !self.width.is_finite()
            || !self.height.is_finite()
            || self.width <= 0.0
            || self.height <= 0.0
        {
            return (0.0, 0.0, 1.0, 1.0);
        }
        let source_aspect = f64::from(width) / f64::from(height);
        let destination_aspect = f64::from(self.width) / f64::from(self.height);
        if source_aspect > destination_aspect {
            let crop_width = destination_aspect / source_aspect;
            ((1.0 - crop_width) * 0.5, 0.0, crop_width, 1.0)
        } else {
            let crop_height = source_aspect / destination_aspect;
            (0.0, (1.0 - crop_height) * 0.5, 1.0, crop_height)
        }
    }

    pub fn is_opaque(&self) -> bool {
        super::software::rgba_size_matches(self.rgba.len(), self.source_width, self.source_height)
            && super::software::rgba_is_opaque(&self.rgba)
    }
}

pub fn tinted_pixels(pixels: &Arc<Vec<u8>>, tint: Option<[u8; 4]>) -> Arc<Vec<u8>> {
    let Some([r, g, b, alpha]) = tint.filter(|color| color[3] != 0) else {
        return pixels.clone();
    };
    let table = [r, g, b].map(|channel| {
        std::array::from_fn::<_, 256, _>(|value| {
            ((value as u32 * ((255 - alpha as u32) * 255 + alpha as u32 * channel as u32) + 32512)
                / 65025) as u8
        })
    });
    let mut output = Vec::with_capacity(pixels.len());
    for pixel in pixels.chunks_exact(4) {
        output.extend_from_slice(&[
            table[0][pixel[0] as usize],
            table[1][pixel[1] as usize],
            table[2][pixel[2] as usize],
            pixel[3],
        ]);
    }
    Arc::new(output)
}
