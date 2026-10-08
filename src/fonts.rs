//! Bulk blending of packed grayscale glyph masks into premultiplied RGBA pixels.
//! Rasterization and glyph caching remain the caller's responsibility.

use tiny_skia::{Mask, Pixmap};

/// Existing clients use different integer source-over rounding conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendRounding {
    /// Floor the combined source and destination channel numerator (Canvas).
    Combined,
    /// Floor each term before adding the channels (webcore text).
    Separate,
}

/// Blend a tightly packed, one-byte-per-pixel glyph coverage mask.
///
/// `color` contains straight RGB and caller/base alpha, not premultiplied RGB.
/// Alpha is floored after multiplying coverage, then floored again after clip.
/// The optional clip must have exactly the target's dimensions.
///
/// Returns false, without modifying the target, for zero dimensions, arithmetic
/// overflow, nonexact coverage length or a mismatched clip. Valid transparent
/// and fully offscreen masks return true. Extreme coordinates are safe no-ops.
#[allow(clippy::too_many_arguments)]
pub fn blit_mask(
    target: &mut Pixmap,
    coverage: &[u8],
    width: u32,
    height: u32,
    x: i64,
    y: i64,
    color: [u8; 4],
    clip: Option<&Mask>,
    rounding: BlendRounding,
) -> bool {
    blit_mask_impl::<false, true>(target, coverage, width, height, x, y, color, clip, rounding)
}

/// Opt-in run-aware version of [`blit_mask`], with identical validation and
/// rounding. This alternative specializes uniform coverage runs.
#[allow(clippy::too_many_arguments)]
pub fn blit_mask_accelerated(
    target: &mut Pixmap,
    coverage: &[u8],
    width: u32,
    height: u32,
    x: i64,
    y: i64,
    color: [u8; 4],
    clip: Option<&Mask>,
    rounding: BlendRounding,
) -> bool {
    blit_mask_impl::<true, false>(target, coverage, width, height, x, y, color, clip, rounding)
}

#[allow(clippy::too_many_arguments)]
fn blit_mask_impl<const ACCELERATED: bool, const SIMD: bool>(
    target: &mut Pixmap,
    coverage: &[u8],
    width: u32,
    height: u32,
    x: i64,
    y: i64,
    color: [u8; 4],
    clip: Option<&Mask>,
    rounding: BlendRounding,
) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    let Some(source_len) = (width as usize).checked_mul(height as usize) else {
        return false;
    };
    if coverage.len() != source_len {
        return false;
    }
    let Some(target_len) = (target.width() as usize).checked_mul(target.height() as usize) else {
        return false;
    };
    if target_len.checked_mul(4) != Some(target.data().len())
        || clip.is_some_and(|mask| {
            mask.width() != target.width()
                || mask.height() != target.height()
                || mask.data().len() != target_len
        })
    {
        return false;
    }
    if color[3] == 0 {
        return true;
    }

    let left = x.max(0);
    let top = y.max(0);
    let right = x
        .saturating_add(i64::from(width))
        .min(i64::from(target.width()));
    let bottom = y
        .saturating_add(i64::from(height))
        .min(i64::from(target.height()));
    if left >= right || top >= bottom {
        return true;
    }
    // For an intersecting span these differences are bounded by the source
    // dimensions, even when the original coordinates approach i64's limits.
    let span = Span {
        source: (top - y) as usize * width as usize + (left - x) as usize,
        target: top as usize * target.width() as usize + left as usize,
        width: (right - left) as usize,
        height: (bottom - top) as usize,
    };
    match rounding {
        BlendRounding::Combined => {
            if ACCELERATED {
                blend_rows_accelerated::<false>(target, coverage, width as usize, span, color, clip)
            } else {
                blend_rows::<false, SIMD>(target, coverage, width as usize, span, color, clip)
            }
        }
        BlendRounding::Separate => {
            if ACCELERATED {
                blend_rows_accelerated::<true>(target, coverage, width as usize, span, color, clip)
            } else {
                blend_rows::<true, SIMD>(target, coverage, width as usize, span, color, clip)
            }
        }
    }
    true
}

struct Span {
    source: usize,
    target: usize,
    width: usize,
    height: usize,
}

fn blend_rows<const SEPARATE: bool, const SIMD: bool>(
    target: &mut Pixmap,
    coverage: &[u8],
    source_width: usize,
    span: Span,
    color: [u8; 4],
    clip: Option<&Mask>,
) {
    let target_width = target.width() as usize;
    let color = color.map(u32::from);
    let pixels = target.data_mut();
    let clip = clip.map(Mask::data);
    for row in 0..span.height {
        let source = span.source + row * source_width;
        let destination = span.target + row * target_width;
        let coverage = &coverage[source..source + span.width];
        let pixels = &mut pixels[destination * 4..(destination + span.width) * 4];
        #[cfg(target_arch = "aarch64")]
        if SIMD && span.width >= 8 {
            // Every chunk is bounded by the validated clipped row slices.
            unsafe {
                blend_row_neon::<SEPARATE>(
                    pixels,
                    coverage,
                    clip.map(|mask| &mask[destination..destination + span.width]),
                    color,
                );
            }
            continue;
        }
        if let Some(clip) = clip {
            let clip = &clip[destination..destination + span.width];
            for ((pixel, &coverage), &clip) in pixels.chunks_exact_mut(4).zip(coverage).zip(clip) {
                blend_pixel_scalar::<SEPARATE>(pixel, coverage, clip, color);
            }
        } else {
            for (pixel, &coverage) in pixels.chunks_exact_mut(4).zip(coverage) {
                blend_pixel_scalar::<SEPARATE>(pixel, coverage, 255, color);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn blend_row_neon<const SEPARATE: bool>(
    pixels: &mut [u8],
    coverage: &[u8],
    clip: Option<&[u8]>,
    color: [u32; 4],
) {
    use std::arch::aarch64::*;
    // Exact floor(x / 255) for products and combined numerators up to 65025.
    #[inline]
    unsafe fn divide(value: uint16x8_t) -> uint8x8_t {
        unsafe {
            vmovn_u16(vshrq_n_u16::<8>(vaddq_u16(
                vaddq_u16(value, vdupq_n_u16(1)),
                vshrq_n_u16::<8>(value),
            )))
        }
    }
    unsafe {
        let mut offset = 0;
        while offset + 8 <= coverage.len() {
            let mut alpha = divide(vmull_u8(
                vld1_u8(coverage.as_ptr().add(offset)),
                vdup_n_u8(color[3] as u8),
            ));
            if let Some(clip) = clip {
                alpha = divide(vmull_u8(alpha, vld1_u8(clip.as_ptr().add(offset))));
            }
            let inverse = vsub_u8(vdup_n_u8(255), alpha);
            let destination = vld4_u8(pixels.as_ptr().add(offset * 4));
            let blend = |source: u32, destination: uint8x8_t| {
                let source = vmull_u8(vdup_n_u8(source as u8), alpha);
                let destination = vmull_u8(destination, inverse);
                if SEPARATE {
                    vadd_u8(divide(source), divide(destination))
                } else {
                    divide(vaddq_u16(source, destination))
                }
            };
            let result = uint8x8x4_t(
                blend(color[0], destination.0),
                blend(color[1], destination.1),
                blend(color[2], destination.2),
                vadd_u8(alpha, divide(vmull_u8(destination.3, inverse))),
            );
            vst4_u8(pixels.as_mut_ptr().add(offset * 4), result);
            offset += 8;
        }
        for index in offset..coverage.len() {
            blend_pixel_scalar::<SEPARATE>(
                &mut pixels[index * 4..index * 4 + 4],
                coverage[index],
                clip.map_or(255, |mask| mask[index]),
                color,
            );
        }
    }
}

#[inline]
fn blend_pixel_scalar<const SEPARATE: bool>(
    pixel: &mut [u8],
    coverage: u8,
    clip: u8,
    color: [u32; 4],
) {
    if coverage == 0 || clip == 0 {
        return;
    }
    let alpha = color[3] * u32::from(coverage) / 255;
    let alpha = alpha * u32::from(clip) / 255;
    if alpha == 0 {
        return;
    }
    if alpha == 255 {
        pixel.copy_from_slice(&[color[0] as u8, color[1] as u8, color[2] as u8, 255]);
        return;
    }
    let inverse = 255 - alpha;
    for channel in 0..3 {
        let source = color[channel] * alpha;
        let destination = u32::from(pixel[channel]) * inverse;
        pixel[channel] = if SEPARATE {
            (source / 255 + destination / 255) as u8
        } else {
            ((source + destination) / 255) as u8
        };
    }
    pixel[3] = (alpha + u32::from(pixel[3]) * inverse / 255) as u8;
}

fn blend_rows_accelerated<const SEPARATE: bool>(
    target: &mut Pixmap,
    coverage: &[u8],
    source_width: usize,
    span: Span,
    color: [u8; 4],
    clip: Option<&Mask>,
) {
    let target_width = target.width() as usize;
    let color = MaskColor::new(color);
    let full = color.prepare::<SEPARATE>(u32::from(color.rgba[3]));
    let pixels = target.data_mut();
    let clip = clip.map(Mask::data);
    for row in 0..span.height {
        let source = span.source + row * source_width;
        let destination = span.target + row * target_width;
        let coverage = &coverage[source..source + span.width];
        let pixels = &mut pixels[destination * 4..(destination + span.width) * 4];
        if let Some(clip) = clip {
            let clip = &clip[destination..destination + span.width];
            blend_row::<SEPARATE, true>(pixels, coverage, clip, &color, full);
        } else {
            blend_row::<SEPARATE, false>(pixels, coverage, &[], &color, full);
        }
    }
}

struct MaskColor {
    rgba: [u8; 4],
    rgb: [u32; 3],
    grayscale: bool,
}

#[derive(Clone, Copy)]
struct PixelBlend {
    alpha: u32,
    inverse: u32,
    source: [u32; 3],
}

impl MaskColor {
    fn new(rgba: [u8; 4]) -> Self {
        Self {
            rgba,
            rgb: [u32::from(rgba[0]), u32::from(rgba[1]), u32::from(rgba[2])],
            grayscale: rgba[0] == rgba[1] && rgba[1] == rgba[2],
        }
    }

    #[inline]
    fn prepare<const SEPARATE: bool>(&self, alpha: u32) -> PixelBlend {
        let term = |channel| {
            let value = channel * alpha;
            if SEPARATE { value / 255 } else { value }
        };
        let source = if self.grayscale {
            [term(self.rgb[0]); 3]
        } else {
            self.rgb.map(term)
        };
        PixelBlend {
            alpha,
            inverse: 255 - alpha,
            source,
        }
    }
}

fn blend_row<const SEPARATE: bool, const MASKED: bool>(
    pixels: &mut [u8],
    coverage: &[u8],
    clip: &[u8],
    color: &MaskColor,
    full: PixelBlend,
) {
    let mut offset = 0;
    while offset < coverage.len() {
        if coverage[offset] == 0 {
            offset += 1 + zero_prefix(&coverage[offset + 1..]);
            continue;
        }
        let mask = if MASKED { clip[offset] } else { 255 };
        if mask == 0 {
            offset += 1 + zero_prefix(&clip[offset + 1..]);
            continue;
        }
        if coverage[offset] == 255 {
            // One source-color preparation per full-coverage run, including a
            // constant partial clip. Transparent runs never touch destination.
            let mut end = offset + 1;
            while end < coverage.len() && coverage[end] == 255 && (!MASKED || clip[end] == mask) {
                end += 1;
            }
            let blend = if mask == 255 {
                full
            } else {
                color.prepare::<SEPARATE>(full.alpha * u32::from(mask) / 255)
            };
            blend_span::<SEPARATE>(&mut pixels[offset * 4..end * 4], color.rgba, blend);
            offset = end;
            continue;
        }
        let mut alpha = if color.rgba[3] == 255 {
            u32::from(coverage[offset])
        } else {
            u32::from(color.rgba[3]) * u32::from(coverage[offset]) / 255
        };
        if mask != 255 {
            // Keep this second floor separate from coverage * base alpha.
            alpha = alpha * u32::from(mask) / 255;
        }
        if alpha != 0 {
            blend_pixel::<SEPARATE>(
                &mut pixels[offset * 4..offset * 4 + 4],
                color.prepare::<SEPARATE>(alpha),
            );
        }
        offset += 1;
    }
}

#[inline]
fn zero_prefix(values: &[u8]) -> usize {
    values
        .iter()
        .position(|&value| value != 0)
        .unwrap_or(values.len())
}

#[inline]
fn blend_span<const SEPARATE: bool>(pixels: &mut [u8], rgba: [u8; 4], blend: PixelBlend) {
    if blend.alpha == 0 {
        return;
    }
    if blend.alpha == 255 {
        for pixel in pixels.chunks_exact_mut(4) {
            pixel.copy_from_slice(&rgba);
        }
    } else {
        for pixel in pixels.chunks_exact_mut(4) {
            blend_pixel::<SEPARATE>(pixel, blend);
        }
    }
}

#[inline]
fn blend_pixel<const SEPARATE: bool>(pixel: &mut [u8], blend: PixelBlend) {
    for channel in 0..3 {
        let destination = u32::from(pixel[channel]) * blend.inverse;
        pixel[channel] = if SEPARATE {
            (blend.source[channel] + destination / 255) as u8
        } else {
            ((blend.source[channel] + destination) / 255) as u8
        };
    }
    pixel[3] = (blend.alpha + u32::from(pixel[3]) * blend.inverse / 255) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiny_skia::PremultipliedColorU8;

    fn background() -> Pixmap {
        background_sized(7, 5)
    }

    fn background_sized(width: u32, height: u32) -> Pixmap {
        let mut pixmap = Pixmap::new(width, height).unwrap();
        for (index, pixel) in pixmap.pixels_mut().iter_mut().enumerate() {
            let alpha = (index * 53 % 256) as u8;
            *pixel =
                PremultipliedColorU8::from_rgba(alpha / 2, alpha / 3, alpha / 4, alpha).unwrap();
        }
        pixmap
    }

    #[allow(clippy::too_many_arguments)]
    fn reference(
        target: &mut Pixmap,
        coverage: &[u8],
        width: u32,
        height: u32,
        x: i64,
        y: i64,
        color: [u8; 4],
        clip: Option<&Mask>,
        rounding: BlendRounding,
    ) {
        let target_width = target.width();
        let target_height = target.height();
        for row in 0..height {
            for column in 0..width {
                let px = i128::from(x) + i128::from(column);
                let py = i128::from(y) + i128::from(row);
                if px < 0
                    || px >= i128::from(target_width)
                    || py < 0
                    || py >= i128::from(target_height)
                {
                    continue;
                }
                let index = py as usize * target_width as usize + px as usize;
                let a = u32::from(coverage[(row * width + column) as usize]) * u32::from(color[3])
                    / 255;
                let a = a * clip.map_or(255, |mask| u32::from(mask.data()[index])) / 255;
                if a == 0 {
                    continue;
                }
                let inverse = 255 - a;
                let dest = target.pixels()[index];
                let blend = |source: u8, destination: u8| match rounding {
                    BlendRounding::Combined => {
                        (u32::from(source) * a + u32::from(destination) * inverse) / 255
                    }
                    BlendRounding::Separate => {
                        u32::from(source) * a / 255 + u32::from(destination) * inverse / 255
                    }
                };
                target.pixels_mut()[index] = PremultipliedColorU8::from_rgba(
                    blend(color[0], dest.red()) as u8,
                    blend(color[1], dest.green()) as u8,
                    blend(color[2], dest.blue()) as u8,
                    (a + u32::from(dest.alpha()) * inverse / 255) as u8,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn both_rounding_modes_match_scalar_oracle_with_clips_and_extreme_origins() {
        let coverage = [0, 1, 2, 63, 127, 128, 254, 255, 7, 193, 32, 239];
        let mut clip = Mask::new(7, 5).unwrap();
        for (index, value) in clip.data_mut().iter_mut().enumerate() {
            *value = (index * 71 % 256) as u8;
        }
        for rounding in [BlendRounding::Combined, BlendRounding::Separate] {
            for alpha in [0, 1, 2, 17, 127, 128, 254, 255] {
                for x in [i64::MIN, -5, -4, -3, -1, 0, 3, 6, 7, i64::MAX] {
                    for y in [i64::MIN, -4, -3, -2, -1, 0, 2, 4, 5, i64::MAX] {
                        for clip in [None, Some(&clip)] {
                            let mut actual = background();
                            let mut expected = actual.clone();
                            let color = [239, 31, 137, alpha];
                            reference(&mut expected, &coverage, 4, 3, x, y, color, clip, rounding);
                            assert!(blit_mask(
                                &mut actual,
                                &coverage,
                                4,
                                3,
                                x,
                                y,
                                color,
                                clip,
                                rounding
                            ));
                            assert_eq!(
                                actual.data(),
                                expected.data(),
                                "{rounding:?}, alpha={alpha}, x={x}, y={y}, clip={}",
                                clip.is_some()
                            );
                            let mut accelerated = background();
                            assert!(blit_mask_accelerated(
                                &mut accelerated,
                                &coverage,
                                4,
                                3,
                                x,
                                y,
                                color,
                                clip,
                                rounding
                            ));
                            assert_eq!(accelerated.data(), expected.data());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn separate_and_combined_rounding_are_intentionally_distinct() {
        for (rounding, expected) in [(BlendRounding::Combined, 1), (BlendRounding::Separate, 0)] {
            let mut target = Pixmap::new(1, 1).unwrap();
            target.data_mut().copy_from_slice(&[1, 1, 1, 255]);
            assert!(blit_mask(
                &mut target,
                &[128],
                1,
                1,
                0,
                0,
                [1, 1, 1, 255],
                None,
                rounding
            ));
            assert_eq!(target.data(), &[expected, expected, expected, 255]);
        }
    }

    #[test]
    fn coverage_then_clip_each_floor_before_blending() {
        let mut clip = Mask::new(1, 1).unwrap();
        clip.data_mut()[0] = 129;
        for rounding in [BlendRounding::Combined, BlendRounding::Separate] {
            let mut target = Pixmap::new(1, 1).unwrap();
            assert!(blit_mask(
                &mut target,
                &[129],
                1,
                1,
                0,
                0,
                [255, 0, 0, 129],
                Some(&clip),
                rounding
            ));
            assert_eq!(target.data(), &[32, 0, 0, 32]);
        }
    }

    #[test]
    fn invalid_inputs_never_write_even_when_transparent_or_offscreen() {
        let original = background();
        for (coverage, width, height) in [
            (&[][..], 0, 1),
            (&[][..], 1, 0),
            (&[][..], 1, 1),
            (&[255][..], 2, 1),
            (&[255, 255][..], 1, 1),
            (&[][..], u32::MAX, u32::MAX),
        ] {
            let mut target = original.clone();
            assert!(!blit_mask(
                &mut target,
                coverage,
                width,
                height,
                0,
                0,
                [255; 4],
                None,
                BlendRounding::Combined
            ));
            assert_eq!(target.data(), original.data());
        }
        for (width, height) in [(6, 5), (7, 4)] {
            let clip = Mask::new(width, height).unwrap();
            for (x, alpha) in [(0, 255), (i64::MAX, 255), (0, 0)] {
                let mut target = original.clone();
                assert!(!blit_mask(
                    &mut target,
                    &[255],
                    1,
                    1,
                    x,
                    0,
                    [255, 0, 0, alpha],
                    Some(&clip),
                    BlendRounding::Separate
                ));
                assert_eq!(target.data(), original.data());
            }
        }
    }

    #[test]
    fn opaque_source_and_zero_coverage_preserve_premultiplication() {
        let mut target = Pixmap::new(3, 1).unwrap();
        target
            .data_mut()
            .copy_from_slice(&[4, 5, 6, 7, 0, 0, 0, 0, 31, 37, 41, 63]);
        assert!(blit_mask(
            &mut target,
            &[255, 0, 255],
            3,
            1,
            0,
            0,
            [123, 67, 89, 255],
            None,
            BlendRounding::Combined
        ));
        assert_eq!(
            target.data(),
            &[123, 67, 89, 255, 0, 0, 0, 0, 123, 67, 89, 255]
        );
        assert!(
            target
                .pixels()
                .iter()
                .all(|pixel| pixel.red() <= pixel.alpha()
                    && pixel.green() <= pixel.alpha()
                    && pixel.blue() <= pixel.alpha())
        );
    }

    #[test]
    fn accelerated_runs_match_reference_with_clipped_row_boundaries() {
        let coverage: Vec<u8> = (0..96 * 4)
            .map(|index| match (index % 96) / 16 {
                0 | 3 => 0,
                1 | 4 => 255,
                _ => (index * 73 % 256) as u8,
            })
            .collect();
        let mut clip = Mask::new(79, 6).unwrap();
        for (index, value) in clip.data_mut().iter_mut().enumerate() {
            *value = [0, 255, 127, 1, 254][(index % 79) / 16];
        }
        for rounding in [BlendRounding::Combined, BlendRounding::Separate] {
            for color in [
                [0, 0, 0, 255],
                [173, 173, 173, 255],
                [255, 37, 129, 255],
                [173, 173, 173, 129],
                [255, 37, 129, 129],
                [255, 37, 129, 1],
                [255, 37, 129, 0],
            ] {
                for (x, y) in [(-19, -1), (-1, 2), (0, 0), (17, 4)] {
                    for clip in [None, Some(&clip)] {
                        let mut actual = background_sized(79, 6);
                        let mut expected = actual.clone();
                        reference(&mut expected, &coverage, 96, 4, x, y, color, clip, rounding);
                        assert!(blit_mask_accelerated(
                            &mut actual,
                            &coverage,
                            96,
                            4,
                            x,
                            y,
                            color,
                            clip,
                            rounding
                        ));
                        assert_eq!(
                            actual.data(),
                            expected.data(),
                            "{rounding:?} {color:?} ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn accelerated_all_coverage_clip_pairs_preserve_both_rounding_policies() {
        let coverage: Vec<u8> = (0..256 * 256).map(|index| (index % 256) as u8).collect();
        let mut clip = Mask::new(256, 256).unwrap();
        for (index, value) in clip.data_mut().iter_mut().enumerate() {
            *value = (index / 256) as u8;
        }
        for rounding in [BlendRounding::Combined, BlendRounding::Separate] {
            for alpha in [1, 2, 17, 127, 129, 254, 255] {
                for rgb in [[239, 31, 137], [173; 3], [0; 3]] {
                    let color = [rgb[0], rgb[1], rgb[2], alpha];
                    let mut actual = background_sized(256, 256);
                    let mut expected = actual.clone();
                    assert!(blit_mask(
                        &mut expected,
                        &coverage,
                        256,
                        256,
                        0,
                        0,
                        color,
                        Some(&clip),
                        rounding
                    ));
                    assert!(blit_mask_accelerated(
                        &mut actual,
                        &coverage,
                        256,
                        256,
                        0,
                        0,
                        color,
                        Some(&clip),
                        rounding
                    ));
                    assert_eq!(actual.data(), expected.data(), "{rounding:?} {color:?}");
                }
            }
        }
    }

    #[test]
    fn accelerated_invalid_inputs_are_atomic() {
        let original = background();
        let wrong_clip = Mask::new(6, 5).unwrap();
        for rounding in [BlendRounding::Combined, BlendRounding::Separate] {
            for (coverage, width, height, clip) in [
                (&[][..], 0, 1, None),
                (&[][..], 1, 0, None),
                (&[][..], 1, 1, None),
                (&[255][..], 2, 1, None),
                (&[255, 255][..], 1, 1, None),
                (&[][..], u32::MAX, u32::MAX, None),
                (&[255][..], 1, 1, Some(&wrong_clip)),
            ] {
                for (x, alpha) in [(0, 255), (i64::MAX, 255), (0, 0)] {
                    let mut target = original.clone();
                    assert!(!blit_mask_accelerated(
                        &mut target,
                        coverage,
                        width,
                        height,
                        x,
                        0,
                        [255, 31, 137, alpha],
                        clip,
                        rounding
                    ));
                    assert_eq!(target.data(), original.data());
                }
            }
        }
    }

    #[test]
    #[ignore = "manual release A/B benchmark; no font system or caller wiring"]
    fn benchmark_scalar_vs_accelerated_cached_masks() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const ITERATIONS: usize = 1024;
        const RUNS: usize = 4;
        let runs: Vec<u8> = (0..32 * 16)
            .map(|index| {
                if (index % 32) < 8 || (index % 32) >= 24 {
                    0
                } else {
                    255
                }
            })
            .collect();
        let opaque = vec![255; 32 * 16];
        let antialias: Vec<u8> = (0..32 * 16).map(|index| (index * 73 % 256) as u8).collect();
        let mut clip = Mask::new(40, 24).unwrap();
        for (index, value) in clip.data_mut().iter_mut().enumerate() {
            *value = [0, 255, 127, 255][(index % 40) / 10];
        }
        let cases = [
            ("sparse-black", &runs, [0, 0, 0, 255], None),
            ("opaque-color", &opaque, [239, 31, 137, 255], None),
            ("antialias-gray", &antialias, [173, 173, 173, 255], None),
            ("translucent-runs", &runs, [239, 31, 137, 129], None),
            ("masked-runs", &runs, [173, 173, 173, 129], Some(&clip)),
            (
                "masked-antialias",
                &antialias,
                [239, 31, 137, 255],
                Some(&clip),
            ),
        ];
        for rounding in [BlendRounding::Combined, BlendRounding::Separate] {
            for (name, coverage, color, clip) in cases {
                let measure = |accelerated: bool, iterations: usize| {
                    let mut target = background_sized(40, 24);
                    let entry = if accelerated {
                        blit_mask_accelerated
                    } else {
                        blit_mask_scalar
                    };
                    let mut valid = true;
                    let start = Instant::now();
                    for _ in 0..iterations {
                        valid &= black_box(entry(
                            black_box(&mut target),
                            black_box(coverage),
                            32,
                            16,
                            -3,
                            4,
                            black_box(color),
                            clip,
                            rounding,
                        ));
                    }
                    let elapsed = start.elapsed();
                    assert!(valid);
                    black_box(target.data());
                    (elapsed, target)
                };
                // Warm both entries; alternate order to reduce ordering bias.
                let _ = measure(false, 32);
                let _ = measure(true, 32);
                let mut scalar_total = Duration::ZERO;
                let mut accelerated_total = Duration::ZERO;
                for run in 0..RUNS {
                    let (first_time, first) = measure(run % 2 != 0, ITERATIONS);
                    let (second_time, second) = measure(run % 2 == 0, ITERATIONS);
                    assert_eq!(first.data(), second.data());
                    let (scalar, accelerated) = if run % 2 == 0 {
                        (first_time, second_time)
                    } else {
                        (second_time, first_time)
                    };
                    scalar_total += scalar;
                    accelerated_total += accelerated;
                    eprintln!(
                        "{name} {rounding:?} run={run} scalar={scalar:?} accelerated={accelerated:?}"
                    );
                }
                eprintln!(
                    "{name} {rounding:?} {RUNS}x{ITERATIONS}: scalar={scalar_total:?} accelerated={accelerated_total:?} scalar/accelerated={:.3}",
                    scalar_total.as_secs_f64() / accelerated_total.as_secs_f64()
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn blit_mask_scalar(
        target: &mut Pixmap,
        coverage: &[u8],
        width: u32,
        height: u32,
        x: i64,
        y: i64,
        color: [u8; 4],
        clip: Option<&Mask>,
        rounding: BlendRounding,
    ) -> bool {
        blit_mask_impl::<false, false>(target, coverage, width, height, x, y, color, clip, rounding)
    }

    #[test]
    fn vector_rows_match_scalar_for_all_coverage_clip_pairs_and_tails() {
        let coverage: Vec<u8> = (0..256u32).flat_map(|a| std::iter::repeat_n(a as u8, 256)).collect();
        let mut clip = Mask::new(259, 256).unwrap();
        for row in clip.data_mut().chunks_exact_mut(259) {
            for (index, alpha) in row.iter_mut().enumerate() {
                *alpha = index as u8;
            }
        }
        let mut source = vec![0; 259 * 256];
        for (dst, src) in source.chunks_exact_mut(259).zip(coverage.chunks_exact(256)) {
            dst[..256].copy_from_slice(src);
            dst[256..].copy_from_slice(&[1, 129, 255]);
        }
        for rounding in [BlendRounding::Combined, BlendRounding::Separate] {
            for color in [[12, 155, 231, 255], [221, 17, 94, 129], [0, 0, 0, 1]] {
                for masked in [false, true] {
                    for x in [-3, 0, 5] {
                        let mut expected = background_sized(259, 256);
                        let mut actual = expected.clone();
                        let mask = masked.then_some(&clip);
                        assert!(blit_mask_scalar(
                            &mut expected,
                            &source,
                            259,
                            256,
                            x,
                            0,
                            color,
                            mask,
                            rounding
                        ));
                        assert!(blit_mask(
                            &mut actual,
                            &source,
                            259,
                            256,
                            x,
                            0,
                            color,
                            mask,
                            rounding
                        ));
                        assert_eq!(
                            actual.data(),
                            expected.data(),
                            "{rounding:?} {color:?} masked={masked} x={x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual repeated scalar/vector glyph blend A/B"]
    fn benchmark_vector_glyph_rows() {
        use std::hint::black_box;
        let coverage: Vec<u8> = (0..48 * 24).map(|n| (n * 73 % 256) as u8).collect();
        let mut clip = Mask::new(64, 40).unwrap();
        for (n, alpha) in clip.data_mut().iter_mut().enumerate() {
            *alpha = [0, 127, 255, 255][n % 4];
        }
        for rounding in [BlendRounding::Combined, BlendRounding::Separate] {
            for masked in [false, true] {
                let mut scalar = Vec::new();
                let mut vector = Vec::new();
                for run in 0..8 {
                    for simd in [run % 2 == 0, run % 2 != 0] {
                        let mut target = background_sized(64, 40);
                        let entry = if simd { blit_mask } else { blit_mask_scalar };
                        let started = std::time::Instant::now();
                        for _ in 0..4096 {
                            assert!(black_box(entry(
                                black_box(&mut target),
                                black_box(&coverage),
                                48,
                                24,
                                3,
                                5,
                                [231, 19, 137, 129],
                                masked.then_some(&clip),
                                rounding
                            )));
                        }
                        black_box(target.data());
                        if simd {
                            vector.push(started.elapsed());
                        } else {
                            scalar.push(started.elapsed());
                        }
                    }
                }
                scalar.sort();
                vector.sort();
                eprintln!(
                    "glyph rows {rounding:?} masked={masked}: scalar={:?} vector={:?} ratio={:.2}x",
                    scalar[4],
                    vector[4],
                    scalar[4].as_secs_f64() / vector[4].as_secs_f64()
                );
            }
        }
    }
}
