//! CPU surface drawing and pixel fast paths shared by GPU backends and replay.

use tiny_skia::{BlendMode, Mask, Pixmap, PixmapPaint, PixmapRef, Transform};

/// Validate tightly packed RGBA storage before passing it to a native API.
pub fn rgba_size_matches(bytes: usize, width: u32, height: u32) -> bool {
    width != 0
        && height != 0
        && (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            == Some(bytes)
}

/// Convert straight RGBA to premultiplied RGBA, preserving floor division by 255.
pub fn premultiply_rgba(rgba: &mut [u8]) {
    #[cfg(target_arch = "aarch64")]
    let offset = unsafe { premultiply_rgba_neon(rgba) };
    #[cfg(not(target_arch = "aarch64"))]
    let offset = 0;
    for pixel in rgba[offset..].chunks_exact_mut(4) {
        let alpha = pixel[3] as u16;
        if alpha != 255 {
            for channel in &mut pixel[..3] {
                *channel = (*channel as u16 * alpha / 255) as u8;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn premultiply_rgba_neon(rgba: &mut [u8]) -> usize {
    use std::arch::aarch64::*;
    unsafe fn channel(value: uint8x16_t, alpha: uint8x16_t) -> uint8x16_t {
        use std::arch::aarch64::*;
        unsafe {
            let divide = |value: uint16x8_t| {
                // Exact for channel * alpha <= 65025; no rounding-policy change.
                let sum = vaddq_u16(vaddq_u16(value, vdupq_n_u16(1)), vshrq_n_u16::<8>(value));
                vmovn_u16(vshrq_n_u16::<8>(sum))
            };
            vcombine_u8(
                divide(vmull_u8(vget_low_u8(value), vget_low_u8(alpha))),
                divide(vmull_u8(vget_high_u8(value), vget_high_u8(alpha))),
            )
        }
    }
    let mut offset = 0;
    while offset + 64 <= rgba.len() {
        let pixels = unsafe { vld4q_u8(rgba.as_ptr().add(offset)) };
        if vminvq_u8(pixels.3) != 255 {
            let output = unsafe {
                uint8x16x4_t(
                    channel(pixels.0, pixels.3),
                    channel(pixels.1, pixels.3),
                    channel(pixels.2, pixels.3),
                    pixels.3,
                )
            };
            unsafe {
                vst4q_u8(rgba.as_mut_ptr().add(offset), output);
            }
        }
        offset += 64;
    }
    offset
}

#[inline]
pub fn rgba_is_opaque(rgba: &[u8]) -> bool {
    rgba_alpha_matches::<255>(rgba)
}

pub fn rgba_is_transparent(rgba: &[u8]) -> bool {
    rgba_alpha_matches::<0>(rgba)
}

fn rgba_alpha_matches<const ALPHA: u8>(rgba: &[u8]) -> bool {
    if rgba.len() % 4 != 0 {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { rgba_alpha_matches_neon::<ALPHA>(rgba) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    rgba.chunks_exact(4).all(|pixel| pixel[3] == ALPHA)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rgba_alpha_matches_neon<const ALPHA: u8>(rgba: &[u8]) -> bool {
    use std::arch::aarch64::*;
    let mut offset = 0;
    while offset + 64 <= rgba.len() {
        let channels = unsafe { vld4q_u8(rgba.as_ptr().add(offset)) };
        if vminvq_u8(vceqq_u8(channels.3, vdupq_n_u8(ALPHA))) != 255 {
            return false;
        }
        offset += 64;
    }
    rgba[offset..]
        .chunks_exact(4)
        .all(|pixel| pixel[3] == ALPHA)
}

/// Copy an opaque RGBA image with an integer translation, clipped to the target.
/// Returns false without modifying the target when the fast path cannot apply.
pub fn blit_opaque_unscaled_image(
    target: &mut Pixmap,
    rgba: &[u8],
    width: u32,
    height: u32,
    transform: Transform,
) -> bool {
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4));
    if width == 0
        || height == 0
        || expected_len != Some(rgba.len())
        || transform.sx != 1.0
        || transform.sy != 1.0
        || transform.kx != 0.0
        || transform.ky != 0.0
        || !transform.tx.is_finite()
        || !transform.ty.is_finite()
        || transform.tx.fract() != 0.0
        || transform.ty.fract() != 0.0
        || !rgba_is_opaque(rgba)
    {
        return false;
    }
    let left = transform.tx as i64;
    let top = transform.ty as i64;
    let x0 = left.max(0);
    let y0 = top.max(0);
    let x1 = left
        .saturating_add(i64::from(width))
        .min(i64::from(target.width()));
    let y1 = top
        .saturating_add(i64::from(height))
        .min(i64::from(target.height()));
    if x0 >= x1 || y0 >= y1 {
        return true;
    }
    let copy_bytes = (x1 - x0) as usize * 4;
    let source_stride = width as usize * 4;
    let target_stride = target.width() as usize * 4;
    let source_x = (x0 - left) as usize * 4;
    let target_x = x0 as usize * 4;
    let target_bytes = target.data_mut();
    for y in y0..y1 {
        let source = (y - top) as usize * source_stride + source_x;
        let destination = y as usize * target_stride + target_x;
        target_bytes[destination..destination + copy_bytes]
            .copy_from_slice(&rgba[source..source + copy_bytes]);
    }
    true
}

/// Draw a premultiplied surface, using row copies for opaque aligned placement.
pub fn draw_surface(
    target: &mut Pixmap,
    source: PixmapRef<'_>,
    x: i32,
    y: i32,
    paint: &PixmapPaint,
    transform: Transform,
    mask: Option<&Mask>,
) {
    // tiny-skia applies the placement before the supplied transform.
    let placement = transform.pre_translate(x as f32, y as f32);
    if mask.is_none()
        && paint.opacity == 1.0
        && paint.blend_mode == BlendMode::SourceOver
        && blit_opaque_unscaled_image(
            target,
            source.data(),
            source.width(),
            source.height(),
            placement,
        )
    {
        return;
    }
    target.draw_pixmap(x, y, source, paint, transform, mask);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn premultiplication_matches_scalar_for_every_channel_alpha_pair_and_tails() {
        let mut rgba: Vec<u8> = (0..=255_u16)
            .flat_map(|alpha| {
                (0..=255_u16).flat_map(move |value| {
                    [value as u8, (255 - value) as u8, value as u8, alpha as u8]
                })
            })
            .collect();
        rgba.extend_from_slice(&[200, 100, 50, 128, 11, 13, 17, 0, 31, 37, 41, 255, 7]);
        let mut expected = rgba.clone();
        for pixel in expected.chunks_exact_mut(4) {
            let alpha = pixel[3] as u16;
            for channel in &mut pixel[..3] {
                *channel = (*channel as u16 * alpha / 255) as u8;
            }
        }
        premultiply_rgba(&mut rgba);
        assert_eq!(rgba, expected);
    }

    #[test]
    fn storage_validation_rejects_overflow_and_incomplete_pixels() {
        assert!(rgba_size_matches(16, 2, 2));
        assert!(!rgba_size_matches(15, 2, 2));
        assert!(!rgba_size_matches(0, 0, 0));
        assert!(!rgba_size_matches(0, u32::MAX, u32::MAX));
    }

    fn opaque_source() -> Pixmap {
        let rgba: Vec<u8> = (0..16)
            .flat_map(|index| [index * 13, index * 7, index * 3, 255])
            .collect();
        PixmapRef::from_bytes(&rgba, 4, 4).unwrap().to_owned()
    }

    fn target() -> Pixmap {
        let mut pixmap = Pixmap::new(8, 8).unwrap();
        pixmap.fill(tiny_skia::Color::from_rgba8(40, 30, 20, 255));
        pixmap
    }

    fn assert_draw_matches(
        source: PixmapRef<'_>,
        x: i32,
        y: i32,
        paint: &PixmapPaint,
        transform: Transform,
        mask: Option<&Mask>,
    ) {
        let mut expected = target();
        let mut actual = expected.clone();
        expected.draw_pixmap(x, y, source, paint, transform, mask);
        draw_surface(&mut actual, source, x, y, paint, transform, mask);
        assert_eq!(
            actual.data(),
            expected.data(),
            "placement ({x}, {y}), {transform:?}"
        );
    }

    #[test]
    fn alpha_checks_cover_vector_tail_and_malformed_lengths() {
        assert!(rgba_is_opaque(&[]));
        assert!(rgba_is_transparent(&[]));
        for count in [1, 15, 16, 17, 31, 32, 33] {
            let mut pixels = vec![255; count * 4];
            assert!(rgba_is_opaque(&pixels));
            assert!(!rgba_is_transparent(&pixels));
            for index in [0, count / 2, count - 1] {
                pixels[index * 4 + 3] = 254;
                assert!(!rgba_is_opaque(&pixels));
                pixels[index * 4 + 3] = 255;
            }
            for pixel in pixels.chunks_exact_mut(4) {
                pixel[3] = 0;
            }
            assert!(rgba_is_transparent(&pixels));
            assert!(!rgba_is_opaque(&pixels));
            for index in [0, count / 2, count - 1] {
                pixels[index * 4 + 3] = 1;
                assert!(!rgba_is_transparent(&pixels));
                pixels[index * 4 + 3] = 0;
            }
        }
        for length in [1, 2, 3, 5, 63, 65, 66, 67] {
            assert!(!rgba_is_opaque(&vec![255; length]));
            assert!(!rgba_is_transparent(&vec![0; length]));
        }
    }

    #[test]
    fn opaque_blit_rejects_malformed_dimensions_without_writes() {
        let mut pixmap = target();
        let before = pixmap.data().to_vec();
        for (length, width, height) in [
            (0, 4, 4),
            (63, 4, 4),
            (65, 4, 4),
            (68, 4, 4),
            (0, 0, 4),
            (0, 4, 0),
            (4, u32::MAX, u32::MAX),
        ] {
            assert!(!blit_opaque_unscaled_image(
                &mut pixmap,
                &vec![255; length],
                width,
                height,
                Transform::identity()
            ));
            assert_eq!(pixmap.data(), before);
        }
    }

    #[test]
    fn opaque_draw_matches_clipping_and_combined_translation() {
        let source = opaque_source();
        for quality in [
            tiny_skia::FilterQuality::Nearest,
            tiny_skia::FilterQuality::Bilinear,
            tiny_skia::FilterQuality::Bicubic,
        ] {
            let paint = PixmapPaint {
                quality,
                ..PixmapPaint::default()
            };
            for (x, y, tx, ty) in [
                (0, 0, 0.0, 0.0),
                (1, 2, 2.0, 1.0),
                (-2, -1, 0.0, 0.0),
                (7, 7, 0.0, 0.0),
                (-8, 0, 0.0, 0.0),
                (0, 0, 9.0, 0.0),
                (0, -8, 0.0, 0.0),
            ] {
                let transform = Transform::from_translate(tx, ty);
                assert_draw_matches(source.as_ref(), x, y, &paint, transform, None);
                let mut actual = target();
                assert!(blit_opaque_unscaled_image(
                    &mut actual,
                    source.data(),
                    4,
                    4,
                    transform.pre_translate(x as f32, y as f32)
                ));
            }
        }
    }

    #[test]
    fn transparent_surface_uses_blending_fallback() {
        for alpha in [0, 128] {
            let mut source = Pixmap::new(4, 4).unwrap();
            source.fill(tiny_skia::Color::from_rgba8(200, 100, 50, alpha));
            let mut pixmap = target();
            assert!(!blit_opaque_unscaled_image(
                &mut pixmap,
                source.data(),
                4,
                4,
                Transform::identity()
            ));
            assert_draw_matches(
                source.as_ref(),
                -1,
                2,
                &PixmapPaint::default(),
                Transform::identity(),
                None,
            );
        }
    }

    #[test]
    fn scaled_fractional_and_rotated_surfaces_match_fallback() {
        let source = opaque_source();
        for transform in [
            Transform::from_scale(1.5, 0.5),
            Transform::from_translate(0.5, 0.25),
            Transform::from_rotate(20.0),
        ] {
            let mut pixmap = target();
            assert!(!blit_opaque_unscaled_image(
                &mut pixmap,
                source.data(),
                4,
                4,
                transform
            ));
            let paint = PixmapPaint {
                quality: tiny_skia::FilterQuality::Bilinear,
                ..PixmapPaint::default()
            };
            assert_draw_matches(source.as_ref(), 1, 2, &paint, transform, None);
        }
    }

    #[test]
    fn opacity_blend_mode_and_mask_match_fallback() {
        let source = opaque_source();
        let paint = PixmapPaint {
            opacity: 0.5,
            ..PixmapPaint::default()
        };
        assert_draw_matches(source.as_ref(), 1, 1, &paint, Transform::identity(), None);
        let paint = PixmapPaint {
            blend_mode: BlendMode::DestinationOut,
            ..PixmapPaint::default()
        };
        assert_draw_matches(source.as_ref(), 1, 1, &paint, Transform::identity(), None);
        let mut mask = Mask::new(8, 8).unwrap();
        for (index, coverage) in mask.data_mut().iter_mut().enumerate() {
            *coverage = (index * 4) as u8;
        }
        assert_draw_matches(
            source.as_ref(),
            1,
            1,
            &PixmapPaint::default(),
            Transform::identity(),
            Some(&mask),
        );
    }
}
