//! Bounded CPU solid fills for CSS rectangles and axis-aligned border strips.
//! Callers retain tiny-skia for unsupported geometry or paint settings.

use tiny_skia::{Color, Mask, Pixmap, Transform};

/// Fill `[x, y, width, height]` using premultiplied SourceOver.
///
/// Matches tiny-skia's default low-precision solid-color pipeline
/// (`ColorSpace::Linear`, SourceOver, no forced high-quality pipeline).
/// Fractional coverage is supplied
/// by a device-space mask with exactly the target dimensions. Integer device
/// edges have identical coverage with or without rectangle antialiasing.
///
/// Only finite, positive axis scales and exactly integral transformed edges are
/// supported. Invalid rectangles, mismatched masks, rotation, shear, reflection,
/// and fractional device edges return false without changing any target bytes.
/// Valid transparent and offscreen fills return true. This API does not select
/// a fallback or modify any caller's rendering behavior.
pub fn fill_solid_aligned_rect(
    target: &mut Pixmap,
    rect: [f32; 4],
    color: Color,
    transform: Transform,
    mask: Option<&Mask>,
) -> bool {
    let [x, y, width, height] = rect;
    if !rect.iter().all(|v| v.is_finite())
        || width <= 0.0
        || height <= 0.0
        || transform.kx != 0.0
        || transform.ky != 0.0
        || !transform.sx.is_finite()
        || !transform.sy.is_finite()
        || transform.sx <= 0.0
        || transform.sy <= 0.0
        || !transform.tx.is_finite()
        || !transform.ty.is_finite()
    {
        return false;
    }
    let edges = [
        x * transform.sx + transform.tx,
        y * transform.sy + transform.ty,
        (x + width) * transform.sx + transform.tx,
        (y + height) * transform.sy + transform.ty,
    ];
    if !edges.iter().all(|v| v.is_finite() && v.fract() == 0.0)
        || edges[0] >= edges[2]
        || edges[1] >= edges[3]
    {
        return false;
    }
    let tw = target.width() as usize;
    let th = target.height() as usize;
    let Some(len) = tw.checked_mul(th) else {
        return false;
    };
    if len.checked_mul(4) != Some(target.data().len())
        || mask.is_some_and(|m| {
            m.width() != target.width() || m.height() != target.height() || m.data().len() != len
        })
    {
        return false;
    }
    let [left, top, right, bottom] = edges;
    let x0 = (left.max(0.0) as usize).min(tw);
    let y0 = (top.max(0.0) as usize).min(th);
    let x1 = (right.max(0.0) as usize).min(tw);
    let y1 = (bottom.max(0.0) as usize).min(th);
    let source = color.premultiply().to_color_u8();
    let source = [source.red(), source.green(), source.blue(), source.alpha()];
    if x0 >= x1 || y0 >= y1 || source[3] == 0 {
        return true;
    }
    let pixels = target.data_mut();
    let stride = tw * 4;
    let start = y0 * stride + x0 * 4;
    let bytes = (x1 - x0) * 4;
    if mask.is_none() && source[3] == 255 {
        fill_run(&mut pixels[start..start + bytes], source);
        for y in y0 + 1..y1 {
            pixels.copy_within(start..start + bytes, y * stride + x0 * 4);
        }
        return true;
    }
    for y in y0..y1 {
        let row_start = y * tw + x0;
        let row = &mut pixels[row_start * 4..(row_start + x1 - x0) * 4];
        if let Some(mask) = mask {
            let coverage = &mask.data()[row_start..row_start + x1 - x0];
            let mut x = 0;
            while x < coverage.len() {
                let value = coverage[x];
                let begin = x;
                x += 1;
                while x < coverage.len() && coverage[x] == value {
                    x += 1;
                }
                if value != 0 {
                    let covered = source.map(|c| scale(c, value));
                    blend_run(&mut row[begin * 4..x * 4], covered);
                }
            }
        } else {
            blend_run(row, source);
        }
    }
    true
}

// tiny-skia 0.12 lowp uses (product + 255) >> 8, not floor/round / 255.
// Text and video kernels have different contracts; do not reuse their math.
#[inline]
fn scale(channel: u8, alpha: u8) -> u8 {
    ((u16::from(channel) * u16::from(alpha) + 255) >> 8) as u8
}

fn blend_run(pixels: &mut [u8], source: [u8; 4]) {
    if source[3] == 255 {
        fill_run(pixels, source);
    } else if source[3] != 0 {
        #[cfg(target_arch = "aarch64")]
        if pixels.len() >= 256 {
            // Short spans stay scalar; contiguous NEON wins on longer rows.
            unsafe {
                blend_run_neon(pixels, source);
            }
            return;
        }
        blend_run_scalar(pixels, source);
    }
}

#[cfg(test)]
fn blend_run_bulk(pixels: &mut [u8], source: [u8; 4]) {
    #[cfg(target_arch = "aarch64")]
    if pixels.len() >= 16 {
        // NEON supports unaligned loads; each block stays inside the row.
        unsafe {
            blend_run_neon(pixels, source);
        }
        return;
    }
    blend_run_scalar(pixels, source);
}

fn blend_run_scalar(pixels: &mut [u8], source: [u8; 4]) {
    let inverse = 255 - source[3];
    for pixel in pixels.chunks_exact_mut(4) {
        for (destination, source) in pixel.iter_mut().zip(source) {
            *destination = source + scale(*destination, inverse);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn blend_run_neon(pixels: &mut [u8], source: [u8; 4]) {
    use std::arch::aarch64::*;
    unsafe {
        let inverse = vdup_n_u8(255 - source[3]);
        let bias = vdupq_n_u16(255);
        let pattern = [
            source[0], source[1], source[2], source[3], source[0], source[1], source[2], source[3],
            source[0], source[1], source[2], source[3], source[0], source[1], source[2], source[3],
        ];
        let source = vld1q_u8(pattern.as_ptr());
        let blend = |destination: uint8x16_t| {
            let low =
                vshrn_n_u16::<8>(vaddq_u16(vmull_u8(vget_low_u8(destination), inverse), bias));
            let high = vshrn_n_u16::<8>(vaddq_u16(
                vmull_u8(vget_high_u8(destination), inverse),
                bias,
            ));
            vaddq_u8(vcombine_u8(low, high), source)
        };
        let mut offset = 0;
        while offset + 16 <= pixels.len() {
            let result = blend(vld1q_u8(pixels.as_ptr().add(offset)));
            vst1q_u8(pixels.as_mut_ptr().add(offset), result);
            offset += 16;
        }
        blend_run_scalar(
            &mut pixels[offset..],
            [pattern[0], pattern[1], pattern[2], pattern[3]],
        );
    }
}

fn fill_run(pixels: &mut [u8], source: [u8; 4]) {
    if pixels.is_empty() {
        return;
    }
    pixels[..4].copy_from_slice(&source);
    let mut initialized = 4;
    while initialized < pixels.len() {
        let count = initialized.min(pixels.len() - initialized);
        pixels.copy_within(..count, initialized);
        initialized += count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tiny_skia::{Paint, PremultipliedColorU8, Rect};

    #[test]
    fn solid_blend_bulk_matches_scalar_for_all_alphas_and_tails() {
        for alpha in 1..255 {
            let source = [alpha / 2, alpha / 3, alpha, alpha];
            for channel in 0..=255 {
                for count in [1, 7, 8, 9, 15, 16, 17, 33, 63, 64, 65, 80] {
                    let mut actual = vec![0; count * 4 + 3];
                    for pixel in actual[3..].chunks_exact_mut(4) {
                        pixel.copy_from_slice(&[channel, channel / 2, 255 - channel, 255]);
                    }
                    let mut expected = actual.clone();
                    let mut production = actual.clone();
                    blend_run_bulk(&mut actual[3..], source);
                    blend_run(&mut production[3..], source);
                    blend_run_scalar(&mut expected[3..], source);
                    assert_eq!(
                        actual, expected,
                        "alpha={alpha} channel={channel} count={count}"
                    );
                    assert_eq!(
                        production, expected,
                        "production alpha={alpha} channel={channel} count={count}"
                    );
                }
            }
        }
    }

    #[test]
    fn wide_translucent_fills_and_mask_runs_match_tiny_skia() {
        for coverage in [0, 1, 127, 254, 255] {
            for alpha in [1, 32, 128, 254, 255] {
                let mut actual = Pixmap::new(257, 3).unwrap();
                for (i, pixel) in actual.pixels_mut().iter_mut().enumerate() {
                    let a = (i * 53 % 256) as u8;
                    *pixel = PremultipliedColorU8::from_rgba(a / 2, a / 3, a / 4, a).unwrap();
                }
                let mut expected = actual.clone();
                let mut mask = Mask::new(257, 3).unwrap();
                mask.data_mut().fill(coverage);
                let color = Color::from_rgba8(31, 129, 241, alpha);
                let mut paint = Paint::default();
                paint.set_color(color);
                expected.fill_rect(
                    Rect::from_xywh(0.0, 0.0, 257.0, 3.0).unwrap(),
                    &paint,
                    Transform::identity(),
                    Some(&mask),
                );
                assert!(fill_solid_aligned_rect(
                    &mut actual,
                    [0.0, 0.0, 257.0, 3.0],
                    color,
                    Transform::identity(),
                    Some(&mask)
                ));
                assert!(
                    actual.data() == expected.data(),
                    "wide fill coverage={coverage} alpha={alpha}"
                );
            }
        }
    }

    #[test]
    #[ignore = "component benchmark; run alone in release mode"]
    fn solid_blend_bulk_alternating_benchmark() {
        let source = [64, 32, 16, 128];
        for count in [64, 4096, 262144] {
            let mut timings = [Vec::new(), Vec::new()];
            for round in 0..8 {
                for bulk in if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let mut pixels = vec![200; count * 4];
                    let start = std::time::Instant::now();
                    for _ in 0..32 {
                        let pixels = std::hint::black_box(pixels.as_mut_slice());
                        if bulk {
                            blend_run_bulk(pixels, source);
                        } else {
                            blend_run_scalar(pixels, source);
                        }
                    }
                    std::hint::black_box(pixels);
                    timings[usize::from(bulk)].push(start.elapsed().as_nanos());
                }
            }
            for samples in &mut timings {
                samples.sort_unstable();
            }
            let median = |samples: &[u128]| (samples[3] + samples[4]) as f64 / 2.0;
            eprintln!(
                "solid blend pixels={count} scalar_ns={} bulk_ns={} speedup={:.3}",
                median(&timings[0]),
                median(&timings[1]),
                median(&timings[0]) / median(&timings[1])
            );
        }
    }

    fn background() -> Pixmap {
        let mut p = Pixmap::new(33, 7).unwrap();
        for (i, pixel) in p.pixels_mut().iter_mut().enumerate() {
            let a = (i * 53 % 256) as u8;
            *pixel = PremultipliedColorU8::from_rgba(a / 2, a / 3, a / 4, a).unwrap();
        }
        p
    }

    fn compare(rect: [f32; 4], color: Color, transform: Transform, mask: Option<&Mask>) {
        let mut actual = background();
        let mut expected = actual.clone();
        let mut paint = Paint::default();
        paint.set_color(color);
        expected.fill_rect(
            Rect::from_xywh(rect[0], rect[1], rect[2], rect[3]).unwrap(),
            &paint,
            transform,
            mask,
        );
        assert!(fill_solid_aligned_rect(
            &mut actual,
            rect,
            color,
            transform,
            mask
        ));
        assert_eq!(
            actual.data(),
            expected.data(),
            "rect={rect:?} color={color:?} transform={transform:?}"
        );
    }

    #[test]
    fn integer_transforms_clipping_and_row_tails_match_tiny_skia() {
        for transform in [
            Transform::identity(),
            Transform::from_translate(-3.0, 2.0),
            Transform::from_row(2.0, 0.0, 0.0, 3.0, -4.0, -2.0),
        ] {
            for rect in [
                [0.0, 0.0, 33.0, 7.0],
                [-3.0, -2.0, 13.0, 8.0],
                [2.0, 1.0, 1.0, 1.0],
                [100.0, 100.0, 4.0, 4.0],
            ] {
                compare(rect, Color::from_rgba8(31, 129, 241, 255), transform, None);
            }
        }
        compare(
            [0.5, 0.5, 8.0, 2.0],
            Color::WHITE,
            Transform::from_scale(2.0, 2.0),
            None,
        );
    }

    #[test]
    fn source_over_all_alphas_and_fractional_masks_match_tiny_skia() {
        let mut mask = Mask::new(33, 7).unwrap();
        for (i, c) in mask.data_mut().iter_mut().enumerate() {
            *c = (i % 256) as u8;
        }
        for a in 0..=255 {
            let color = Color::from_rgba8(19, 127, 243, a);
            compare([-2.0, 0.0, 35.0, 7.0], color, Transform::identity(), None);
            compare(
                [-2.0, 0.0, 35.0, 7.0],
                color,
                Transform::identity(),
                Some(&mask),
            );
        }
    }

    #[test]
    fn binary_and_uniform_mask_runs_match_tiny_skia() {
        let mut mask = Mask::new(33, 7).unwrap();
        for value in [0, 1, 127, 254, 255] {
            mask.data_mut().fill(value);
            for a in [0, 1, 128, 255] {
                compare(
                    [1.0, 1.0, 31.0, 5.0],
                    Color::from_rgba8(240, 80, 12, a),
                    Transform::identity(),
                    Some(&mask),
                );
            }
        }
        for (i, c) in mask.data_mut().iter_mut().enumerate() {
            *c = if i / 5 % 2 == 0 { 0 } else { 255 };
        }
        compare(
            [0.0, 0.0, 33.0, 7.0],
            Color::WHITE,
            Transform::identity(),
            Some(&mask),
        );
    }

    #[test]
    fn floating_colors_and_transformed_masks_match_tiny_skia() {
        let mut mask = Mask::new(33, 7).unwrap();
        for (i, value) in mask.data_mut().iter_mut().enumerate() {
            *value = (i * 137 % 256) as u8;
        }
        for color in [
            Color::from_rgba(0.123, 0.456, 0.789, 0.333).unwrap(),
            Color::from_rgba(0.5, 0.1, 0.9, 0.001).unwrap(),
        ] {
            compare(
                [-1.0, -1.0, 9.0, 4.0],
                color,
                Transform::from_row(2.0, 0.0, 0.0, 2.0, 3.0, 1.0),
                Some(&mask),
            );
        }
    }

    #[test]
    fn transparent_and_offscreen_are_successful_no_ops() {
        let mut p = background();
        let before = p.clone();
        assert!(fill_solid_aligned_rect(
            &mut p,
            [0.0, 0.0, 33.0, 7.0],
            Color::TRANSPARENT,
            Transform::identity(),
            None
        ));
        assert!(fill_solid_aligned_rect(
            &mut p,
            [1.0e20, 1.0e20, 1.0e20, 1.0e20],
            Color::WHITE,
            Transform::identity(),
            None
        ));
        assert_eq!(p.data(), before.data());
    }

    #[test]
    fn unsupported_and_malformed_inputs_never_mutate() {
        let mut p = background();
        let before = p.clone();
        for rect in [
            [0.0, 0.0, -1.0, 2.0],
            [0.0, 0.0, 0.0, 2.0],
            [f32::NAN, 0.0, 2.0, 2.0],
            [0.0, 0.0, f32::INFINITY, 2.0],
            [0.25, 0.0, 2.0, 2.0],
            [f32::MAX, 0.0, f32::MAX, 2.0],
        ] {
            assert!(!fill_solid_aligned_rect(
                &mut p,
                rect,
                Color::WHITE,
                Transform::identity(),
                None
            ));
            assert_eq!(p.data(), before.data());
        }
        for transform in [
            Transform::from_rotate(30.0),
            Transform::from_scale(-1.0, 1.0),
            Transform::from_scale(0.0, 1.0),
            Transform::from_translate(0.5, 0.0),
            Transform::from_translate(f32::NAN, 0.0),
            Transform::from_scale(f32::INFINITY, 1.0),
        ] {
            assert!(!fill_solid_aligned_rect(
                &mut p,
                [0.0, 0.0, 2.0, 2.0],
                Color::WHITE,
                transform,
                None
            ));
            assert_eq!(p.data(), before.data());
        }
        let wrong_mask = Mask::new(7, 33).unwrap();
        for color in [Color::WHITE, Color::TRANSPARENT] {
            assert!(!fill_solid_aligned_rect(
                &mut p,
                [100.0, 100.0, 2.0, 2.0],
                color,
                Transform::identity(),
                Some(&wrong_mask)
            ));
            assert_eq!(p.data(), before.data());
        }
    }

    #[test]
    fn source_over_rounding_and_premultiplied_invariant() {
        assert_eq!(scale(128, 128), 64);
        assert_eq!(scale(255, 254), 254);
        let mut pixel = [40, 30, 20, 100];
        blend_run(&mut pixel, [64, 32, 16, 128]);
        assert_eq!(pixel, [84, 47, 26, 178]);
        for a in 0..=255 {
            for coverage in 0..=255 {
                let source = [a, a / 2, a / 3, a].map(|c| scale(c, coverage));
                let mut pixel = [200, 150, 100, 220];
                blend_run(&mut pixel, source);
                assert!(pixel[..3].iter().all(|c| *c <= pixel[3]));
            }
        }
    }
}
