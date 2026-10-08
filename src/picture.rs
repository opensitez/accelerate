use crate::RgbaSurface;
use tiny_skia::{Mask, PathBuilder, Pixmap, PixmapPaint, Transform};

pub const MAX_PICTURES: usize = 64;
pub const MAX_PICTURE_BYTES: usize = 64 * 1024 * 1024;
const ENABLE_ROI_CLIP_COMPOSITION: bool = false;

/// An immutable painted picture, composed after the framebuffer/video layers.
/// Slice order is painter order. This does not rasterize fonts or other paint.
#[derive(Clone, Debug)]
pub struct PictureLayer {
    pub surface: RgbaSurface,
    /// Source physical pixels to logical top-left window coordinates [a,b,c,d,e,f].
    pub transform: [f32; 6],
    pub opacity: f32,
    /// Destination-space logical rectangle [x,y,width,height], independent of transform.
    /// Physical clip edges must be integer aligned; unsupported clips return false.
    pub clip: Option<[f32; 4]>,
}

impl PictureLayer {
    pub fn is_valid(&self) -> bool {
        let [a, b, c, d, e, f] = self.transform;
        let determinant = a * d - b * c;
        self.transform.iter().all(|value| value.is_finite())
            && determinant.is_finite()
            && determinant != 0.0
            && self.opacity.is_finite()
            && (0.0..=1.0).contains(&self.opacity)
            && self.clip.is_none_or(|rect| {
                rect.iter().all(|value| value.is_finite())
                    && rect[2] >= 0.0
                    && rect[3] >= 0.0
                    && (rect[0] + rect[2]).is_finite()
                    && (rect[1] + rect[3]).is_finite()
                    && (rect[2] == 0.0 || rect[0] + rect[2] > rect[0])
                    && (rect[3] == 0.0 || rect[1] + rect[3] > rect[1])
            })
            && {
                let (w, h) = self.surface.dimensions();
                [
                    (0.0, 0.0),
                    (w as f32, 0.0),
                    (0.0, h as f32),
                    (w as f32, h as f32),
                ]
                .iter()
                .all(|&(x, y)| (a * x + c * y + e).is_finite() && (b * x + d * y + f).is_finite())
            }
    }

    pub(crate) fn valid_at_scale(&self, scale: f32) -> bool {
        self.is_valid()
            && scale.is_finite()
            && scale > 0.0
            && self
                .transform
                .iter()
                .all(|value| (*value * scale).is_finite())
            && {
                let [a, b, c, d, e, f] = self.transform;
                let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|value| value * scale);
                let determinant = a * d - b * c;
                let (w, h) = self.surface.dimensions();
                [
                    (0., 0.),
                    (w as f32, 0.),
                    (0., h as f32),
                    (w as f32, h as f32),
                ]
                .iter()
                .all(|&(x, y)| {
                    determinant.is_finite()
                        && determinant != 0.0
                        && (a * x + c * y + e).is_finite()
                        && (b * x + d * y + f).is_finite()
                })
            }
            && self.clip.is_none_or(|[x, y, w, h]| {
                [x, y, w, h, x + w, y + h].iter().all(|value| {
                    let value = *value * scale;
                    value.is_finite() && (value - value.round()).abs() <= 0.0001
                })
            })
    }
}

pub(crate) fn valid_picture_storage(pictures: &[PictureLayer]) -> bool {
    if pictures.len() > MAX_PICTURES {
        return false;
    }
    let mut bytes = 0usize;
    pictures.iter().all(|picture| {
        bytes = bytes.saturating_add(picture.surface.pixels().len());
        bytes <= MAX_PICTURE_BYTES && picture.is_valid()
    })
}

pub(crate) fn valid_pictures(pictures: &[PictureLayer], scale: f32) -> bool {
    valid_picture_storage(pictures)
        && scale.is_finite()
        && scale > 0.0
        && pictures.iter().all(|picture| picture.valid_at_scale(scale))
}

/// Software composition of the same ordered pictures; invalid batches do not write.
/// Embedders may instead keep their original direct paint path when native presentation fails.
pub fn draw_pictures(target: &mut Pixmap, pictures: &[PictureLayer], scale: f32) -> bool {
    if ENABLE_ROI_CLIP_COMPOSITION {
        draw_pictures_roi(target, pictures, scale)
    } else {
        draw_pictures_impl::<true>(target, pictures, scale)
    }
}

// Keep at most one idle viewport allocation; composition never holds this lock.
static PICTURE_CLIP_SCRATCH: std::sync::Mutex<Option<Pixmap>> = std::sync::Mutex::new(None);

struct ClipScratch {
    pixmap: Option<Pixmap>,
}

impl Drop for ClipScratch {
    fn drop(&mut self) {
        if let Ok(mut cached) = PICTURE_CLIP_SCRATCH.lock() {
            if cached.is_none() {
                *cached = self.pixmap.take();
            }
        }
    }
}

fn acquire_clip_scratch(width: u32, height: u32, byte_limit: usize) -> Option<ClipScratch> {
    let size = tiny_skia::IntSize::from_wh(width, height)?;
    let len = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)?;
    let byte_limit = byte_limit.min(MAX_PICTURE_BYTES);
    if len > byte_limit {
        return None;
    }
    let mut data = PICTURE_CLIP_SCRATCH
        .lock()
        .ok()
        .and_then(|mut cached| cached.take())
        .map(Pixmap::take)
        .unwrap_or_default();
    if data.capacity() > byte_limit {
        data = Vec::new();
    }
    if data.len() < len {
        data.try_reserve_exact(len - data.len()).ok()?;
    }
    if data.capacity() > byte_limit {
        return None;
    }
    data.resize(len, 0);
    Some(ClipScratch {
        pixmap: Some(Pixmap::from_vec(data, size)?),
    })
}

enum RoiPicture {
    Skip,
    Direct(Transform),
    Clipped { bounds: [u32; 4], transform: Transform },
}

fn prepare_roi_clips(
    target: &Pixmap,
    pictures: &[PictureLayer],
    scale: f32,
    byte_limit: usize,
) -> Option<(Vec<RoiPicture>, Option<ClipScratch>)> {
    let mut plans = Vec::new();
    plans.try_reserve_exact(pictures.len()).ok()?;
    let mut needs_scratch = false;
    for picture in pictures {
        if picture.opacity == 0.0
            || picture.clip.is_some_and(|clip| clip[2] == 0.0 || clip[3] == 0.0)
        {
            plans.push(RoiPicture::Skip);
            continue;
        }
        let [a, b, c, d, e, f] = picture.transform.map(|value| value * scale);
        let Some([x, y, w, h]) = picture.clip else {
            plans.push(RoiPicture::Direct(Transform::from_row(a, b, c, d, e, f)));
            continue;
        };
        let clip = tiny_skia::Rect::from_xywh(x * scale, y * scale, w * scale, h * scale)?;
        let left = (clip.left().round().clamp(0.0, target.width() as f32) as u32).min(target.width());
        let top = (clip.top().round().clamp(0.0, target.height() as f32) as u32).min(target.height());
        let right = (clip.right().round().clamp(0.0, target.width() as f32) as u32).min(target.width());
        let bottom = (clip.bottom().round().clamp(0.0, target.height() as f32) as u32).min(target.height());
        if right <= left || bottom <= top {
            plans.push(RoiPicture::Skip);
            continue;
        }
        if left == 0 && top == 0 && right == target.width() && bottom == target.height() {
            plans.push(RoiPicture::Direct(Transform::from_row(a, b, c, d, e, f)));
            continue;
        }
        let bounds = [left, top, right - left, bottom - top];
        needs_scratch = true;
        plans.push(RoiPicture::Clipped {
            bounds,
            transform: Transform::from_row(a, b, c, d, e, f),
        });
    }
    let scratch = if needs_scratch {
        Some(acquire_clip_scratch(target.width(), target.height(), byte_limit)?)
    } else {
        None
    };
    Some((plans, scratch))
}

fn draw_pictures_roi(target: &mut Pixmap, pictures: &[PictureLayer], scale: f32) -> bool {
    if target.data().len() > MAX_PICTURE_BYTES {
        return draw_pictures_impl::<true>(target, pictures, scale);
    }
    if draw_pictures_roi_with_limit(target, pictures, scale, MAX_PICTURE_BYTES) {
        true
    } else {
        draw_pictures_impl::<true>(target, pictures, scale)
    }
}

fn draw_pictures_roi_with_limit(
    target: &mut Pixmap,
    pictures: &[PictureLayer],
    scale: f32,
    byte_limit: usize,
) -> bool {
    if !valid_pictures(pictures, scale) {
        return false;
    }
    // No destination writes occur until every plan and scratch allocation succeeds.
    let Some((plans, mut scratch_lease)) = prepare_roi_clips(target, pictures, scale, byte_limit) else {
        return false;
    };
    let mut scratch = scratch_lease.as_mut().and_then(|lease| lease.pixmap.as_mut());
    let target_stride = target.width() as usize * 4;
    for (picture, plan) in pictures.iter().zip(plans) {
        let paint = PixmapPaint {
            opacity: picture.opacity,
            quality: tiny_skia::FilterQuality::Bilinear,
            ..PixmapPaint::default()
        };
        match plan {
            RoiPicture::Skip => {}
            RoiPicture::Direct(transform) => {
                crate::draw_surface(target, picture.surface.as_pixmap(), 0, 0, &paint, transform, None);
            }
            RoiPicture::Clipped { bounds: [x, y, w, h], transform } => {
                let scratch = scratch.as_deref_mut().expect("clipped plans preallocate scratch");
                let row_bytes = w as usize * 4;
                // Painter order requires the CURRENT destination, not a batch-start snapshot.
                for row in 0..h as usize {
                    let destination = (y as usize + row) * target_stride + x as usize * 4;
                    scratch.data_mut()[destination..destination + row_bytes]
                        .copy_from_slice(&target.data()[destination..destination + row_bytes]);
                }
                // Keep original viewport dimensions and matrix: translating a
                // small ROI changes f32 shader inversion and bilinear samples.
                crate::draw_surface(
                    scratch, picture.surface.as_pixmap(), 0, 0, &paint, transform, None,
                );
                for row in 0..h as usize {
                    let destination = (y as usize + row) * target_stride + x as usize * 4;
                    target.data_mut()[destination..destination + row_bytes]
                        .copy_from_slice(&scratch.data()[destination..destination + row_bytes]);
                }
            }
        }
    }
    true
}

fn draw_pictures_impl<const SKIP_FULL_CLIP: bool>(
    target: &mut Pixmap,
    pictures: &[PictureLayer],
    scale: f32,
) -> bool {
    if !valid_pictures(pictures, scale) {
        return false;
    }
    // Finish every fallible clip allocation before the first destination write.
    let mut masks = Vec::with_capacity(pictures.len());
    let mut shared_masks = std::collections::HashMap::new();
    let mut mask_bytes = 0usize;
    for picture in pictures {
        if picture.opacity == 0.0
            || picture
                .clip
                .is_some_and(|rect| rect[2] == 0.0 || rect[3] == 0.0)
        {
            masks.push(None);
            continue;
        }
        let mask = if let Some([x, y, w, h]) = picture.clip {
            let key = [x, y, w, h].map(f32::to_bits);
            if let Some(mask) = shared_masks.get(&key) {
                masks.push(Some(std::sync::Arc::clone(mask)));
                continue;
            }
            let Some(rect) = tiny_skia::Rect::from_xywh(x * scale, y * scale, w * scale, h * scale)
            else {
                return false;
            };
            // Target bounds already supply this clip. Validate the physical
            // rectangle first, and retain masks for every partially covering clip.
            if SKIP_FULL_CLIP
                && rect.left() <= 0.0
                && rect.top() <= 0.0
                && rect.right() >= target.width() as f32
                && rect.bottom() >= target.height() as f32
            {
                masks.push(None);
                continue;
            }
            mask_bytes =
                mask_bytes.saturating_add(target.width() as usize * target.height() as usize);
            if mask_bytes > MAX_PICTURE_BYTES {
                return false;
            }
            let Some(mut mask) = Mask::new(target.width(), target.height()) else {
                return false;
            };
            mask.fill_path(
                &PathBuilder::from_rect(rect),
                tiny_skia::FillRule::Winding,
                false,
                Transform::identity(),
            );
            let mask = std::sync::Arc::new(mask);
            shared_masks.insert(key, mask.clone());
            Some(mask)
        } else {
            None
        };
        masks.push(mask);
    }
    for (picture, mask) in pictures.iter().zip(&masks) {
        if picture.opacity == 0.0
            || picture
                .clip
                .is_some_and(|rect| rect[2] == 0.0 || rect[3] == 0.0)
        {
            continue;
        }
        let [a, b, c, d, e, f] = picture.transform.map(|value| value * scale);
        let paint = PixmapPaint {
            opacity: picture.opacity,
            quality: tiny_skia::FilterQuality::Bilinear,
            ..PixmapPaint::default()
        };
        crate::draw_surface(
            target,
            picture.surface.as_pixmap(),
            0,
            0,
            &paint,
            Transform::from_row(a, b, c, d, e, f),
            mask.as_deref(),
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn assert_same_pixels(actual: &Pixmap, expected: &Pixmap, context: &str) {
        if let Some(pixel) = actual.data().chunks_exact(4).zip(expected.data().chunks_exact(4))
            .position(|(actual, expected)| actual != expected)
        {
            let at = pixel * 4;
            panic!("{context}: first differing pixel ({}, {}): {:?} != {:?}",
                pixel % actual.width() as usize, pixel / actual.width() as usize,
                &actual.data()[at..at + 4], &expected.data()[at..at + 4]);
        }
    }

    #[test]
    fn roi_clips_match_mask_reference_across_affine_opacity_overlap_and_dpr() {
        let mut rgba = Vec::new();
        for y in 0..5 {
            for x in 0..7 {
                let alpha = ((x * 37 + y * 43) % 256) as u8;
                rgba.extend_from_slice(&[alpha / 2, alpha / 3, alpha, alpha]);
            }
        }
        let surface = RgbaSurface::new(Arc::new(rgba), 7, 5).unwrap();
        for scale in [1.0, 2.0] {
            for matrix in [
                [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, -1.0, 0.0],
                [0.8, 0.6, -0.6, 0.8], [0.65, 0.2, 0.15, 1.25],
                [-1.0, 0.0, 0.0, 0.75],
            ] {
                for opacity in [0.0, 0.3, 0.7, 1.0] {
                    for clip in [
                        None, Some([0.0, 0.0, 18.0, 14.0]),
                        Some([3.0, 2.0, 8.0, 7.0]), Some([-2.0, -1.0, 9.0, 8.0]),
                        Some([16.0, 12.0, 6.0, 5.0]), Some([30.0, 25.0, 4.0, 4.0]),
                        Some([3.0, 2.0, 0.0, 7.0]),
                    ] {
                        let [a, b, c, d] = matrix;
                        let first = PictureLayer {
                            surface: surface.clone(), transform: [a, b, c, d, 7.125, 5.375],
                            opacity, clip,
                        };
                        let mut overlap = first.clone();
                        overlap.transform[4] -= 0.25;
                        overlap.transform[5] += 0.5;
                        overlap.opacity = 0.6;
                        overlap.clip = Some([2.0, 1.0, 10.0, 9.0]);
                        let mut reused = overlap.clone();
                        reused.transform[4] += 1.5;
                        reused.opacity = 0.4;
                        let mut direct = layer([0, 96, 0, 96]);
                        direct.transform = [2.0, 0.0, 0.0, 2.0, 4.0, 3.0];
                        let pictures = [first, overlap, direct, reused];
                        for background in [tiny_skia::Color::TRANSPARENT,
                            tiny_skia::Color::from_rgba8(20, 40, 60, 160)]
                        {
                            let mut actual = Pixmap::new((18.0 * scale) as u32, (14.0 * scale) as u32).unwrap();
                            actual.fill(background);
                            let mut expected = actual.clone();
                            assert!(draw_pictures_roi(&mut actual, &pictures, scale));
                            assert!(draw_pictures_impl::<false>(&mut expected, &pictures, scale));
                            assert_same_pixels(&actual, &expected,
                                &format!("DPR={scale}, matrix={matrix:?}, opacity={opacity}, clip={clip:?}"));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn roi_clip_reuse_reads_current_backdrop_and_keeps_outside_pixels_unchanged() {
        let mut first = layer([128, 0, 0, 128]);
        first.transform = [3.0, 0.0, 0.0, 3.0, 2.0, 2.0];
        first.clip = Some([3.0, 3.0, 4.0, 4.0]);
        let mut second = first.clone();
        second.surface = layer([0, 0, 96, 96]).surface;
        let mut intervening = layer([0, 80, 0, 80]);
        intervening.transform = first.transform;
        let pictures = [first, intervening, second];
        let mut actual = Pixmap::new(10, 10).unwrap();
        actual.fill(tiny_skia::Color::from_rgba8(30, 60, 90, 120));
        let original = actual.clone();
        assert!(draw_pictures_roi(&mut actual, &pictures[..1], 1.0));
        for y in 0..10 {
            for x in 0..10 {
                if !(3..7).contains(&x) || !(3..7).contains(&y) {
                    let at = (y * 10 + x) * 4;
                    assert_eq!(&actual.data()[at..at + 4], &original.data()[at..at + 4],
                        "outside ROI ({x}, {y})");
                }
            }
        }
        actual = original;
        let mut expected = actual.clone();
        let (plans, scratch) = prepare_roi_clips(&actual, &pictures, 1.0, MAX_PICTURE_BYTES).unwrap();
        assert_eq!(plans.iter().filter(|plan| matches!(plan, RoiPicture::Clipped { .. })).count(), 2);
        assert_eq!(scratch.as_ref().unwrap().pixmap.as_ref().unwrap().data().len(), 10 * 10 * 4);
        drop(scratch);
        assert!(draw_pictures_roi(&mut actual, &pictures, 1.0));
        assert!(draw_pictures_impl::<false>(&mut expected, &pictures, 1.0));
        assert_same_pixels(&actual, &expected, "shared ROI current backdrop");
    }

    #[test]
    fn roi_clip_preflight_failure_is_atomic_and_bounds_memory() {
        let direct = layer([255, 0, 0, 255]);
        let mut clipped = direct.clone();
        clipped.clip = Some([1.0, 1.0, 4.0, 4.0]);
        let mut target = Pixmap::new(8, 8).unwrap();
        target.fill(tiny_skia::Color::from_rgba8(10, 20, 30, 128));
        let original = target.clone();
        assert!(!draw_pictures_roi_with_limit(&mut target, &[direct.clone(), clipped.clone()], 1.0, 255));
        assert_same_pixels(&target, &original, "scratch limit rejection must be atomic");
        let mut invalid = clipped.clone();
        invalid.clip = Some([0.25, 0.0, 4.0, 4.0]);
        assert!(!draw_pictures_roi(&mut target, &[direct, invalid], 1.0));
        assert_same_pixels(&target, &original, "invalid clip rejection must be atomic");
        let mut disjoint = clipped.clone();
        disjoint.clip = Some([5.0, 5.0, 2.0, 2.0]);
        let (plans, scratch) = prepare_roi_clips(&target, &[clipped.clone(), disjoint, clipped], 1.0, 256).unwrap();
        assert_eq!(plans.len(), 3);
        assert_eq!(scratch.as_ref().unwrap().pixmap.as_ref().unwrap().data().len(), 256,
            "identical and distinct clips share one viewport byte budget");
    }

    #[test]
    fn roi_clips_accept_logical_half_pixel_edges_only_at_dpr_two() {
        let mut picture = layer([64, 96, 128, 160]);
        picture.transform = [0.8, 0.6, -0.6, 0.8, 4.125, 3.375];
        picture.clip = Some([2.5, 1.5, 6.5, 7.5]);
        let mut target = Pixmap::new(24, 20).unwrap();
        let original = target.clone();
        assert!(!draw_pictures_roi(&mut target, &[picture.clone()], 1.0));
        assert_same_pixels(&target, &original, "fractional device clip rejection");
        let mut expected = target.clone();
        assert!(draw_pictures_roi(&mut target, &[picture.clone()], 2.0));
        assert!(draw_pictures_impl::<false>(&mut expected, &[picture], 2.0));
        assert_same_pixels(&target, &expected, "DPR2 half-logical-pixel clip");
    }

    #[test]
    #[ignore = "release-only alternating masked/ROI Retina composition benchmark"]
    fn roi_clip_composition_release_ab() {
        use std::{hint::black_box, time::{Duration, Instant}};
        assert!(!cfg!(debug_assertions), "run with --release");
        let mut rgba = Vec::new();
        for y in 0..320 {
            for x in 0..512 {
                let alpha = if x % 32 < 20 && y % 24 < 16 { 180 } else { 0 };
                rgba.extend_from_slice(&[alpha / 2, alpha / 3, alpha, alpha]);
            }
        }
        let surface = RgbaSurface::new(Arc::new(rgba), 512, 320).unwrap();
        type Draw = fn(&mut Pixmap, &[PictureLayer], f32) -> bool;
        let kernels: [Draw; 2] = [draw_pictures, draw_pictures_roi];
        for count in [1, 3, 6] {
            let pictures: Vec<_> = (0..count).map(|index| {
                let group = index % 3;
                let x = 100.0 + group as f32 * 380.0;
                let y = 120.0 + group as f32 * 100.0;
                PictureLayer {
                    surface: surface.clone(),
                    transform: [0.5, 0.0, 0.0, 0.5, x + 12.125, y + 10.375],
                    opacity: 0.7, clip: Some([x, y, 360.0, 300.0]),
                }
            }).collect();
            let mut target = Pixmap::new(3840, 2410).unwrap();
            target.fill(tiny_skia::Color::from_rgba8(20, 40, 60, 160));
            let mut reference = target.clone();
            assert!(kernels[0](&mut reference, &pictures, 2.0));
            assert!(kernels[1](&mut target, &pictures, 2.0));
            assert_same_pixels(&target, &reference, "benchmark parity prerequisite");
            let mut samples = [Vec::new(), Vec::new()];
            for round in 0..8 {
                for index in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..3 {
                        target.fill(tiny_skia::Color::from_rgba8(20, 40, 60, 160));
                        let start = Instant::now();
                        assert!(kernels[index](black_box(&mut target), black_box(&pictures), 2.0));
                        elapsed += start.elapsed();
                        black_box(target.data());
                    }
                    samples[index].push(elapsed / 3);
                }
            }
            for samples in &mut samples { samples.sort_unstable(); }
            let (_, scratch) = prepare_roi_clips(&target, &pictures, 2.0, MAX_PICTURE_BYTES).unwrap();
            let scratch_bytes = scratch.as_ref().unwrap().pixmap.as_ref().unwrap().data().len();
            eprintln!("3840x2410 DPR2 pictures={count}: masked {:?}, global-scratch ROI {:?}, ratio {:.3}, scratch bytes {}",
                samples[0][4], samples[1][4], samples[0][4].as_secs_f64() / samples[1][4].as_secs_f64(),
                scratch_bytes);
        }
    }

    fn layer(color: [u8; 4]) -> PictureLayer {
        PictureLayer {
            surface: RgbaSurface::new(Arc::new(color.repeat(4)), 2, 2).unwrap(),
            transform: [1., 0., 0., 1., 0., 0.],
            opacity: 1.,
            clip: None,
        }
    }
    #[test]
    fn order_opacity_affine_clip_and_dpr() {
        let mut red = layer([255, 0, 0, 255]);
        red.transform = [0., 1., -1., 0., 4., 1.];
        let mut blue = layer([0, 0, 255, 255]);
        blue.transform = [1., 0., 0., 1., 3., 1.];
        blue.opacity = 0.5;
        blue.clip = Some([3., 1., 1., 2.]);
        let mut out = Pixmap::new(8, 8).unwrap();
        assert!(draw_pictures(&mut out, &[red, blue], 1.));
        assert_eq!(
            &out.data()[(1 * 8 + 2) * 4..(1 * 8 + 2) * 4 + 4],
            &[255, 0, 0, 255]
        );
        let pixel = &out.data()[(1 * 8 + 3) * 4..(1 * 8 + 3) * 4 + 4];
        assert!(pixel[0] >= 126 && pixel[2] >= 126 && pixel[3] == 255);
        assert_eq!(out.data()[(1 * 8 + 4) * 4 + 3], 0);
        let mut at_dpr = layer([255, 0, 0, 255]);
        at_dpr.transform = [0.5, 0., 0., 0.5, 1., 1.];
        let mut out = Pixmap::new(8, 8).unwrap();
        assert!(draw_pictures(&mut out, &[at_dpr], 2.));
        assert_eq!(out.data()[(2 * 8 + 2) * 4 + 3], 255);
    }
    #[test]
    fn invalid_batches_are_atomic() {
        for invalid in [f32::NAN, f32::INFINITY, -1., 2.] {
            let mut bad = layer([255, 0, 0, 255]);
            bad.opacity = invalid;
            let mut out = Pixmap::new(4, 4).unwrap();
            assert!(!draw_pictures(
                &mut out,
                &[layer([255, 0, 0, 255]), bad],
                1.
            ));
            assert!(out.data().iter().all(|byte| *byte == 0));
        }
        let mut bad = layer([255, 0, 0, 255]);
        bad.clip = Some([0.25, 0., 1., 1.]);
        assert!(!valid_pictures(&[bad], 1.));
        assert!(!valid_pictures(
            &vec![layer([255, 0, 0, 255]); MAX_PICTURES + 1],
            1.
        ));
        // Reject a finite positive clip whose right edge collapses in f32.
        let mut bad = layer([0, 0, 255, 255]);
        bad.clip = Some([1.0e20, 0., 1., 1.]);
        let mut out = Pixmap::new(4, 4).unwrap();
        assert!(!draw_pictures(
            &mut out,
            &[layer([255, 0, 0, 255]), bad],
            1.
        ));
        assert!(out.data().iter().all(|byte| *byte == 0));
    }

    #[test]
    fn full_target_clip_skip_matches_mask_reference_exactly() {
        let surface = RgbaSurface::new(
            Arc::new(vec![128, 0, 0, 128, 0, 96, 0, 96, 0, 0, 64, 64, 0, 0, 0, 0]),
            2,
            2,
        )
        .unwrap();
        for scale in [1., 2.] {
            for clip in [
                Some([0., 0., 10., 8.]),
                Some([-2., -1., 14., 10.]),
                Some([1., 1., 8., 6.]),
                Some([0., 0., 0., 8.]),
                None,
            ] {
                for opacity in [0., 0.3, 0.7, 1.] {
                    for [a, b, c, d] in [[1., 0., 0., 1.], [0., 1., -1., 0.], [0.8, 0.6, -0.6, 0.8]]
                    {
                        let mut pictures = vec![PictureLayer {
                            surface: surface.clone(),
                            transform: [a, b, c, d, 3.25, 2.75],
                            opacity,
                            clip,
                        }];
                        let mut second = pictures[0].clone();
                        second.transform[4] += 0.25;
                        second.opacity = 0.6;
                        second.clip = Some([2., 1., 6., 5.]);
                        pictures.push(second);
                        let mut opaque = layer([0, 255, 0, 255]);
                        opaque.transform[4] = 6.;
                        opaque.transform[5] = 4.;
                        opaque.clip = Some([0., 0., 10., 8.]);
                        pictures.push(opaque);
                        for background in [
                            tiny_skia::Color::TRANSPARENT,
                            tiny_skia::Color::from_rgba8(20, 40, 60, 160),
                        ] {
                            let mut actual =
                                Pixmap::new((10. * scale) as u32, (8. * scale) as u32).unwrap();
                            actual.fill(background);
                            let mut expected = actual.clone();
                            assert!(draw_pictures(&mut actual, &pictures, scale));
                            assert!(draw_pictures_impl::<false>(&mut expected, &pictures, scale));
                            if let Some(i) = actual
                                .data()
                                .iter()
                                .zip(expected.data())
                                .position(|(a, b)| a != b)
                            {
                                panic!(
                                    "DPR {scale}, clip {clip:?}, opacity {opacity}, matrix {a},{b},{c},{d}, byte {i}: {} != {}",
                                    actual.data()[i],
                                    expected.data()[i]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn full_target_clip_skip_keeps_invalid_batch_atomic_and_empty_clip_invisible() {
        let mut full = layer([255, 0, 0, 255]);
        full.clip = Some([0., 0., 8., 8.]);
        let mut bad = full.clone();
        bad.clip = Some([0.25, 0., 8., 8.]);
        let mut target = Pixmap::new(8, 8).unwrap();
        target.fill(tiny_skia::Color::from_rgba8(10, 20, 30, 128));
        let original = target.data().to_vec();
        assert!(!draw_pictures(&mut target, &[full.clone(), bad], 1.));
        assert_eq!(target.data(), original);
        full.clip = Some([0., 0., 0., 8.]);
        assert!(draw_pictures(&mut target, &[full], 1.));
        assert_eq!(target.data(), original);
    }

    #[test]
    #[ignore = "release-only alternating full-target clip mask/skip Retina benchmark"]
    fn full_target_clip_skip_release_ab() {
        use std::{
            hint::black_box,
            time::{Duration, Instant},
        };
        assert!(!cfg!(debug_assertions), "run this benchmark with --release");
        let mut rgba = Vec::with_capacity(1280 * 384 * 4);
        for y in 0..384 {
            for x in 0..1280 {
                let a = if x % 32 < 20 && y % 24 < 16 { 180 } else { 0 };
                rgba.extend_from_slice(&[a / 2, a / 3, a, a]);
            }
        }
        let picture = PictureLayer {
            surface: RgbaSurface::new(Arc::new(rgba), 1280, 384).unwrap(),
            transform: [0.5, 0., 0., 0.5, 40.125, 120.375],
            opacity: 0.7,
            clip: Some([0., 0., 1280., 900.]),
        };
        type Draw = fn(&mut Pixmap, &[PictureLayer], f32) -> bool;
        let kernels: [Draw; 2] = [draw_pictures_impl::<false>, draw_pictures];
        for count in [1, 3] {
            let pictures: Vec<_> = (0..count)
                .map(|i| {
                    let mut picture = picture.clone();
                    picture.transform[4] += i as f32 * 12.5;
                    picture.transform[5] += i as f32 * 16.25;
                    picture
                })
                .collect();
            let mut target = Pixmap::new(2560, 1800).unwrap();
            for kernel in kernels {
                target.fill(tiny_skia::Color::BLACK);
                assert!(kernel(&mut target, &pictures, 2.));
            }
            let mut samples = [Vec::new(), Vec::new()];
            for round in 0..8 {
                for index in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..3 {
                        target.fill(tiny_skia::Color::BLACK);
                        let start = Instant::now();
                        assert!(kernels[index](
                            black_box(&mut target),
                            black_box(&pictures),
                            2.
                        ));
                        elapsed += start.elapsed();
                        black_box(target.data());
                    }
                    samples[index].push(elapsed / 3);
                }
            }
            for samples in &mut samples {
                samples.sort_unstable();
            }
            eprintln!(
                "2560x1800 DPR2 pictures={count}: old mask {:?}, bounds skip {:?}, speedup {:.3}",
                samples[0][4],
                samples[1][4],
                samples[0][4].as_secs_f64() / samples[1][4].as_secs_f64()
            );
        }
    }
}
