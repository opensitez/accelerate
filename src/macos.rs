use super::{LayerFrame as VideoLayerFrame, tinted_pixels as tinted_video_pixels};
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2::{MainThreadMarker, msg_send};
use objc2_core_foundation::{CFRetained, CGAffineTransform, CGPoint, CGRect, CGSize};
use objc2_core_graphics::{
    CGBitmapInfo, CGColorRenderingIntent, CGColorSpace, CGDataProvider, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, CGImageComponentInfo, CGImagePixelFormatInfo, kCGColorSpaceSRGB,
};
use objc2_foundation::NSObject;
use objc2_quartz_core::{CALayer, CATransaction, kCAGravityResize};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
fn commit_thread_cpu_time() -> Option<Duration> {
    use std::ffi::{c_int, c_long, c_uint};
    // Local Darwin SDK _time.h and arm/i386 _types.h: enum clockid_t,
    // CLOCK_THREAD_CPUTIME_ID=16; timespec contains two C longs.
    #[repr(C)]
    struct Timespec {
        seconds: c_long,
        nanoseconds: c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: c_uint, time: *mut Timespec) -> c_int;
    }
    let mut time = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    if unsafe { clock_gettime(16, &mut time) } != 0
        || time.seconds < 0
        || !(0..1_000_000_000).contains(&time.nanoseconds)
    {
        return None;
    }
    Some(Duration::new(time.seconds as u64, time.nanoseconds as u32))
}

#[cfg(not(target_os = "macos"))]
fn commit_thread_cpu_time() -> Option<Duration> {
    None
}

#[derive(Clone, Copy, Default)]
struct CommitChanges {
    geometry: bool,
    scale: bool,
    opacity: bool,
    visibility: bool,
}

struct CommitSample {
    wall: Duration,
    cpu: Option<Duration>,
    dimensions: (u32, u32),
    changes: CommitChanges,
}

#[derive(Default)]
struct CommitSummary {
    frames: u64,
    cpu_frames: u64,
    wall: Duration,
    wall_max: Duration,
    cpu: Duration,
    cpu_max: Duration,
    paired_wall: Duration,
    dimensions: (u32, u32),
    dimension_changes: u64,
    geometry_changes: u64,
    scale_changes: u64,
    opacity_changes: u64,
    visibility_changes: u64,
}

struct CommitWindow {
    started: Instant,
    summary: CommitSummary,
}

impl CommitWindow {
    fn new(now: Instant) -> Self {
        Self {
            started: now,
            summary: CommitSummary::default(),
        }
    }

    fn record(&mut self, sample: CommitSample, now: Instant) -> Option<CommitSummary> {
        let summary = &mut self.summary;
        if summary.frames != 0 && summary.dimensions != sample.dimensions {
            summary.dimension_changes += 1;
        }
        summary.frames += 1;
        summary.wall += sample.wall;
        summary.wall_max = summary.wall_max.max(sample.wall);
        if let Some(cpu) = sample.cpu {
            summary.cpu_frames += 1;
            summary.cpu += cpu;
            summary.cpu_max = summary.cpu_max.max(cpu);
            summary.paired_wall += sample.wall;
        }
        summary.dimensions = sample.dimensions;
        summary.geometry_changes += u64::from(sample.changes.geometry);
        summary.scale_changes += u64::from(sample.changes.scale);
        summary.opacity_changes += u64::from(sample.changes.opacity);
        summary.visibility_changes += u64::from(sample.changes.visibility);
        if now.saturating_duration_since(self.started) < Duration::from_secs(1) {
            return None;
        }
        self.started = now;
        Some(std::mem::take(&mut self.summary))
    }
}

thread_local! {
    static COMMIT_DIAGNOSTICS: std::cell::RefCell<Option<CommitWindow>> = const { std::cell::RefCell::new(None) };
}

fn record_commit_sample(sample: CommitSample) {
    let now = Instant::now();
    let summary = COMMIT_DIAGNOSTICS.with(|slot| {
        slot.borrow_mut()
            .get_or_insert_with(|| CommitWindow::new(now))
            .record(sample, now)
    });
    let Some(summary) = summary else {
        return;
    };
    let cpu_average = if summary.cpu_frames == 0 {
        f64::NAN
    } else {
        summary.cpu.as_secs_f64() * 1000.0 / summary.cpu_frames as f64
    };
    let unaccounted = if summary.cpu_frames == 0 {
        f64::NAN
    } else {
        summary
            .paired_wall
            .saturating_sub(summary.cpu)
            .as_secs_f64()
            * 1000.0
            / summary.cpu_frames as f64
    };
    eprintln!(
        "media framebuffer commit size={}x{} frames={} wall_avg_ms={:.3} wall_max_ms={:.3} main_cpu_avg_ms={:.3} main_cpu_max_ms={:.3} wall_minus_cpu_avg_ms={:.3} cpu_samples={} cpu_unavailable={} dimension_changes={} geometry_changes={} scale_changes={} opacity_changes={} visibility_changes={}",
        summary.dimensions.0,
        summary.dimensions.1,
        summary.frames,
        summary.wall.as_secs_f64() * 1000.0 / summary.frames as f64,
        summary.wall_max.as_secs_f64() * 1000.0,
        cpu_average,
        if summary.cpu_frames == 0 {
            f64::NAN
        } else {
            summary.cpu_max.as_secs_f64() * 1000.0
        },
        unaccounted,
        summary.cpu_frames,
        summary.frames - summary.cpu_frames,
        summary.dimension_changes,
        summary.geometry_changes,
        summary.scale_changes,
        summary.opacity_changes,
        summary.visibility_changes
    );
}

#[cfg(test)]
mod commit_diagnostic_tests {
    use super::*;

    #[test]
    fn commit_window_is_bounded_and_reports_at_most_once_per_second() {
        let now = Instant::now();
        let mut window = CommitWindow::new(now);
        for frame in 0..100 {
            assert!(
                window
                    .record(
                        CommitSample {
                            wall: Duration::from_millis(24),
                            cpu: Some(Duration::from_millis(3)),
                            dimensions: (2560, 1800),
                            changes: CommitChanges::default(),
                        },
                        now + Duration::from_millis(frame)
                    )
                    .is_none()
            );
        }
        let summary = window
            .record(
                CommitSample {
                    wall: Duration::from_millis(32),
                    cpu: None,
                    dimensions: (3840, 2346),
                    changes: CommitChanges {
                        geometry: true,
                        scale: true,
                        opacity: true,
                        visibility: true,
                    },
                },
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!((summary.frames, summary.cpu_frames), (101, 100));
        assert_eq!(summary.wall, Duration::from_millis(2432));
        assert_eq!(summary.wall_max, Duration::from_millis(32));
        assert_eq!(summary.cpu, Duration::from_millis(300));
        assert_eq!(summary.cpu_max, Duration::from_millis(3));
        assert_eq!(
            summary.paired_wall.saturating_sub(summary.cpu),
            Duration::from_millis(2100)
        );
        assert_eq!(summary.dimensions, (3840, 2346));
        assert_eq!(summary.dimension_changes, 1);
        assert_eq!(
            (
                summary.geometry_changes,
                summary.scale_changes,
                summary.opacity_changes,
                summary.visibility_changes
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(window.summary.frames, 0);
        assert!(
            window
                .record(
                    CommitSample {
                        wall: Duration::from_millis(1),
                        cpu: None,
                        dimensions: (3840, 2346),
                        changes: CommitChanges::default(),
                    },
                    now + Duration::from_millis(1999)
                )
                .is_none()
        );
    }

    #[test]
    fn commit_window_does_not_invent_cpu_samples_when_clock_is_unavailable() {
        let now = Instant::now();
        let mut window = CommitWindow::new(now);
        let summary = window
            .record(
                CommitSample {
                    wall: Duration::from_millis(24),
                    cpu: None,
                    dimensions: (2560, 1800),
                    changes: CommitChanges::default(),
                },
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(summary.cpu_frames, 0);
        assert_eq!(summary.cpu, Duration::ZERO);
        assert_eq!(summary.paired_wall, Duration::ZERO);
    }

    #[test]
    fn commit_thread_cpu_clock_has_valid_monotonic_samples() {
        #[cfg(target_os = "macos")]
        {
            let before = commit_thread_cpu_time().expect("Darwin thread CPU clock");
            for value in 0..1000 {
                std::hint::black_box(value);
            }
            let after = commit_thread_cpu_time().expect("Darwin thread CPU clock");
            assert!(after >= before);
        }
        #[cfg(not(target_os = "macos"))]
        assert!(commit_thread_cpu_time().is_none());
    }
}

fn set_layer_frame(layer: &CALayer, frame: CGRect) -> bool {
    if layer.frame() != frame {
        layer.setFrame(frame);
        return true;
    }
    false
}

fn set_layer_hidden(layer: &CALayer, hidden: bool) -> bool {
    if layer.isHidden() != hidden {
        layer.setHidden(hidden);
        return true;
    }
    false
}

struct NativePicture {
    container: Retained<CALayer>,
    content: Retained<CALayer>,
    pixels: Option<super::RgbaSurface>,
}

impl Drop for NativePicture {
    fn drop(&mut self) {
        self.container.removeFromSuperlayer();
    }
}

pub(super) struct VideoLayer {
    pub(super) diagnostics: super::Diagnostics,
    root: Retained<CALayer>,
    framebuffer_layer: Retained<CALayer>,
    layer: Retained<CALayer>,
    foreground_layer: Retained<CALayer>,
    mask_layer: Retained<CALayer>,
    color_space: CFRetained<CGColorSpace>,
    last_pixels: Option<Arc<Vec<u8>>>,
    last_tint: Option<[u8; 4]>,
    last_foreground: Option<Arc<Vec<u8>>>,
    last_mask: Option<super::RgbaSurface>,
    last_framebuffer: std::cell::RefCell<Option<(Arc<tiny_skia::Pixmap>, bool)>>,
    pictures: Vec<NativePicture>,
}

impl VideoLayer {
    pub(super) fn new(window: &impl HasWindowHandle) -> Option<Self> {
        let _main_thread = MainThreadMarker::new()?;
        let handle = window.window_handle().ok()?;
        let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
            return None;
        };
        let view: &NSObject = unsafe { handle.ns_view.cast().as_ref() };
        let root: Retained<CALayer> = unsafe { msg_send![view, layer] };
        let framebuffer_layer = CALayer::new();
        framebuffer_layer.setAnchorPoint(CGPoint::new(0.0, 0.0));
        framebuffer_layer.setGeometryFlipped(true);
        framebuffer_layer.setContentsGravity(unsafe { kCAGravityResize });
        framebuffer_layer.setHidden(true);
        root.addSublayer(&framebuffer_layer);
        let layer = CALayer::new();
        layer.setAnchorPoint(CGPoint::new(0.0, 0.0));
        layer.setMasksToBounds(true);
        layer.setHidden(true);
        root.addSublayer(&layer);
        let foreground_layer = CALayer::new();
        foreground_layer.setAnchorPoint(CGPoint::new(0.0, 0.0));
        foreground_layer.setContentsGravity(unsafe { kCAGravityResize });
        foreground_layer.setHidden(true);
        root.addSublayer(&foreground_layer);
        let mask_layer = CALayer::new();
        mask_layer.setAnchorPoint(CGPoint::new(0.0, 0.0));
        mask_layer.setContentsGravity(unsafe { kCAGravityResize });
        Some(Self {
            diagnostics: super::Diagnostics::default(),
            root,
            framebuffer_layer,
            layer,
            foreground_layer,
            mask_layer,
            color_space: presentation_color_space()?,
            last_pixels: None,
            last_tint: None,
            last_foreground: None,
            last_mask: None,
            last_framebuffer: Default::default(),
            pictures: Vec::new(),
        })
    }

    pub(super) fn present(&mut self, frame: Option<&VideoLayerFrame>) {
        self.present_with_crop(frame, None);
    }

    pub(super) fn present_pictures(&mut self, pictures: &[super::PictureLayer]) -> bool {
        let scale = self.root.contentsScale() as f32;
        if !super::picture::valid_pictures(pictures, scale) {
            return false;
        }
        let mut images = Vec::with_capacity(pictures.len());
        // Prepare images before changing layers, so image creation failure is atomic.
        for (index, picture) in pictures.iter().enumerate() {
            let unchanged = self
                .pictures
                .get(index)
                .and_then(|slot| slot.pixels.as_ref())
                .is_some_and(|last| {
                    last.dimensions() == picture.surface.dimensions()
                        && Arc::ptr_eq(last.pixels(), picture.surface.pixels())
                });
            if unchanged {
                images.push(None);
            } else {
                let (w, h) = picture.surface.dimensions();
                let Some(image) = self.image_for_pixels(picture.surface.pixels(), w, h) else {
                    return false;
                };
                images.push(Some(image));
            }
        }
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        self.pictures.truncate(pictures.len());
        while self.pictures.len() < pictures.len() {
            let container = CALayer::new();
            container.setAnchorPoint(CGPoint::new(0., 0.));
            // Cancel the host's geometry flip: this clip uses bottom-origin
            // coordinates, matching CGImage contents and the affine below.
            container.setGeometryFlipped(self.root.isGeometryFlipped());
            container.setMasksToBounds(true);
            let content = CALayer::new();
            content.setAnchorPoint(CGPoint::new(0., 0.));
            content.setContentsGravity(unsafe { kCAGravityResize });
            container.addSublayer(&content);
            self.root.addSublayer(&container);
            self.pictures.push(NativePicture {
                container,
                content,
                pixels: None,
            });
        }
        let root = self.root.bounds();
        for ((slot, picture), image) in self.pictures.iter_mut().zip(pictures).zip(images) {
            slot.container
                .setGeometryFlipped(self.root.isGeometryFlipped());
            let [x, y, w, h] =
                picture
                    .clip
                    .unwrap_or([0., 0., root.size.width as f32, root.size.height as f32]);
            let window_y = layer_y_for_window_y(
                root.size.height,
                y as f64,
                h as f64,
                self.root.isGeometryFlipped(),
            );
            set_layer_frame(
                &slot.container,
                CGRect::new(
                    CGPoint::new(x as f64, window_y),
                    CGSize::new(w as f64, h as f64),
                ),
            );
            slot.container.setBounds(CGRect::new(
                CGPoint::new(0., 0.),
                CGSize::new(w as f64, h as f64),
            ));
            slot.container
                .setHidden(w == 0. || h == 0. || picture.opacity == 0.);
            let (width, height) = picture.surface.dimensions();
            slot.content.setBounds(CGRect::new(
                CGPoint::new(0., 0.),
                CGSize::new(width as f64, height as f64),
            ));
            slot.content.setPosition(CGPoint::new(0., 0.));
            let [a, b, c, d, e, f] = picture.transform.map(f64::from);
            // Conjugate top-origin source/destination coordinates with vertical
            // reflections; source image rows and clip-local geometry both invert.
            let source_h = height as f64;
            slot.content.setAffineTransform(CGAffineTransform {
                a,
                b: -b,
                c: -c,
                d,
                tx: e - x as f64 + c * source_h,
                ty: h as f64 - (f - y as f64) - d * source_h,
            });
            slot.content.setOpacity(picture.opacity);
            if let Some(image) = image {
                unsafe {
                    slot.content.setContents(Some(image.as_ref() as &AnyObject));
                }
                slot.pixels = Some(picture.surface.clone());
            }
        }
        CATransaction::commit();
        true
    }

    pub(super) fn clear_pictures(&mut self) {
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        self.pictures.clear();
        CATransaction::commit();
    }

    pub(super) fn present_with_crop(
        &mut self,
        frame: Option<&VideoLayerFrame>,
        crop: Option<[f32; 4]>,
    ) {
        self.present_with_mask(frame, crop, None);
    }

    pub(super) fn present_masked_cropped_layer(
        &mut self,
        layer: Option<&super::CroppedLayerFrame>,
        mask: Option<&super::RgbaSurface>,
    ) -> bool {
        if layer.is_some_and(|layer| !layer.is_valid()) || !super::layer::valid_mask(mask) {
            return false;
        }
        self.present_with_mask(
            layer.map(|layer| &layer.frame),
            layer.map(|layer| layer.crop),
            layer.and(mask),
        )
    }

    fn present_with_mask(
        &mut self,
        frame: Option<&VideoLayerFrame>,
        crop: Option<[f32; 4]>,
        mask: Option<&super::RgbaSurface>,
    ) -> bool {
        let mask_changed = match (&self.last_mask, mask) {
            (Some(last), Some(mask)) => {
                last.dimensions() != mask.dimensions() || !Arc::ptr_eq(last.pixels(), mask.pixels())
            }
            (None, None) => false,
            _ => true,
        };
        let mask_image = if mask_changed {
            if let Some(mask) = mask {
                let (width, height) = mask.dimensions();
                let Some(image) = self.image_for_pixels(mask.pixels(), width, height) else {
                    return false;
                };
                Some(image)
            } else {
                None
            }
        } else {
            None
        };
        let Some(frame) = frame.filter(|frame| {
            frame.source_width > 0
                && frame.source_height > 0
                && frame.width > 0.0
                && frame.height > 0.0
                && super::software::rgba_size_matches(
                    frame.rgba.len(),
                    frame.source_width,
                    frame.source_height,
                )
        }) else {
            set_layer_hidden(&self.layer, true);
            set_layer_hidden(&self.foreground_layer, true);
            self.last_pixels = None;
            self.last_tint = None;
            self.last_foreground = None;
            unsafe { self.layer.setMask(None) };
            unsafe { self.mask_layer.setContents(None) };
            self.last_mask = None;
            return true;
        };

        CATransaction::begin();
        CATransaction::setDisableActions(true);
        let y = layer_y_for_window_y(
            self.root.bounds().size.height,
            frame.y as f64,
            frame.height as f64,
            self.root.isGeometryFlipped(),
        );
        set_layer_frame(
            &self.layer,
            CGRect::new(
                CGPoint::new(frame.x as f64, y),
                CGSize::new(frame.width as f64, frame.height as f64),
            ),
        );
        if self.layer.cornerRadius() != frame.corner_radius as f64 {
            self.layer.setCornerRadius(frame.corner_radius as f64);
        }
        let (x, y, width, height) = crop
            .map(|crop| {
                (
                    crop[0] as f64,
                    crop[1] as f64,
                    crop[2] as f64,
                    crop[3] as f64,
                )
            })
            .unwrap_or_else(|| frame.contents_rect());
        self.layer
            .setContentsRect(CGRect::new(CGPoint::new(x, y), CGSize::new(width, height)));
        self.layer.setContentsGravity(unsafe { kCAGravityResize });
        if let Some(mask) = mask {
            set_layer_frame(
                &self.mask_layer,
                CGRect::new(
                    CGPoint::new(0.0, 0.0),
                    CGSize::new(frame.width as f64, frame.height as f64),
                ),
            );
            if let Some(image) = mask_image {
                unsafe {
                    self.mask_layer
                        .setContents(Some(image.as_ref() as &AnyObject))
                };
            }
            unsafe { self.layer.setMask(Some(&self.mask_layer)) };
            self.last_mask = Some(mask.clone());
        } else {
            unsafe { self.layer.setMask(None) };
            if mask_changed {
                unsafe { self.mask_layer.setContents(None) };
            }
            self.last_mask = None;
        }
        if self.last_tint != frame.tint
            || !self
                .last_pixels
                .as_ref()
                .is_some_and(|last| Arc::ptr_eq(last, &frame.rgba))
        {
            let _span = self.diagnostics.span(super::Phase::LayerPresent);
            let pixels = tinted_video_pixels(&frame.rgba, frame.tint);
            if let Some(image) =
                self.image_for_pixels(&pixels, frame.source_width, frame.source_height)
            {
                let opaque =
                    CGImage::alpha_info(Some(&image)).0 == CGImageAlphaInfo::NoneSkipLast.0;
                if self.layer.isOpaque() != opaque {
                    self.layer.setOpaque(opaque);
                }
                unsafe { self.layer.setContents(Some(image.as_ref() as &AnyObject)) };
                self.last_pixels = Some(frame.rgba.clone());
                self.last_tint = frame.tint;
            }
        }
        set_layer_hidden(&self.layer, false);
        if let Some(foreground) = &frame.foreground {
            let y = layer_y_for_window_y(
                self.root.bounds().size.height,
                foreground.y as f64,
                foreground.height as f64,
                self.root.isGeometryFlipped(),
            );
            set_layer_frame(
                &self.foreground_layer,
                CGRect::new(
                    CGPoint::new(foreground.x as f64, y),
                    CGSize::new(foreground.width as f64, foreground.height as f64),
                ),
            );
            if !self
                .last_foreground
                .as_ref()
                .is_some_and(|last| Arc::ptr_eq(last, &foreground.rgba))
            {
                if let Some(image) = self.image_for_pixels(
                    &foreground.rgba,
                    foreground.source_width,
                    foreground.source_height,
                ) {
                    unsafe {
                        self.foreground_layer
                            .setContents(Some(image.as_ref() as &AnyObject))
                    };
                    self.last_foreground = Some(foreground.rgba.clone());
                }
            }
            set_layer_hidden(&self.foreground_layer, false);
        } else {
            set_layer_hidden(&self.foreground_layer, true);
            self.last_foreground = None;
        }
        CATransaction::commit();
        true
    }

    pub(super) fn present_framebuffer_with_pictures(
        &mut self,
        pixels: &Arc<tiny_skia::Pixmap>,
        pictures: &[super::PictureLayer],
    ) -> bool {
        if !super::picture::valid_pictures(pictures, self.root.contentsScale() as f32) {
            return false;
        }
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        let changes = self.stage_framebuffer(pixels);
        let success = changes.is_some() && self.present_pictures(pictures);
        self.commit_framebuffer_transaction(pixels, changes.unwrap_or_default());
        success
    }

    pub(super) fn present_framebuffer(&self, pixels: &Arc<tiny_skia::Pixmap>) -> bool {
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        let changes = self.stage_framebuffer(pixels);
        self.commit_framebuffer_transaction(pixels, changes.unwrap_or_default());
        changes.is_some()
    }

    fn stage_framebuffer(&self, pixels: &Arc<tiny_skia::Pixmap>) -> Option<CommitChanges> {
        self.stage_framebuffer_in_format(
            pixels,
            std::env::var("ACCELERATE_MACOS_FRAMEBUFFER_BGRA").as_deref() == Ok("1"),
        )
    }

    fn stage_framebuffer_in_format(
        &self,
        pixels: &Arc<tiny_skia::Pixmap>,
        bgra: bool,
    ) -> Option<CommitChanges> {
        // Retaining the previous Arc makes identity reuse immutable even for the
        // BGRA path, whose CGImage owns a converted copy rather than this Pixmap.
        let unchanged = self
            .last_framebuffer
            .borrow()
            .as_ref()
            .is_some_and(|(last, format)| Arc::ptr_eq(last, pixels) && *format == bgra);
        let image = if unchanged {
            None
        } else {
            let _span = self.diagnostics.span(super::Phase::ImageCreate);
            let image = if bgra {
                self.image_for_bgra_framebuffer(pixels.data(), pixels.width(), pixels.height())
            } else {
                self.image_for_data(pixels, pixels.data(), pixels.width(), pixels.height())
            };
            let Some(image) = image else {
                self.last_framebuffer.borrow_mut().take();
                set_layer_hidden(&self.framebuffer_layer, true);
                return None;
            };
            Some(image)
        };
        let geometry_changed = set_layer_frame(&self.framebuffer_layer, self.root.bounds());
        let scale = self.root.contentsScale();
        let scale_changed = self.framebuffer_layer.contentsScale() != scale;
        if scale_changed {
            self.framebuffer_layer.setContentsScale(scale);
        }
        let mut opacity_changed = false;
        if let Some(image) = image {
            let _span = self.diagnostics.span(super::Phase::Upload);
            let alpha = CGImage::alpha_info(Some(&image));
            let opaque =
                alpha == CGImageAlphaInfo::NoneSkipLast || alpha == CGImageAlphaInfo::NoneSkipFirst;
            opacity_changed = self.framebuffer_layer.isOpaque() != opaque;
            if opacity_changed {
                self.framebuffer_layer.setOpaque(opaque);
            }
            unsafe {
                self.framebuffer_layer
                    .setContents(Some(image.as_ref() as &AnyObject))
            };
            *self.last_framebuffer.borrow_mut() = Some((pixels.clone(), bgra));
        }
        let visibility_changed = set_layer_hidden(&self.framebuffer_layer, false);
        Some(CommitChanges {
            geometry: geometry_changed,
            scale: scale_changed,
            opacity: opacity_changed,
            visibility: visibility_changed,
        })
    }

    fn commit_framebuffer_transaction(
        &self,
        pixels: &Arc<tiny_skia::Pixmap>,
        changes: CommitChanges,
    ) {
        let diagnostics = (self.diagnostics.media_enabled)() && MainThreadMarker::new().is_some();
        let cpu_start = diagnostics.then(commit_thread_cpu_time).flatten();
        let sample;
        {
            let _span = self.diagnostics.span(super::Phase::Commit);
            let wall_start = diagnostics.then(Instant::now);
            CATransaction::commit();
            let wall = wall_start.map(|start| start.elapsed());
            let cpu = if diagnostics {
                cpu_start
                    .zip(commit_thread_cpu_time())
                    .and_then(|(start, end)| end.checked_sub(start))
            } else {
                None
            };
            sample = wall.map(|wall| CommitSample {
                wall,
                cpu,
                dimensions: (pixels.width(), pixels.height()),
                changes,
            });
        }
        if let Some(sample) = sample {
            record_commit_sample(sample);
        }
    }

    fn image_for_pixels(
        &self,
        pixels: &Arc<Vec<u8>>,
        width: u32,
        height: u32,
    ) -> Option<CFRetained<CGImage>> {
        self.image_for_data(pixels, pixels.as_slice(), width, height)
    }

    fn image_for_data<T: Send + Sync + 'static>(
        &self,
        owner: &Arc<T>,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Option<CFRetained<CGImage>> {
        if !super::software::rgba_size_matches(pixels.len(), width, height) {
            return None;
        }
        let alpha = if super::software::rgba_is_opaque(pixels) {
            CGImageAlphaInfo::NoneSkipLast
        } else {
            CGImageAlphaInfo::PremultipliedLast
        };
        self.image_for_bitmap_data(
            owner,
            pixels,
            width,
            height,
            CGBitmapInfo(
                alpha.0
                    | CGImageComponentInfo::Integer.0
                    | CGImageByteOrderInfo::Order32Big.0
                    | CGImagePixelFormatInfo::Packed.0,
            ),
        )
    }

    fn image_for_bgra_framebuffer(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
    ) -> Option<CFRetained<CGImage>> {
        if !super::software::rgba_size_matches(rgba.len(), width, height) {
            return None;
        }
        // Convert actual bytes, not just CGImage metadata. The provider owns this
        // immutable snapshot independently of the caller's recyclable framebuffer.
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(rgba.len()).ok()?;
        bytes.extend_from_slice(rgba);
        for pixel in bytes.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        let owner = Arc::new(bytes);
        let alpha = if super::software::rgba_is_opaque(&owner) {
            CGImageAlphaInfo::NoneSkipFirst
        } else {
            CGImageAlphaInfo::PremultipliedFirst
        };
        self.image_for_bitmap_data(
            &owner,
            owner.as_slice(),
            width,
            height,
            CGBitmapInfo(
                alpha.0
                    | CGImageComponentInfo::Integer.0
                    | CGImageByteOrderInfo::Order32Little.0
                    | CGImagePixelFormatInfo::Packed.0,
            ),
        )
    }

    fn image_for_bitmap_data<T: Send + Sync + 'static>(
        &self,
        owner: &Arc<T>,
        pixels: &[u8],
        width: u32,
        height: u32,
        bitmap_info: CGBitmapInfo,
    ) -> Option<CFRetained<CGImage>> {
        if !super::software::rgba_size_matches(pixels.len(), width, height) {
            return None;
        }
        unsafe extern "C-unwind" fn release<T>(
            info: *mut c_void,
            _data: NonNull<c_void>,
            _size: usize,
        ) {
            drop(unsafe { Arc::<T>::from_raw(info.cast()) });
        }
        let retained = Arc::into_raw(owner.clone()) as *mut c_void;
        let provider = unsafe {
            CGDataProvider::with_data(
                retained,
                pixels.as_ptr().cast_mut().cast(),
                pixels.len(),
                Some(release::<T>),
            )
        };
        let Some(provider) = provider else {
            drop(unsafe { Arc::<T>::from_raw(retained.cast()) });
            return None;
        };
        unsafe {
            CGImage::new(
                width as usize,
                height as usize,
                8,
                32,
                width as usize * 4,
                Some(&self.color_space),
                bitmap_info,
                Some(&provider),
                std::ptr::null(),
                false,
                CGColorRenderingIntent::RenderingIntentDefault,
            )
        }
    }
}

fn presentation_color_space() -> Option<CFRetained<CGColorSpace>> {
    // The CPU framebuffer and decoded RGBA samples are sRGB, not monitor RGB.
    CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }))
}

fn layer_y_for_window_y(root_height: f64, y: f64, height: f64, flipped: bool) -> f64 {
    if flipped { y } else { root_height - y - height }
}

#[test]
fn native_video_frame_stays_below_browser_chrome_in_either_coordinate_system() {
    assert_eq!(layer_y_for_window_y(1080.0, 80.0, 1000.0, true), 80.0);
    assert_eq!(layer_y_for_window_y(1080.0, 80.0, 1000.0, false), 0.0);
}

#[test]
fn presentation_images_retain_pixels_and_preserve_transparency() {
    let layer = VideoLayer {
        diagnostics: super::Diagnostics::default(),
        root: CALayer::new(),
        framebuffer_layer: CALayer::new(),
        layer: CALayer::new(),
        foreground_layer: CALayer::new(),
        mask_layer: CALayer::new(),
        color_space: CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB })).unwrap(),
        last_pixels: None,
        last_tint: None,
        last_foreground: None,
        last_mask: None,
        last_framebuffer: Default::default(),
        pictures: Vec::new(),
    };
    for alpha in [255, 128] {
        let pixels = Arc::new([30, 60, 90, alpha].repeat(4));
        let image = layer.image_for_pixels(&pixels, 2, 2).unwrap();
        let expected = if alpha == 255 {
            CGImageAlphaInfo::NoneSkipLast
        } else {
            CGImageAlphaInfo::PremultipliedLast
        };
        assert_eq!(CGImage::alpha_info(Some(&image)).0, expected.0);
        let space = CGImage::color_space(Some(&image)).unwrap();
        let name = CGColorSpace::name(Some(&space)).unwrap();
        assert_eq!(&*name, unsafe { kCGColorSpaceSRGB });
        assert_eq!(Arc::strong_count(&pixels), 2);
        drop(image);
        assert_eq!(Arc::strong_count(&pixels), 1);
    }
}

#[test]
fn ordered_pictures_retain_images_across_property_changes() {
    let mut native = VideoLayer {
        diagnostics: super::Diagnostics::default(),
        root: CALayer::new(),
        framebuffer_layer: CALayer::new(),
        layer: CALayer::new(),
        foreground_layer: CALayer::new(),
        mask_layer: CALayer::new(),
        color_space: presentation_color_space().unwrap(),
        last_pixels: None,
        last_tint: None,
        last_foreground: None,
        last_mask: None,
        last_framebuffer: Default::default(),
        pictures: Vec::new(),
    };
    native
        .root
        .setBounds(CGRect::new(CGPoint::new(0., 0.), CGSize::new(8., 8.)));
    native.root.setContentsScale(2.);
    native.root.setGeometryFlipped(true);
    let mut picture = super::PictureLayer {
        surface: super::RgbaSurface::new(Arc::new([255, 0, 0, 255].repeat(4)), 2, 2).unwrap(),
        transform: [0.5, 0., 0., 0.5, 1., 1.],
        opacity: 1.,
        clip: Some([0.5, 0.5, 3., 3.]),
    };
    assert!(native.present_pictures(&[picture.clone()]));
    let image = unsafe { native.pictures[0].content.contents() }.unwrap();
    picture.transform[4] = 2.;
    picture.opacity = 0.5;
    picture.clip = Some([1., 1., 3., 3.]);
    assert!(native.present_pictures(&[picture.clone()]));
    let retained_image = unsafe { native.pictures[0].content.contents() }.unwrap();
    assert!(std::ptr::eq(&*image, &*retained_image));
    assert_eq!(native.pictures[0].content.opacity(), 0.5);
    assert_eq!(native.pictures[0].content.affineTransform().tx, 1.);
    assert_eq!(
        native.pictures[0].container.bounds().origin,
        CGPoint::new(0., 0.)
    );
    picture.opacity = f32::NAN;
    assert!(!native.present_pictures(&[picture]));
    assert_eq!(native.pictures[0].content.opacity(), 0.5);
    assert!(native.present_pictures(&[]));
    assert!(native.pictures.is_empty());
}

#[cfg(test)]
mod framebuffer_format_tests {
    use super::*;
    use objc2_core_graphics::{CGBitmapContextCreate, CGContext};

    #[derive(Clone, Copy, Debug)]
    enum Format {
        Rgba,
        Bgra,
    }

    fn image(
        layer: &VideoLayer,
        rgba: &Arc<Vec<u8>>,
        width: u32,
        height: u32,
        format: Format,
    ) -> CFRetained<CGImage> {
        assert_eq!(rgba.len(), width as usize * height as usize * 4);
        if matches!(format, Format::Rgba) {
            return layer.image_for_pixels(rgba, width, height).unwrap();
        }
        layer
            .image_for_bgra_framebuffer(rgba, width, height)
            .unwrap()
    }

    fn test_layer() -> VideoLayer {
        VideoLayer {
            diagnostics: crate::Diagnostics::default(),
            root: CALayer::new(),
            framebuffer_layer: CALayer::new(),
            layer: CALayer::new(),
            foreground_layer: CALayer::new(),
            mask_layer: CALayer::new(),
            color_space: presentation_color_space().unwrap(),
            last_pixels: None,
            last_tint: None,
            last_foreground: None,
            last_mask: None,
            last_framebuffer: Default::default(),
            pictures: Vec::new(),
        }
    }

    thread_local! {
        static FRAMEBUFFER_PHASES: std::cell::Cell<[usize; 3]> = const {
            std::cell::Cell::new([0; 3])
        };
    }

    fn record_framebuffer_phase(phase: crate::Phase, _: Duration) {
        let index = match phase {
            crate::Phase::ImageCreate => 0,
            crate::Phase::Upload => 1,
            crate::Phase::Commit => 2,
            _ => return,
        };
        FRAMEBUFFER_PHASES.with(|counts| {
            let mut value = counts.get();
            value[index] += 1;
            counts.set(value);
        });
    }

    fn profiled_layer() -> VideoLayer {
        FRAMEBUFFER_PHASES.with(|counts| counts.set([0; 3]));
        let mut layer = test_layer();
        layer
            .root
            .setBounds(CGRect::new(CGPoint::new(0., 0.), CGSize::new(4., 1.)));
        layer.root.setContentsScale(1.);
        layer.diagnostics = crate::Diagnostics {
            enabled: || true,
            media_enabled: || false,
            record: record_framebuffer_phase,
        };
        layer
    }

    fn submit_format(
        layer: &VideoLayer,
        pixels: &Arc<tiny_skia::Pixmap>,
        bgra: bool,
    ) -> CommitChanges {
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        let changes = layer.stage_framebuffer_in_format(pixels, bgra).unwrap();
        layer.commit_framebuffer_transaction(pixels, changes);
        changes
    }

    #[test]
    fn framebuffer_identity_reuses_image_but_updates_geometry_scale_and_visibility() {
        for bgra in [false, true] {
            let native = profiled_layer();
            let mut pixels = tiny_skia::Pixmap::new(4, 1).unwrap();
            pixels.fill(tiny_skia::Color::from_rgba8(255, 0, 0, 255));
            let pixels = Arc::new(pixels);
            submit_format(&native, &pixels, bgra);
            let first = unsafe { native.framebuffer_layer.contents() }.unwrap();
            submit_format(&native, &pixels, bgra);
            let second = unsafe { native.framebuffer_layer.contents() }.unwrap();
            assert!(std::ptr::eq(&*first, &*second));
            FRAMEBUFFER_PHASES.with(|counts| assert_eq!(counts.get(), [1, 1, 2]));
            native
                .root
                .setBounds(CGRect::new(CGPoint::new(0., 0.), CGSize::new(8., 2.)));
            native.root.setContentsScale(2.);
            native.framebuffer_layer.setHidden(true);
            let changes = submit_format(&native, &pixels, bgra);
            assert!(changes.geometry && changes.scale && changes.visibility);
            assert_eq!(native.framebuffer_layer.frame(), native.root.bounds());
            assert_eq!(native.framebuffer_layer.contentsScale(), 2.);
            assert!(!native.framebuffer_layer.isHidden());
            let third = unsafe { native.framebuffer_layer.contents() }.unwrap();
            assert!(std::ptr::eq(&*first, &*third));
            FRAMEBUFFER_PHASES.with(|counts| assert_eq!(counts.get(), [1, 1, 3]));

            let mut replacement = tiny_skia::Pixmap::new(4, 1).unwrap();
            replacement.fill(tiny_skia::Color::from_rgba8(0, 0, 255, 255));
            let replacement = Arc::new(replacement);
            submit_format(&native, &replacement, bgra);
            let fourth = unsafe { native.framebuffer_layer.contents() }.unwrap();
            assert!(!std::ptr::eq(&*first, &*fourth));
            // This contents object was installed exclusively from a CGImage.
            let image = unsafe { &*((&*fourth as *const AnyObject).cast::<CGImage>()) };
            assert_eq!(render(&native, image, false), [0, 0, 255, 255].repeat(4));
            assert!(Arc::ptr_eq(
                &native.last_framebuffer.borrow().as_ref().unwrap().0,
                &replacement
            ));
            submit_format(&native, &replacement, !bgra);
            let fifth = unsafe { native.framebuffer_layer.contents() }.unwrap();
            assert!(!std::ptr::eq(&*fourth, &*fifth));
            let image = unsafe { &*((&*fifth as *const AnyObject).cast::<CGImage>()) };
            assert_eq!(render(&native, image, false), [0, 0, 255, 255].repeat(4));
            FRAMEBUFFER_PHASES.with(|counts| assert_eq!(counts.get(), [3, 3, 5]));
        }
    }

    #[test]
    fn framebuffer_picture_batch_profiles_only_one_outer_commit() {
        let mut native = profiled_layer();
        let pixels = Arc::new(tiny_skia::Pixmap::new(4, 1).unwrap());
        let picture = crate::PictureLayer {
            surface: crate::RgbaSurface::new(Arc::new(vec![255; 4]), 1, 1).unwrap(),
            transform: [1., 0., 0., 1., 0., 0.],
            opacity: 1.,
            clip: None,
        };
        assert!(native.present_framebuffer_with_pictures(&pixels, &[picture.clone()]));
        FRAMEBUFFER_PHASES.with(|counts| assert_eq!(counts.get(), [1, 1, 1]));
        let first = unsafe { native.framebuffer_layer.contents() }.unwrap();
        assert!(native.present_framebuffer_with_pictures(&pixels, &[picture]));
        let second = unsafe { native.framebuffer_layer.contents() }.unwrap();
        assert!(std::ptr::eq(&*first, &*second));
        FRAMEBUFFER_PHASES.with(|counts| assert_eq!(counts.get(), [1, 1, 2]));
        assert!(native.present_framebuffer(&pixels));
        FRAMEBUFFER_PHASES.with(|counts| assert_eq!(counts.get(), [1, 1, 3]));
    }

    #[test]
    #[ignore = "requires a macOS graphics session; offscreen CALayer composition, not onscreen GPU rasterization"]
    fn ordered_pictures_rendered_affine_opacity_clip_and_dpr_parity() {
        for (flipped, scale) in [(true, 1usize), (true, 2), (false, 1), (false, 2)] {
            let mut native = test_layer();
            native
                .root
                .setBounds(CGRect::new(CGPoint::new(0., 0.), CGSize::new(8., 8.)));
            native.root.setGeometryFlipped(flipped);
            native.root.setContentsScale(scale as f64);
            let pictures = [
                crate::PictureLayer {
                    surface: crate::RgbaSurface::new(Arc::new([255, 0, 0, 255].repeat(4)), 2, 2)
                        .unwrap(),
                    transform: [0., 1., -1., 0., 4., 1.],
                    opacity: 1.,
                    clip: None,
                },
                crate::PictureLayer {
                    surface: crate::RgbaSurface::new(Arc::new([0, 0, 255, 255].repeat(4)), 2, 2)
                        .unwrap(),
                    transform: [1., 0., 0., 1., 3., 1.],
                    opacity: 0.5,
                    clip: Some([3., 1., 1., 2.]),
                },
                crate::PictureLayer {
                    surface: crate::RgbaSurface::new(
                        Arc::new(vec![
                            0, 255, 0, 255, 255, 255, 0, 255, 255, 0, 255, 255, 0, 255, 255, 255,
                        ]),
                        2,
                        2,
                    )
                    .unwrap(),
                    transform: [1. / scale as f32, 0., 0., 1. / scale as f32, 5., 4.],
                    opacity: 1.,
                    clip: None,
                },
            ];
            assert!(native.present_pictures(&pictures));
            let size = 8 * scale;
            let mut expected = tiny_skia::Pixmap::new(size as u32, size as u32).unwrap();
            assert!(crate::draw_pictures(&mut expected, &pictures, scale as f32));
            let bitmap_context = |bytes: &mut [u8]| unsafe {
                CGBitmapContextCreate(
                    bytes.as_mut_ptr().cast(),
                    size,
                    size,
                    8,
                    size * 4,
                    Some(&native.color_space),
                    CGImageAlphaInfo::PremultipliedLast.0 | CGImageByteOrderInfo::Order32Big.0,
                )
                .unwrap()
            };
            let host_coordinates = |context: &CGContext| {
                if flipped {
                    CGContext::translate_ctm(Some(context), 0., size as f64);
                    CGContext::scale_ctm(Some(context), scale as f64, -(scale as f64));
                } else {
                    CGContext::scale_ctm(Some(context), scale as f64, scale as f64);
                }
            };
            // Independently calibrate bitmap row orientation without a CALayer.
            // The same window rectangle is y=1 in a flipped host, y=8-1-2=5
            // in a bottom-origin host. Both must occupy bitmap rows 1..3.
            let mut calibration = vec![0u8; size * size * 4];
            let context = bitmap_context(&mut calibration);
            host_coordinates(&context);
            CGContext::set_rgb_fill_color(Some(&context), 1., 0., 0., 1.);
            CGContext::fill_rect(
                Some(&context),
                CGRect::new(
                    CGPoint::new(2., if flipped { 1. } else { 5. }),
                    CGSize::new(2., 2.),
                ),
            );
            drop(context);
            for (i, rgba) in calibration.chunks_exact(4).enumerate() {
                let (x, y) = (i % size, i / size);
                let occupied =
                    (2 * scale..4 * scale).contains(&x) && (scale..3 * scale).contains(&y);
                assert_eq!(
                    rgba,
                    if occupied {
                        &[255, 0, 0, 255]
                    } else {
                        &[0, 0, 0, 0]
                    },
                    "Quartz calibration root flipped {flipped}, DPR {scale}, pixel ({x}, {y})"
                );
            }
            let mut actual = vec![0u8; size * size * 4];
            let context = bitmap_context(&mut actual);
            host_coordinates(&context);
            native.root.renderInContext(&context);
            drop(context);
            for (index, (&actual, &expected)) in actual.iter().zip(expected.data()).enumerate() {
                assert!(
                    actual.abs_diff(expected) <= 3,
                    "root flipped {flipped}, DPR {scale}, pixel ({}, {}), channel {}: native {actual}, software {expected}",
                    index / 4 % size,
                    index / 4 / size,
                    index % 4
                );
            }
        }
    }

    // Readback is rendered output, not the CGDataProvider's unchanged source bytes.
    // CALayer readback is still offscreen and is not an onscreen parity claim.
    fn render(layer: &VideoLayer, image: &CGImage, via_layer: bool) -> Vec<u8> {
        render_scaled(layer, image, via_layer, 1)
    }

    fn render_scaled(
        layer: &VideoLayer,
        image: &CGImage,
        via_layer: bool,
        scale: usize,
    ) -> Vec<u8> {
        let width = 4 * scale;
        let height = scale;
        let mut output = vec![0; width * height * 4];
        let context = unsafe {
            CGBitmapContextCreate(
                output.as_mut_ptr().cast(),
                width,
                height,
                8,
                width * 4,
                Some(&layer.color_space),
                CGImageAlphaInfo::PremultipliedLast.0 | CGImageByteOrderInfo::Order32Big.0,
            )
        }
        .unwrap();
        let rect = CGRect::new(
            CGPoint::new(0.0, 0.0),
            CGSize::new(width as f64, height as f64),
        );
        CGContext::set_rgb_fill_color(Some(&context), 0.0, 1.0, 0.0, 1.0);
        CGContext::fill_rect(Some(&context), rect);
        if via_layer {
            CATransaction::begin();
            CATransaction::setDisableActions(true);
            layer.framebuffer_layer.setFrame(rect);
            layer.framebuffer_layer.setContentsScale(1.0);
            layer
                .framebuffer_layer
                .setContentsGravity(unsafe { kCAGravityResize });
            unsafe {
                layer
                    .framebuffer_layer
                    .setContents(Some(image.as_ref() as &AnyObject))
            };
            CATransaction::commit();
            layer.framebuffer_layer.renderInContext(&context);
        } else {
            CGContext::draw_image(Some(&context), rect, Some(image));
        }
        drop(context);
        output
    }

    #[test]
    fn framebuffer_rgba_bgra_rendered_color_and_alpha_parity() {
        let layer = test_layer();
        for rgba in [
            vec![
                255, 0, 0, 255, 0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255,
            ],
            vec![255, 0, 0, 255, 0, 0, 255, 255, 128, 0, 0, 128, 0, 0, 0, 0],
        ] {
            let source = Arc::new(rgba);
            let a = image(&layer, &source, 4, 1, Format::Rgba);
            let b = image(&layer, &source, 4, 1, Format::Bgra);
            let reference = render(&layer, &a, false);
            assert_eq!(reference, render(&layer, &b, false));
            assert_eq!(
                render_scaled(&layer, &a, false, 2),
                render_scaled(&layer, &b, false, 2),
            );
            let mut retina = Vec::new();
            for _ in 0..2 {
                for pixel in source.chunks_exact(4) {
                    retina.extend_from_slice(pixel);
                    retina.extend_from_slice(pixel);
                }
            }
            let retina = Arc::new(retina);
            let retina_rgba = image(&layer, &retina, 8, 2, Format::Rgba);
            let retina_bgra = image(&layer, &retina, 8, 2, Format::Bgra);
            assert_eq!(
                render_scaled(&layer, &retina_rgba, false, 2),
                render_scaled(&layer, &retina_bgra, false, 2),
            );
            assert_eq!(&reference[..8], &[255, 0, 0, 255, 0, 0, 255, 255]);
            assert!(reference.chunks_exact(4).all(|pixel| pixel[3] == 255));
            if source[15] == 0 {
                assert_eq!(&reference[12..], &[0, 255, 0, 255]);
                assert!((127..=129).contains(&reference[8]));
                assert!((126..=128).contains(&reference[9]));
            }
        }
    }

    #[test]
    fn framebuffer_bgra_owns_snapshot_and_rejects_invalid_dimensions() {
        let layer = test_layer();
        let mut source = vec![255, 0, 0, 255, 0, 0, 255, 255, 128, 0, 0, 128, 0, 0, 0, 0];
        let original = source.clone();
        let image = layer.image_for_bgra_framebuffer(&source, 4, 1).unwrap();
        assert_eq!(source, original);
        assert_eq!(
            CGImage::alpha_info(Some(&image)),
            CGImageAlphaInfo::PremultipliedFirst,
        );
        let before = render(&layer, &image, false);
        source.fill(0);
        drop(source);
        assert_eq!(before, render(&layer, &image, false));
        assert!(
            layer
                .image_for_bgra_framebuffer(&original[..15], 4, 1)
                .is_none(),
        );
        let opaque = layer.image_for_bgra_framebuffer(&[255; 16], 4, 1).unwrap();
        assert_eq!(
            CGImage::alpha_info(Some(&opaque)),
            CGImageAlphaInfo::NoneSkipFirst,
        );
    }

    #[test]
    #[ignore = "coordinated preparation benchmark; excludes WindowServer conversion"]
    fn framebuffer_format_preparation_cost_at_native_scales() {
        let layer = test_layer();
        for scale in [1, 2] {
            let width = 1280 * scale;
            let height = 900 * scale;
            let source = Arc::new([31, 63, 127, 255].repeat(width as usize * height as usize));
            for format in [Format::Rgba, Format::Bgra] {
                let start = Instant::now();
                for _ in 0..32 {
                    // Includes the BGRA allocation, full copy, conversion and opacity scan.
                    std::hint::black_box(image(&layer, &source, width, height, format));
                }
                eprintln!(
                    "{format:?} scale={scale} physical={width}x{height} preparation_avg={:?}",
                    start.elapsed() / 32,
                );
            }
        }
    }

    #[test]
    #[ignore = "coordinated offscreen CoreAnimation gate; not onscreen pixel evidence"]
    fn framebuffer_rgba_bgra_layer_render_parity() {
        let layer = test_layer();
        let source = Arc::new(vec![
            255, 0, 0, 255, 0, 0, 255, 255, 128, 0, 0, 128, 0, 0, 0, 0,
        ]);
        let a = image(&layer, &source, 4, 1, Format::Rgba);
        let b = image(&layer, &source, 4, 1, Format::Bgra);
        assert_eq!(render(&layer, &a, true), render(&layer, &b, true));
    }

    // A coordinated attached-window harness must capture actual presented pixels
    // outside this timing interval. Do not substitute renderInContext for that gate.
    #[allow(dead_code)]
    fn present_sample(
        layer: &VideoLayer,
        rgba: &Arc<Vec<u8>>,
        width: u32,
        height: u32,
        format: Format,
    ) -> (Duration, Duration, Option<Duration>) {
        let total_start = Instant::now();
        let image = image(layer, rgba, width, height, format);
        CATransaction::begin();
        CATransaction::setDisableActions(true);
        set_layer_frame(&layer.framebuffer_layer, layer.root.bounds());
        let scale = layer.root.contentsScale();
        if layer.framebuffer_layer.contentsScale() != scale {
            layer.framebuffer_layer.setContentsScale(scale);
        }
        let alpha = CGImage::alpha_info(Some(&image));
        let opaque =
            alpha == CGImageAlphaInfo::NoneSkipLast || alpha == CGImageAlphaInfo::NoneSkipFirst;
        if layer.framebuffer_layer.isOpaque() != opaque {
            layer.framebuffer_layer.setOpaque(opaque);
        }
        unsafe {
            layer
                .framebuffer_layer
                .setContents(Some(image.as_ref() as &AnyObject))
        };
        set_layer_hidden(&layer.framebuffer_layer, false);
        let cpu_start = commit_thread_cpu_time();
        let commit_start = Instant::now();
        CATransaction::commit();
        let commit_wall = commit_start.elapsed();
        let cpu = cpu_start
            .zip(commit_thread_cpu_time())
            .and_then(|(a, b)| b.checked_sub(a));
        (total_start.elapsed(), commit_wall, cpu)
    }
}

impl Drop for VideoLayer {
    fn drop(&mut self) {
        self.framebuffer_layer.removeFromSuperlayer();
        self.layer.removeFromSuperlayer();
        self.foreground_layer.removeFromSuperlayer();
    }
}
