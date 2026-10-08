//! Exercise the shared EGL/WGL shader renderer on a real offscreen macOS GL context.
use super::*;
use libloading::Library;
use std::cell::Cell;
use std::ptr;

type Handle = *mut c_void;
type Choose = unsafe extern "C" fn(*const i32, *mut Handle, *mut i32) -> i32;
type Create = unsafe extern "C" fn(Handle, Handle, *mut Handle) -> i32;
type SetCurrent = unsafe extern "C" fn(Handle) -> i32;
type Release = unsafe extern "C" fn(Handle);
type ReadPixels = unsafe extern "system" fn(i32, i32, i32, i32, u32, u32, *mut c_void);

struct Offscreen {
    library: Library,
    context: Handle,
    set_current: SetCurrent,
    release: Release,
    framebuffer: u32,
    color: u32,
    swaps: Cell<u32>,
}

impl Offscreen {
    fn new() -> Self {
        unsafe {
            let library =
                Library::new("/System/Library/Frameworks/OpenGL.framework/OpenGL").unwrap();
            let choose: Choose = *library.get(b"CGLChoosePixelFormat\0").unwrap();
            let create: Create = *library.get(b"CGLCreateContext\0").unwrap();
            let release_format: Release = *library.get(b"CGLReleasePixelFormat\0").unwrap();
            let set_current: SetCurrent = *library.get(b"CGLSetCurrentContext\0").unwrap();
            let release: Release = *library.get(b"CGLReleaseContext\0").unwrap();
            let mut format = ptr::null_mut();
            let mut count = 0;
            assert_eq!(choose([73, 0].as_ptr(), &mut format, &mut count), 0);
            assert!(!format.is_null());
            let mut context = ptr::null_mut();
            let result = create(format, ptr::null_mut(), &mut context);
            release_format(format);
            assert_eq!(result, 0);
            assert!(!context.is_null());
            assert_eq!(set_current(context), 0);
            let mut this = Self {
                library,
                context,
                set_current,
                release,
                framebuffer: 0,
                color: 0,
                swaps: Cell::new(0),
            };
            let api = Api::load(&this).expect("OpenGL 2 renderer API");
            (api.gen_textures)(1, &mut this.color);
            (api.bind_texture)(0x0DE1, this.color);
            (api.texture_image)(0x0DE1, 0, 0x1908, 8, 8, 0, 0x1908, 0x1401, ptr::null());
            let generate: unsafe extern "system" fn(i32, *mut u32) =
                *this.library.get(b"glGenFramebuffersEXT\0").unwrap();
            let bind: unsafe extern "system" fn(u32, u32) =
                *this.library.get(b"glBindFramebufferEXT\0").unwrap();
            let attach: unsafe extern "system" fn(u32, u32, u32, u32, i32) =
                *this.library.get(b"glFramebufferTexture2DEXT\0").unwrap();
            let status: unsafe extern "system" fn(u32) -> u32 =
                *this.library.get(b"glCheckFramebufferStatusEXT\0").unwrap();
            generate(1, &mut this.framebuffer);
            bind(0x8D40, this.framebuffer);
            attach(0x8D40, 0x8CE0, 0x0DE1, this.color, 0);
            assert_eq!(status(0x8D40), 0x8CD5);
            assert_eq!((api.get_error)(), 0);
            this
        }
    }

    fn pixel(&self, x: i32, y: i32) -> [u8; 4] {
        let mut rgba = [0; 4];
        unsafe {
            let read: ReadPixels = *self.library.get(b"glReadPixels\0").unwrap();
            read(x, 7 - y, 1, 1, 0x1908, 0x1401, rgba.as_mut_ptr().cast());
        }
        rgba
    }
}

impl GlContext for Offscreen {
    fn make_current(&self) -> bool {
        unsafe { (self.set_current)(self.context) == 0 }
    }
    fn swap_buffers(&self) -> bool {
        self.swaps.set(self.swaps.get() + 1);
        true
    }
    fn resize(&mut self, _: u32, _: u32) {}
    unsafe fn get_proc_address(&self, name: &CStr) -> *const c_void {
        unsafe { self.library.get::<*const c_void>(name.to_bytes_with_nul()) }
            .map(|symbol| *symbol)
            .unwrap_or(ptr::null())
    }
}

impl Drop for Offscreen {
    fn drop(&mut self) {
        self.make_current();
        unsafe {
            if let Ok(delete) = self
                .library
                .get::<unsafe extern "system" fn(i32, *const u32)>(b"glDeleteFramebuffersEXT\0")
            {
                delete(1, &self.framebuffer);
            }
            if let Ok(delete) = self
                .library
                .get::<unsafe extern "system" fn(i32, *const u32)>(b"glDeleteTextures\0")
            {
                delete(1, &self.color);
            }
            (self.set_current)(ptr::null_mut());
            (self.release)(self.context);
        }
    }
}

fn near(actual: [u8; 4], expected: [u8; 4]) {
    assert!(
        actual.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 3),
        "{actual:?} != {expected:?}"
    );
}

#[test]
#[ignore = "requires a macOS graphics session; shared EGL/WGL ordered-picture readback"]
fn ordered_pictures_affine_opacity_clip_dpr_and_upload_reuse() {
    let mut surface = GlSurface::new(Offscreen::new()).unwrap();
    surface.resize(8, 8, 1.);
    let mut background = Pixmap::new(8, 8).unwrap();
    background.fill(tiny_skia::Color::BLACK);
    let background = Arc::new(background);
    assert!(surface.present_framebuffer(&background));
    let rgba = crate::RgbaSurface::new(Arc::new([255, 0, 0, 255].repeat(4)), 2, 2).unwrap();
    let mut first = crate::PictureLayer {
        surface: rgba,
        transform: [0., 1., -1., 0., 4., 1.],
        opacity: 1.,
        clip: None,
    };
    let mut second = crate::PictureLayer {
        surface: crate::RgbaSurface::new(Arc::new([0, 0, 255, 255].repeat(4)), 2, 2).unwrap(),
        transform: [1., 0., 0., 1., 3., 1.],
        opacity: 0.5,
        clip: Some([3., 1., 1., 2.]),
    };
    let swaps = surface.context.swaps.get();
    assert!(
        surface.present_framebuffer_with_pictures(&background, &[first.clone(), second.clone()])
    );
    assert_eq!(surface.context.swaps.get(), swaps + 1);
    let mut expected = (*background).clone();
    assert!(crate::draw_pictures(
        &mut expected,
        &[first.clone(), second.clone()],
        1.
    ));
    for y in 0..8 {
        for x in 0..8 {
            let pixel = expected.pixel(x, y).unwrap();
            near(
                surface.context.pixel(x as i32, y as i32),
                [pixel.red(), pixel.green(), pixel.blue(), pixel.alpha()],
            );
        }
    }
    let uploads: Vec<_> = surface.renderer.as_ref().unwrap().textures[4..]
        .iter()
        .map(|texture| texture.uploads)
        .collect();
    first.transform[4] += 1.;
    second.opacity = 0.25;
    second.clip = Some([2., 1., 2., 2.]);
    assert!(surface.present_pictures(&[first.clone(), second.clone()]));
    assert_eq!(
        uploads,
        surface.renderer.as_ref().unwrap().textures[4..]
            .iter()
            .map(|texture| texture.uploads)
            .collect::<Vec<_>>()
    );
    surface.resize(8, 8, 2.);
    first.transform = [0.5, 0., 0., 0.5, 1., 1.];
    first.clip = Some([1., 1., 1., 1.]);
    assert!(surface.present_pictures(&[first.clone()]));
    near(surface.context.pixel(2, 2), [255, 0, 0, 255]);
    let swaps = surface.context.swaps.get();
    first.transform[0] = f32::NAN;
    assert!(!surface.present_framebuffer_with_pictures(&background, &[first.clone()]));
    assert_eq!(swaps, surface.context.swaps.get());
    assert!(!surface.present_pictures(&[first]));
    assert_eq!(swaps, surface.context.swaps.get());
    assert!(surface.present_pictures(&[]));
    assert_eq!(surface.renderer.as_ref().unwrap().textures.len(), 4);
    near(surface.context.pixel(2, 2), [0, 0, 0, 255]);
}

#[test]
#[ignore = "requires a macOS graphics session; validates shared GPU shaders by pixel readback"]
fn texture_scaling_orientation_alpha_tint_crop_and_cache() {
    let mut surface = GlSurface::new(Offscreen::new()).expect("shared GPU shader renderer");
    let mut background = Pixmap::new(8, 8).unwrap();
    background.fill(tiny_skia::Color::BLACK);
    assert!(surface.present_framebuffer(&Arc::new(background)));
    let mut frame = LayerFrame {
        rgba: Arc::new(vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ]),
        source_width: 2,
        source_height: 2,
        presentation_size: None,
        x: 0.,
        y: 0.,
        width: 8.,
        height: 8.,
        cover: false,
        corner_radius: 0.,
        tint: None,
        foreground: None,
    };
    assert!(surface.present_layer(Some(&frame)));
    near(surface.context.pixel(0, 0), [255, 0, 0, 255]);
    near(surface.context.pixel(7, 0), [0, 255, 0, 255]);
    near(surface.context.pixel(0, 7), [0, 0, 255, 255]);
    near(surface.context.pixel(7, 7), [255, 255, 255, 255]);
    let swaps = surface.context.swaps.get();
    assert!(surface.present_layer(Some(&frame)));
    assert_eq!(surface.context.swaps.get(), swaps);
    surface.resize(8, 8, 1.);
    assert!(surface.present_layer(Some(&frame)));
    assert_eq!(surface.context.swaps.get(), swaps);
    surface.resize(8, 8, 0.5);
    assert!(surface.present_layer(Some(&frame)));
    near(surface.context.pixel(7, 7), [0, 0, 0, 255]);
    surface.resize(8, 8, 1.);
    assert!(surface.present_layer(Some(&frame)));
    near(surface.context.pixel(7, 7), [255, 255, 255, 255]);
    frame.foreground = Some(crate::ForegroundLayer {
        rgba: Arc::new(vec![0, 0, 128, 128]),
        source_width: 1,
        source_height: 1,
        x: 0.,
        y: 0.,
        width: 2.,
        height: 2.,
    });
    assert!(surface.present_layer(Some(&frame)));
    near(surface.context.pixel(0, 0), [127, 0, 128, 255]);
    frame.foreground = None;
    frame.tint = Some([0, 0, 0, 128]);
    assert!(surface.present_layer(Some(&frame)));
    near(surface.context.pixel(0, 0), [127, 0, 0, 255]);
    frame.tint = None;
    frame.cover = true;
    frame.presentation_size = Some((4, 2));
    assert!(surface.present_layer(Some(&frame)));
    near(surface.context.pixel(0, 0), [239, 16, 0, 255]);
    assert!(surface.present_layer(None));
    near(surface.context.pixel(0, 0), [0, 0, 0, 255]);
}

#[test]
#[ignore = "requires a macOS graphics session; validates shared GPU mask shader by pixel readback"]
fn destination_alpha_mask_crop_reuse_replacement_and_removal() {
    let mut surface = GlSurface::new(Offscreen::new()).expect("shared GPU shader renderer");
    let mut background = Pixmap::new(8, 8).unwrap();
    background.fill(tiny_skia::Color::BLACK);
    let background = Arc::new(background);
    assert!(surface.present_framebuffer(&background));
    let source =
        crate::RgbaSurface::new(Arc::new(vec![255, 0, 0, 255, 0, 0, 255, 255]), 1, 2).unwrap();
    let mask = crate::RgbaSurface::new(
        Arc::new(vec![
            0, 0, 0, 0, 128, 128, 128, 128, 255, 255, 255, 255, 64, 64, 64, 64,
        ]),
        2,
        2,
    )
    .unwrap();
    let mut layer = crate::CroppedLayerFrame {
        frame: source.layer_frame([0.0, 0.0, 8.0, 8.0]),
        crop: [0.0, 0.0, 1.0, 0.5],
    };
    assert!(surface.present_masked_cropped_layer(Some(&layer), Some(&mask)));
    near(surface.context.pixel(0, 0), [0, 0, 0, 255]);
    near(surface.context.pixel(7, 0), [128, 0, 0, 255]);
    near(surface.context.pixel(0, 7), [143, 0, 112, 255]);
    let uploads = surface
        .renderer
        .as_ref()
        .unwrap()
        .textures[..4]
        .iter()
        .map(|texture| texture.uploads)
        .collect::<Vec<_>>();
    layer.crop[1] = 0.5;
    assert!(surface.present_masked_cropped_layer(Some(&layer), Some(&mask.clone())));
    assert_eq!(
        uploads,
        surface
            .renderer
            .as_ref()
            .unwrap()
            .textures[..4]
            .iter()
            .map(|texture| texture.uploads)
            .collect::<Vec<_>>()
    );
    near(surface.context.pixel(7, 7), [0, 0, 64, 255]);
    let replacement = crate::RgbaSurface::new(Arc::new(vec![255; 4]), 1, 1).unwrap();
    assert!(surface.present_masked_cropped_layer(Some(&layer), Some(&replacement)));
    assert_eq!(
        surface.renderer.as_ref().unwrap().textures[1].uploads,
        uploads[1]
    );
    assert_eq!(
        surface.renderer.as_ref().unwrap().textures[3].uploads,
        uploads[3] + 1
    );
    assert!(surface.present_cropped_layer(Some(&layer)));
    near(surface.context.pixel(7, 7), [0, 0, 255, 255]);
    assert!(surface.present_masked_cropped_layer(Some(&layer), Some(&mask)));
    assert!(surface.present_layer(Some(&source.layer_frame([0.0, 0.0, 8.0, 8.0]))));
    near(surface.context.pixel(0, 0), [255, 0, 0, 255]);
    let white = crate::RgbaSurface::new(Arc::new(vec![255; 4]), 1, 1).unwrap();
    let white_layer = crate::CroppedLayerFrame {
        frame: white.layer_frame([0.0, 0.0, 8.0, 8.0]),
        crop: [0.0, 0.0, 1.0, 1.0],
    };
    assert!(surface.present_masked_cropped_layer(Some(&white_layer), Some(&mask)));
    let mut software = (*background).clone();
    assert!(white_layer.draw_software_masked(&mut software, 1.0, Some(&mask)));
    for y in 0..8 {
        for x in 0..8 {
            let pixel = software.pixel(x, y).unwrap();
            near(
                surface.context.pixel(x as i32, y as i32),
                [pixel.red(), pixel.green(), pixel.blue(), pixel.alpha()],
            );
        }
    }
    let swaps = surface.context.swaps.get();
    layer.crop[0] = f32::NAN;
    assert!(!surface.present_masked_cropped_layer(Some(&layer), Some(&mask)));
    assert_eq!(surface.context.swaps.get(), swaps);
}
