//! Shared GPU texture upload, scaling, and premultiplied composition for EGL/WGL.

use crate::{Diagnostics, LayerFrame};
use std::ffi::{CStr, c_char, c_void};
use std::sync::Arc;
use tiny_skia::Pixmap;

pub(crate) trait GlContext {
    fn make_current(&self) -> bool;
    fn swap_buffers(&self) -> bool;
    fn resize(&mut self, width: u32, height: u32);
    unsafe fn get_proc_address(&self, name: &CStr) -> *const c_void;
}

macro_rules! gl_api {
    ($($field:ident: $symbol:literal ($($arg:ty),*) -> $result:ty;)+) => {
        struct Api { $($field: unsafe extern "system" fn($($arg),*) -> $result,)+ }
        impl Api {
            fn load(context: &impl GlContext) -> Option<Self> {
                Some(Self { $($field: {
                    let name = CStr::from_bytes_with_nul(concat!($symbol, "\0").as_bytes()).ok()?;
                    let address = unsafe { context.get_proc_address(name) };
                    if address.is_null() { return None; }
                    unsafe { std::mem::transmute::<*const c_void, unsafe extern "system" fn($($arg),*) -> $result>(address) }
                },)+ })
            }
        }
    };
}

gl_api! {
    get_error: "glGetError" () -> u32;
    get_integer: "glGetIntegerv" (u32, *mut i32) -> ();
    create_shader: "glCreateShader" (u32) -> u32;
    shader_source: "glShaderSource" (u32, i32, *const *const c_char, *const i32) -> ();
    compile_shader: "glCompileShader" (u32) -> ();
    shader_status: "glGetShaderiv" (u32, u32, *mut i32) -> ();
    delete_shader: "glDeleteShader" (u32) -> ();
    create_program: "glCreateProgram" () -> u32;
    attach_shader: "glAttachShader" (u32, u32) -> ();
    bind_attribute: "glBindAttribLocation" (u32, u32, *const c_char) -> ();
    link_program: "glLinkProgram" (u32) -> ();
    program_status: "glGetProgramiv" (u32, u32, *mut i32) -> ();
    delete_program: "glDeleteProgram" (u32) -> ();
    use_program: "glUseProgram" (u32) -> ();
    uniform_location: "glGetUniformLocation" (u32, *const c_char) -> i32;
    uniform_1i: "glUniform1i" (i32, i32) -> ();
    uniform_1f: "glUniform1f" (i32, f32) -> ();
    uniform_2f: "glUniform2f" (i32, f32, f32) -> ();
    uniform_4f: "glUniform4f" (i32, f32, f32, f32, f32) -> ();
    gen_buffers: "glGenBuffers" (i32, *mut u32) -> ();
    delete_buffers: "glDeleteBuffers" (i32, *const u32) -> ();
    bind_buffer: "glBindBuffer" (u32, u32) -> ();
    buffer_data: "glBufferData" (u32, isize, *const c_void, u32) -> ();
    attribute_pointer: "glVertexAttribPointer" (u32, i32, u32, u8, i32, *const c_void) -> ();
    enable_attribute: "glEnableVertexAttribArray" (u32) -> ();
    gen_textures: "glGenTextures" (i32, *mut u32) -> ();
    delete_textures: "glDeleteTextures" (i32, *const u32) -> ();
    active_texture: "glActiveTexture" (u32) -> ();
    bind_texture: "glBindTexture" (u32, u32) -> ();
    texture_parameter: "glTexParameteri" (u32, u32, i32) -> ();
    pixel_store: "glPixelStorei" (u32, i32) -> ();
    texture_image: "glTexImage2D" (u32, i32, i32, i32, i32, i32, u32, u32, *const c_void) -> ();
    texture_sub_image: "glTexSubImage2D" (u32, i32, i32, i32, i32, i32, u32, u32, *const c_void) -> ();
    viewport: "glViewport" (i32, i32, i32, i32) -> ();
    clear_color: "glClearColor" (f32, f32, f32, f32) -> ();
    clear: "glClear" (u32) -> ();
    enable: "glEnable" (u32) -> ();
    disable: "glDisable" (u32) -> ();
    blend_func: "glBlendFunc" (u32, u32) -> ();
    draw_arrays: "glDrawArrays" (u32, i32, i32) -> ();
}

const VERTEX: &str = "#ifdef GL_ES\n#define COORD highp\n#else\n#define COORD\n#endif\n\
attribute vec2 position;\n\
uniform vec2 viewport_size; uniform vec4 destination;\n\
uniform vec4 linear; uniform vec2 translation;\n\
varying COORD vec2 uv; varying COORD vec2 window_position;\n\
void main() { uv = position; vec2 p = destination.xy + position * destination.zw;\n\
p = vec2(linear.x*p.x + linear.z*p.y, linear.y*p.x + linear.w*p.y) + translation;\n\
window_position = p;\n\
gl_Position = vec4(p.x / viewport_size.x * 2.0 - 1.0,\n\
1.0 - p.y / viewport_size.y * 2.0, 0.0, 1.0); }";
const FRAGMENT: &str = "#ifdef GL_ES\nprecision highp float;\n#define COORD highp\n#else\n#define COORD\n#endif\n\
uniform sampler2D image; uniform sampler2D alpha_mask; uniform float masked;\n\
uniform vec4 crop; uniform vec4 tint;\n\
uniform vec2 draw_size; uniform float radius; varying COORD vec2 uv;\n\
uniform float opacity; uniform vec4 clip_rect; uniform float clipped; varying COORD vec2 window_position;\n\
void main() { float coverage = 1.0;\n\
if (clipped > 0.5 && (window_position.x < clip_rect.x || window_position.y < clip_rect.y || window_position.x >= clip_rect.x+clip_rect.z || window_position.y >= clip_rect.y+clip_rect.w)) discard;\n\
if (radius > 0.0) { vec2 p = abs((uv - 0.5) * draw_size);\n\
float d = length(max(p - (draw_size * 0.5 - radius), vec2(0.0))) - radius;\n\
coverage = clamp(0.5 - d, 0.0, 1.0); }\n\
vec4 color = texture2D(image, crop.xy + uv * crop.zw);\n\
if (masked > 0.5) { coverage *= texture2D(alpha_mask, uv).a; }\n\
color.rgb *= mix(vec3(1.0), tint.rgb, tint.a); gl_FragColor = color * coverage * opacity; }";

#[derive(Clone)]
enum Pixels {
    Framebuffer(Arc<Pixmap>),
    Media(Arc<Vec<u8>>),
}
impl Pixels {
    fn data(&self) -> &[u8] {
        match self {
            Self::Framebuffer(p) => p.data(),
            Self::Media(p) => p,
        }
    }
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Framebuffer(a), Self::Framebuffer(b)) => Arc::ptr_eq(a, b),
            (Self::Media(a), Self::Media(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

struct Texture {
    id: u32,
    width: u32,
    height: u32,
    pixels: Option<Pixels>,
    #[cfg(test)]
    uploads: usize,
}
struct Renderer {
    api: Api,
    program: u32,
    buffer: u32,
    textures: Vec<Texture>,
    uniforms: [i32; 14],
    max_texture: u32,
    release_gl: bool,
}

impl Renderer {
    fn new(context: &impl GlContext) -> Option<Self> {
        let api = Api::load(context)?;
        // Only resources owned by this renderer are touched; the host context is dedicated.
        unsafe {
            let compile = |kind, source: &str| {
                let shader = (api.create_shader)(kind);
                if shader == 0 {
                    return None;
                }
                let ptr = source.as_ptr().cast();
                let len = source.len() as i32;
                (api.shader_source)(shader, 1, &ptr, &len);
                (api.compile_shader)(shader);
                let mut status = 0;
                (api.shader_status)(shader, 0x8B81, &mut status);
                if status == 0 {
                    (api.delete_shader)(shader);
                    None
                } else {
                    Some(shader)
                }
            };
            let vertex = compile(0x8B31, VERTEX)?;
            let Some(fragment) = compile(0x8B30, FRAGMENT) else {
                (api.delete_shader)(vertex);
                return None;
            };
            let program = (api.create_program)();
            if program == 0 {
                (api.delete_shader)(vertex);
                (api.delete_shader)(fragment);
                return None;
            }
            (api.attach_shader)(program, vertex);
            (api.attach_shader)(program, fragment);
            (api.bind_attribute)(program, 0, c"position".as_ptr());
            (api.link_program)(program);
            (api.delete_shader)(vertex);
            (api.delete_shader)(fragment);
            let mut status = 0;
            (api.program_status)(program, 0x8B82, &mut status);
            if status == 0 {
                (api.delete_program)(program);
                return None;
            }
            let vertices: [f32; 12] = [0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 1.];
            let mut buffer = 0;
            (api.gen_buffers)(1, &mut buffer);
            (api.bind_buffer)(0x8892, buffer);
            (api.buffer_data)(
                0x8892,
                std::mem::size_of_val(&vertices) as isize,
                vertices.as_ptr().cast(),
                0x88E4,
            );
            let mut ids = [0; 4];
            (api.gen_textures)(4, ids.as_mut_ptr());
            let mut max_texture = 0;
            (api.get_integer)(0x0D33, &mut max_texture);
            let uniforms = [
                c"viewport_size",
                c"destination",
                c"crop",
                c"tint",
                c"draw_size",
                c"radius",
                c"image",
                c"alpha_mask",
                c"masked",
                c"linear",
                c"translation",
                c"opacity",
                c"clip_rect",
                c"clipped",
            ]
            .map(|name| (api.uniform_location)(program, name.as_ptr()));
            let renderer = Self {
                api,
                program,
                buffer,
                uniforms,
                textures: ids
                    .map(|id| Texture {
                        id,
                        width: 0,
                        height: 0,
                        pixels: None,
                        #[cfg(test)]
                        uploads: 0,
                    })
                    .into(),
                max_texture: max_texture.max(0) as u32,
                release_gl: true,
            };
            if buffer == 0
                || ids.contains(&0)
                || uniforms.contains(&-1)
                || renderer.max_texture == 0
                || (renderer.api.get_error)() != 0
            {
                return None;
            }
            Some(renderer)
        }
    }

    fn upload(&mut self, slot: usize, pixels: Pixels, width: u32, height: u32) -> bool {
        if width == 0
            || height == 0
            || width > self.max_texture
            || height > self.max_texture
            || !valid_pixels(pixels.data(), width, height)
        {
            return false;
        }
        let texture = &mut self.textures[slot];
        if texture.width == width
            && texture.height == height
            && texture
                .pixels
                .as_ref()
                .is_some_and(|last| last.same(&pixels))
        {
            return true;
        }
        unsafe {
            (self.api.active_texture)(0x84C0);
            (self.api.bind_texture)(0x0DE1, texture.id);
            (self.api.pixel_store)(0x0CF5, 4);
            (self.api.texture_parameter)(0x0DE1, 0x2801, 0x2601);
            (self.api.texture_parameter)(0x0DE1, 0x2800, 0x2601);
            (self.api.texture_parameter)(0x0DE1, 0x2802, 0x812F);
            (self.api.texture_parameter)(0x0DE1, 0x2803, 0x812F);
            if texture.width != width || texture.height != height {
                (self.api.texture_image)(
                    0x0DE1,
                    0,
                    0x1908,
                    width as i32,
                    height as i32,
                    0,
                    0x1908,
                    0x1401,
                    pixels.data().as_ptr().cast(),
                );
            } else {
                (self.api.texture_sub_image)(
                    0x0DE1,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    0x1908,
                    0x1401,
                    pixels.data().as_ptr().cast(),
                );
            }
            if (self.api.get_error)() != 0 {
                return false;
            }
        }
        texture.width = width;
        texture.height = height;
        texture.pixels = Some(pixels);
        #[cfg(test)]
        {
            texture.uploads += 1;
        }
        true
    }

    fn picture_slots(&mut self, count: usize) -> bool {
        let needed = 4 + count;
        if needed < self.textures.len() {
            let ids: Vec<_> = self.textures[needed..]
                .iter()
                .map(|texture| texture.id)
                .collect();
            unsafe {
                (self.api.delete_textures)(ids.len() as i32, ids.as_ptr());
            }
            self.textures.truncate(needed);
        }
        if needed > self.textures.len() {
            let mut ids = vec![0; needed - self.textures.len()];
            unsafe {
                (self.api.gen_textures)(ids.len() as i32, ids.as_mut_ptr());
            }
            if ids.contains(&0) || unsafe { (self.api.get_error)() } != 0 {
                unsafe {
                    (self.api.delete_textures)(ids.len() as i32, ids.as_ptr());
                }
                return false;
            }
            self.textures.extend(ids.into_iter().map(|id| Texture {
                id,
                width: 0,
                height: 0,
                pixels: None,
                #[cfg(test)]
                uploads: 0,
            }));
        }
        true
    }

    fn begin(&self, width: u32, height: u32) {
        unsafe {
            (self.api.viewport)(0, 0, width as i32, height as i32);
            (self.api.disable)(0x0B71); // Depth test.
            (self.api.disable)(0x0B44); // Face culling.
            (self.api.disable)(0x0C11); // Scissor test.
            (self.api.clear_color)(0., 0., 0., 0.);
            (self.api.clear)(0x4000);
            (self.api.enable)(0x0BE2);
            (self.api.blend_func)(1, 0x0303);
            (self.api.use_program)(self.program);
            (self.api.bind_buffer)(0x8892, self.buffer);
            (self.api.attribute_pointer)(0, 2, 0x1406, 0, 8, std::ptr::null());
            (self.api.enable_attribute)(0);
            (self.api.uniform_2f)(self.uniforms[0], width as f32, height as f32);
            (self.api.uniform_1i)(self.uniforms[6], 0);
            (self.api.uniform_1i)(self.uniforms[7], 1);
            (self.api.active_texture)(0x84C0);
        }
    }

    fn draw(
        &self,
        slot: usize,
        destination: [f32; 4],
        crop: [f32; 4],
        radius: f32,
        tint: Option<[u8; 4]>,
        masked: bool,
    ) {
        self.draw_transformed(
            slot,
            destination,
            crop,
            radius,
            tint,
            masked,
            [1., 0., 0., 1., 0., 0.],
            1.,
            None,
        );
    }

    fn draw_transformed(
        &self,
        slot: usize,
        destination: [f32; 4],
        crop: [f32; 4],
        radius: f32,
        tint: Option<[u8; 4]>,
        masked: bool,
        transform: [f32; 6],
        opacity: f32,
        clip: Option<[f32; 4]>,
    ) {
        let tint = tint.unwrap_or([255, 255, 255, 0]).map(|v| v as f32 / 255.);
        let radius = radius.max(0.).min(destination[2].min(destination[3]) * 0.5);
        unsafe {
            (self.api.active_texture)(0x84C1);
            (self.api.bind_texture)(0x0DE1, self.textures[if masked { 3 } else { slot }].id);
            (self.api.active_texture)(0x84C0);
            (self.api.bind_texture)(0x0DE1, self.textures[slot].id);
            (self.api.uniform_1f)(self.uniforms[8], if masked { 1.0 } else { 0.0 });
            (self.api.uniform_4f)(
                self.uniforms[1],
                destination[0],
                destination[1],
                destination[2],
                destination[3],
            );
            (self.api.uniform_4f)(self.uniforms[2], crop[0], crop[1], crop[2], crop[3]);
            (self.api.uniform_4f)(self.uniforms[3], tint[0], tint[1], tint[2], tint[3]);
            (self.api.uniform_2f)(self.uniforms[4], destination[2], destination[3]);
            (self.api.uniform_1f)(self.uniforms[5], radius);
            (self.api.uniform_4f)(
                self.uniforms[9],
                transform[0],
                transform[1],
                transform[2],
                transform[3],
            );
            (self.api.uniform_2f)(self.uniforms[10], transform[4], transform[5]);
            (self.api.uniform_1f)(self.uniforms[11], opacity);
            let rect = clip.unwrap_or([0.; 4]);
            (self.api.uniform_4f)(self.uniforms[12], rect[0], rect[1], rect[2], rect[3]);
            (self.api.uniform_1f)(self.uniforms[13], if clip.is_some() { 1. } else { 0. });
            (self.api.draw_arrays)(0x0004, 0, 6);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if !self.release_gl {
            return;
        }
        unsafe {
            let ids: Vec<_> = self.textures.iter().map(|texture| texture.id).collect();
            (self.api.delete_textures)(ids.len() as i32, ids.as_ptr());
            (self.api.delete_buffers)(1, &self.buffer);
            (self.api.delete_program)(self.program);
        }
    }
}

pub(crate) struct GlSurface<C: GlContext> {
    // The renderer must release its GL resources before its context is destroyed.
    renderer: Option<Renderer>,
    context: C,
    frame: Option<LayerFrame>,
    crop: Option<[f32; 4]>,
    mask: Option<crate::RgbaSurface>,
    pictures: Vec<crate::PictureLayer>,
    width: u32,
    height: u32,
    scale: f32,
    dirty: bool,
    pub(crate) diagnostics: Diagnostics,
}

impl<C: GlContext> GlSurface<C> {
    pub(crate) fn new(context: C) -> Option<Self> {
        if !context.make_current() {
            return None;
        }
        let renderer = Renderer::new(&context)?;
        Some(Self {
            renderer: Some(renderer),
            context,
            frame: None,
            crop: None,
            mask: None,
            pictures: Vec::new(),
            width: 1,
            height: 1,
            scale: 1.,
            dirty: true,
            diagnostics: Diagnostics::default(),
        })
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32, scale: f32) {
        let width = width.max(1).min(i32::MAX as u32);
        let height = height.max(1).min(i32::MAX as u32);
        let scale = if scale.is_finite() && scale > 0. {
            scale
        } else {
            1.
        };
        if (width, height) != (self.width, self.height) {
            self.context.resize(width, height);
        }
        if (width, height, scale) != (self.width, self.height, self.scale) {
            self.dirty = true;
        }
        (self.width, self.height, self.scale) = (width, height, scale);
    }

    pub(crate) fn present_layer(&mut self, frame: Option<&LayerFrame>) -> bool {
        if self.mask.take().is_some() {
            self.dirty = true;
        }
        if self.crop.take().is_some() {
            self.dirty = true;
        }
        if !self.dirty && same_frame(self.frame.as_ref(), frame) {
            return true;
        }
        self.frame = frame.cloned();
        if self
            .renderer
            .as_ref()
            .is_some_and(|r| r.textures[0].pixels.is_some())
        {
            self.present()
        } else {
            true
        }
    }

    pub(crate) fn present_framebuffer(&mut self, pixels: &Arc<Pixmap>) -> bool {
        self.stage_framebuffer(pixels) && self.present()
    }

    fn stage_framebuffer(&mut self, pixels: &Arc<Pixmap>) -> bool {
        if !self.context.make_current() {
            return false;
        }
        if pixels.width() != self.width || pixels.height() != self.height {
            self.resize(pixels.width(), pixels.height(), self.scale);
        }
        let _span = self.diagnostics.span(crate::Phase::Upload);
        if !self.renderer.as_mut().unwrap().upload(
            0,
            Pixels::Framebuffer(pixels.clone()),
            pixels.width(),
            pixels.height(),
        ) {
            return false;
        }
        drop(_span);
        true
    }

    pub(crate) fn present_pictures(&mut self, pictures: &[crate::PictureLayer]) -> bool {
        self.stage_pictures(pictures) && self.present()
    }

    pub(crate) fn present_framebuffer_with_pictures(
        &mut self,
        pixels: &Arc<Pixmap>,
        pictures: &[crate::PictureLayer],
    ) -> bool {
        if !crate::picture::valid_pictures(pictures, self.scale) {
            return false;
        }
        self.stage_framebuffer(pixels) && self.stage_pictures(pictures) && self.present()
    }

    fn stage_pictures(&mut self, pictures: &[crate::PictureLayer]) -> bool {
        if !crate::picture::valid_pictures(pictures, self.scale) || !self.context.make_current() {
            return false;
        }
        let renderer = self.renderer.as_mut().unwrap();
        if pictures.iter().any(|picture| {
            let (w, h) = picture.surface.dimensions();
            w > renderer.max_texture || h > renderer.max_texture
        }) || !renderer.picture_slots(pictures.len())
        {
            return false;
        }
        for (index, picture) in pictures.iter().enumerate() {
            let (w, h) = picture.surface.dimensions();
            if !renderer.upload(
                4 + index,
                Pixels::Media(picture.surface.pixels().clone()),
                w,
                h,
            ) {
                return false;
            }
        }
        self.pictures = pictures.to_vec();
        self.dirty = true;
        true
    }

    pub(crate) fn present_cropped_layer(
        &mut self,
        layer: Option<&crate::CroppedLayerFrame>,
    ) -> bool {
        self.present_masked_cropped_layer(layer, None)
    }

    pub(crate) fn present_masked_cropped_layer(
        &mut self,
        layer: Option<&crate::CroppedLayerFrame>,
        mask: Option<&crate::RgbaSurface>,
    ) -> bool {
        if layer.is_some_and(|layer| !layer.is_valid()) || !crate::layer::valid_mask(mask) {
            return false;
        }
        self.crop = layer.map(|layer| layer.crop);
        self.frame = layer.map(|layer| layer.frame.clone());
        self.mask = layer.and(mask).cloned();
        self.dirty = true;
        if self
            .renderer
            .as_ref()
            .is_some_and(|renderer| renderer.textures[0].pixels.is_some())
        {
            self.present()
        } else {
            true
        }
    }

    fn present(&mut self) -> bool {
        if !crate::picture::valid_pictures(&self.pictures, self.scale)
            || !self.context.make_current()
        {
            return false;
        }
        let _span = self.diagnostics.span(crate::Phase::LayerPresent);
        let renderer = self.renderer.as_mut().unwrap();
        renderer.begin(self.width, self.height);
        if renderer.textures[0].pixels.is_some() {
            renderer.draw(
                0,
                [0., 0., self.width as f32, self.height as f32],
                [0., 0., 1., 1.],
                0.,
                None,
                false,
            );
        }
        if let Some(frame) = &self.frame {
            if !valid_destination([frame.x, frame.y, frame.width, frame.height])
                || !frame.corner_radius.is_finite()
                || !renderer.upload(
                    1,
                    Pixels::Media(frame.rgba.clone()),
                    frame.source_width,
                    frame.source_height,
                )
            {
                return false;
            }
            let crop = self
                .crop
                .map(|crop| {
                    (
                        crop[0] as f64,
                        crop[1] as f64,
                        crop[2] as f64,
                        crop[3] as f64,
                    )
                })
                .unwrap_or_else(|| frame.contents_rect());
            if let Some(mask) = &self.mask {
                let (width, height) = mask.dimensions();
                if !renderer.upload(3, Pixels::Media(mask.pixels().clone()), width, height) {
                    return false;
                }
            }
            renderer.draw(
                1,
                [frame.x, frame.y, frame.width, frame.height].map(|v| v * self.scale),
                [crop.0 as f32, crop.1 as f32, crop.2 as f32, crop.3 as f32],
                frame.corner_radius * self.scale,
                frame.tint,
                self.mask.is_some(),
            );
            if let Some(foreground) = &frame.foreground {
                if !valid_destination([
                    foreground.x,
                    foreground.y,
                    foreground.width,
                    foreground.height,
                ]) || !renderer.upload(
                    2,
                    Pixels::Media(foreground.rgba.clone()),
                    foreground.source_width,
                    foreground.source_height,
                ) {
                    return false;
                }
                renderer.draw(
                    2,
                    [
                        foreground.x,
                        foreground.y,
                        foreground.width,
                        foreground.height,
                    ]
                    .map(|v| v * self.scale),
                    [0., 0., 1., 1.],
                    0.,
                    None,
                    false,
                );
            } else {
                renderer.textures[2].pixels = None;
            }
        } else {
            renderer.textures[1].pixels = None;
            renderer.textures[2].pixels = None;
        }
        for (index, picture) in self.pictures.iter().enumerate() {
            let (w, h) = picture.surface.dimensions();
            renderer.draw_transformed(
                4 + index,
                [0., 0., w as f32, h as f32],
                [0., 0., 1., 1.],
                0.,
                None,
                false,
                picture.transform.map(|value| value * self.scale),
                picture.opacity,
                picture
                    .clip
                    .map(|rect| rect.map(|value| value * self.scale)),
            );
        }
        if unsafe { (renderer.api.get_error)() } != 0 {
            return false;
        }
        drop(_span);
        let _span = self.diagnostics.span(crate::Phase::Commit);
        let presented = self.context.swap_buffers();
        if presented {
            self.dirty = false;
        }
        presented
    }
}

impl<C: GlContext> Drop for GlSurface<C> {
    fn drop(&mut self) {
        if self.context.make_current() {
            self.renderer.take();
        } else if let Some(renderer) = self.renderer.take() {
            // The driver frees resources with the context; do not issue GL calls without it.
            let mut renderer = renderer;
            renderer.release_gl = false;
        }
    }
}

fn valid_pixels(data: &[u8], width: u32, height: u32) -> bool {
    width > 0
        && height > 0
        && width <= i32::MAX as u32
        && height <= i32::MAX as u32
        && crate::software::rgba_size_matches(data.len(), width, height)
}
fn valid_destination(destination: [f32; 4]) -> bool {
    destination.iter().all(|v| v.is_finite()) && destination[2] > 0. && destination[3] > 0.
}
fn same_frame(a: Option<&LayerFrame>, b: Option<&LayerFrame>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            Arc::ptr_eq(&a.rgba, &b.rgba)
                && (
                    a.source_width,
                    a.source_height,
                    a.presentation_size,
                    a.cover,
                    a.tint,
                ) == (
                    b.source_width,
                    b.source_height,
                    b.presentation_size,
                    b.cover,
                    b.tint,
                )
                && [a.x, a.y, a.width, a.height, a.corner_radius]
                    == [b.x, b.y, b.width, b.height, b.corner_radius]
                && match (&a.foreground, &b.foreground) {
                    (None, None) => true,
                    (Some(a), Some(b)) => {
                        Arc::ptr_eq(&a.rgba, &b.rgba)
                            && (a.source_width, a.source_height)
                                == (b.source_width, b.source_height)
                            && [a.x, a.y, a.width, a.height] == [b.x, b.y, b.width, b.height]
                    }
                    _ => false,
                }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uploads_reject_malformed_dimensions_and_buffers() {
        assert!(valid_pixels(&[0; 16], 2, 2));
        assert!(!valid_pixels(&[0; 15], 2, 2));
        assert!(!valid_pixels(&[], u32::MAX, u32::MAX));
        assert!(!valid_pixels(&[], 0, 0));
        assert!(!valid_destination([0., f32::NAN, 1., 1.]));
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "gl_tests.rs"]
mod gpu_tests;
