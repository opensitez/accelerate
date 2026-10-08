//! Native, thread-bound WGL context for the shared OpenGL renderer.
//!
//! The host must keep its HWND alive until this context has been dropped.
//! `backend-check` can type-check this module elsewhere; no Win32 calls run there.

use crate::gl::GlContext;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::ffi::{CStr, c_char, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;

type Handle = *mut c_void;

const PFD_DOUBLEBUFFER: u32 = 0x00000001;
const PFD_DRAW_TO_WINDOW: u32 = 0x00000004;
const PFD_SUPPORT_OPENGL: u32 = 0x00000020;
const PFD_GENERIC_FORMAT: u32 = 0x00000040;
const PFD_NEED_PALETTE: u32 = 0x00000080;
const PFD_TYPE_RGBA: u8 = 0;
const GL_RENDERER: u32 = 0x1F01;
const GL_VERSION: u32 = 0x1F02;

// Exact wingdi.h ABI: WORD/DWORD remain 16/32 bits on 64-bit Windows.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PixelFormatDescriptor {
    size: u16,
    version: u16,
    flags: u32,
    pixel_type: u8,
    color_bits: u8,
    red_bits: u8,
    red_shift: u8,
    green_bits: u8,
    green_shift: u8,
    blue_bits: u8,
    blue_shift: u8,
    alpha_bits: u8,
    alpha_shift: u8,
    accum_bits: u8,
    accum_red_bits: u8,
    accum_green_bits: u8,
    accum_blue_bits: u8,
    accum_alpha_bits: u8,
    depth_bits: u8,
    stencil_bits: u8,
    aux_buffers: u8,
    layer_type: u8,
    reserved: u8,
    layer_mask: u32,
    visible_mask: u32,
    damage_mask: u32,
}

const _: () = assert!(std::mem::size_of::<PixelFormatDescriptor>() == 40);

#[cfg_attr(target_os = "windows", link(name = "user32"))]
unsafe extern "system" {
    fn GetDC(window: Handle) -> Handle;
    fn ReleaseDC(window: Handle, dc: Handle) -> i32;
}

#[cfg_attr(target_os = "windows", link(name = "gdi32"))]
unsafe extern "system" {
    fn GetPixelFormat(dc: Handle) -> i32;
    fn ChoosePixelFormat(dc: Handle, descriptor: *const PixelFormatDescriptor) -> i32;
    fn DescribePixelFormat(
        dc: Handle,
        format: i32,
        bytes: u32,
        descriptor: *mut PixelFormatDescriptor,
    ) -> i32;
    fn SetPixelFormat(dc: Handle, format: i32, descriptor: *const PixelFormatDescriptor) -> i32;
    fn SwapBuffers(dc: Handle) -> i32;
}

#[cfg_attr(target_os = "windows", link(name = "opengl32"))]
unsafe extern "system" {
    fn wglCreateContext(dc: Handle) -> Handle;
    fn wglMakeCurrent(dc: Handle, context: Handle) -> i32;
    fn wglGetCurrentContext() -> Handle;
    fn wglDeleteContext(context: Handle) -> i32;
    fn wglGetProcAddress(name: *const c_char) -> *const c_void;
    fn glGetString(name: u32) -> *const u8;
}

#[cfg_attr(target_os = "windows", link(name = "kernel32"))]
unsafe extern "system" {
    fn GetModuleHandleW(name: *const u16) -> Handle;
    fn GetProcAddress(module: Handle, name: *const c_char) -> *const c_void;
}

fn hardware_rgba(descriptor: &PixelFormatDescriptor) -> bool {
    let required = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    descriptor.flags & required == required
        && descriptor.flags & (PFD_GENERIC_FORMAT | PFD_NEED_PALETTE) == 0
        && descriptor.pixel_type == PFD_TYPE_RGBA
        && descriptor.layer_type == 0
        && descriptor.color_bits >= 24
        && descriptor.alpha_bits >= 8
        && descriptor.red_bits >= 8
        && descriptor.green_bits >= 8
        && descriptor.blue_bits >= 8
}

unsafe fn describe(dc: Handle, format: i32) -> Option<PixelFormatDescriptor> {
    let mut descriptor = PixelFormatDescriptor::default();
    // SAFETY: descriptor is writable and has the documented wingdi.h layout.
    let count = unsafe {
        DescribePixelFormat(
            dc,
            format,
            std::mem::size_of_val(&descriptor) as u32,
            &mut descriptor,
        )
    };
    (count > 0).then_some(descriptor)
}

unsafe fn select_format(dc: Handle) -> Option<(i32, PixelFormatDescriptor, bool)> {
    // Windows permits setting a window's pixel format only once.
    let existing = unsafe { GetPixelFormat(dc) };
    if existing > 0 {
        let descriptor = unsafe { describe(dc, existing) }?;
        return hardware_rgba(&descriptor).then_some((existing, descriptor, false));
    }
    let requested = PixelFormatDescriptor {
        size: std::mem::size_of::<PixelFormatDescriptor>() as u16,
        version: 1,
        flags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
        pixel_type: PFD_TYPE_RGBA,
        color_bits: 32,
        alpha_bits: 8,
        ..Default::default()
    };
    let preferred = unsafe { ChoosePixelFormat(dc, &requested) };
    if preferred > 0 {
        if let Some(descriptor) = unsafe { describe(dc, preferred) } {
            if hardware_rgba(&descriptor) {
                return Some((preferred, descriptor, true));
            }
        }
    }
    // ChoosePixelFormat is only a closest match; do not accept its software
    // fallback if another advertised format provides hardware RGBA drawing.
    let mut descriptor = PixelFormatDescriptor::default();
    let count = unsafe {
        DescribePixelFormat(
            dc,
            1,
            std::mem::size_of_val(&descriptor) as u32,
            &mut descriptor,
        )
    };
    for format in 1..=count {
        if let Some(descriptor) = unsafe { describe(dc, format) } {
            if hardware_rgba(&descriptor) {
                return Some((format, descriptor, true));
            }
        }
    }
    None
}

fn valid_proc(pointer: *const c_void) -> bool {
    !matches!(pointer as usize, 0 | 1 | 2 | 3 | usize::MAX)
}

fn supports_gl2(version: &[u8]) -> bool {
    let Ok(version) = std::str::from_utf8(version) else {
        return false;
    };
    let Some(version) = version.split_ascii_whitespace().next() else {
        return false;
    };
    let mut components = version.split('.');
    let major = components
        .next()
        .and_then(|value| value.parse::<u32>().ok());
    let minor = components
        .next()
        .and_then(|value| value.parse::<u32>().ok());
    matches!((major, minor), (Some(major), Some(_)) if major >= 2)
}

pub(crate) struct NativeContext {
    window: Handle,
    dc: Handle,
    context: Handle,
    opengl_module: Handle,
    // GetDC/ReleaseDC and current contexts belong to the creating thread.
    _thread_bound: PhantomData<Rc<()>>,
}

impl NativeContext {
    pub(crate) fn new(
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Option<Self> {
        if !cfg!(target_os = "windows") {
            return None;
        }
        let RawWindowHandle::Win32(handle) = window.window_handle().ok()?.as_raw() else {
            return None;
        };
        if !matches!(
            display.display_handle().ok()?.as_raw(),
            RawDisplayHandle::Windows(_)
        ) {
            return None;
        }
        let window = handle.hwnd.get() as Handle;
        // SAFETY: the raw handle provider supplies a live HWND. The host keeps
        // it alive for this context; all subsequent FFI stays on this thread.
        unsafe {
            let dc = GetDC(window);
            if dc.is_null() {
                return None;
            }
            // Install cleanup before any fallible initialization step.
            let mut native = Self {
                window,
                dc,
                context: ptr::null_mut(),
                opengl_module: ptr::null_mut(),
                _thread_bound: PhantomData,
            };
            let (format, descriptor, needs_setting) = select_format(dc)?;
            if needs_setting && SetPixelFormat(dc, format, &descriptor) == 0 {
                return None;
            }
            native.context = wglCreateContext(dc);
            if native.context.is_null() || !native.make_current() {
                return None;
            }
            let version = glGetString(GL_VERSION);
            let renderer = glGetString(GL_RENDERER);
            if version.is_null()
                || renderer.is_null()
                || !supports_gl2(CStr::from_ptr(version.cast()).to_bytes())
                || CStr::from_ptr(renderer.cast())
                    .to_bytes()
                    .eq_ignore_ascii_case(b"GDI Generic")
            {
                return None;
            }
            // opengl32 is already loaded by the static import. Borrow its
            // module handle instead of loading a DLL from a search path.
            let name: Vec<u16> = "opengl32.dll\0".encode_utf16().collect();
            native.opengl_module = GetModuleHandleW(name.as_ptr());
            if native.opengl_module.is_null() {
                return None;
            }
            Some(native)
        }
    }
}

impl GlContext for NativeContext {
    fn make_current(&self) -> bool {
        cfg!(target_os = "windows")
            && unsafe {
                wglGetCurrentContext() == self.context || wglMakeCurrent(self.dc, self.context) != 0
            }
    }

    fn swap_buffers(&self) -> bool {
        self.make_current() && unsafe { SwapBuffers(self.dc) != 0 }
    }

    fn resize(&mut self, _width: u32, _height: u32) {}

    unsafe fn get_proc_address(&self, name: &CStr) -> *const c_void {
        if !self.make_current() {
            return ptr::null();
        }
        let extension = unsafe { wglGetProcAddress(name.as_ptr()) };
        if valid_proc(extension) {
            return extension;
        }
        let exported = unsafe { GetProcAddress(self.opengl_module, name.as_ptr()) };
        if valid_proc(exported) {
            exported
        } else {
            ptr::null()
        }
    }
}

impl Drop for NativeContext {
    fn drop(&mut self) {
        if !cfg!(target_os = "windows") {
            return;
        }
        // Never unbind a different context that the host made current later.
        unsafe {
            if !self.context.is_null() {
                if wglGetCurrentContext() == self.context {
                    wglMakeCurrent(ptr::null_mut(), ptr::null_mut());
                }
                wglDeleteContext(self.context);
            }
            if !self.dc.is_null() {
                ReleaseDC(self.window, self.dc);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wgl_failure_sentinels() {
        for address in [0, 1, 2, 3, usize::MAX] {
            assert!(!valid_proc(address as *const c_void));
        }
        assert!(valid_proc(0x1000_usize as *const c_void));
    }

    #[test]
    fn rejects_generic_and_incompatible_pixel_formats() {
        let mut descriptor = PixelFormatDescriptor {
            flags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
            color_bits: 24,
            alpha_bits: 8,
            red_bits: 8,
            green_bits: 8,
            blue_bits: 8,
            ..Default::default()
        };
        assert!(hardware_rgba(&descriptor));
        descriptor.flags |= PFD_GENERIC_FORMAT;
        assert!(!hardware_rgba(&descriptor));
        descriptor.flags &= !PFD_GENERIC_FORMAT;
        descriptor.alpha_bits = 0;
        assert!(!hardware_rgba(&descriptor));
    }

    #[test]
    fn requires_desktop_opengl_two_or_newer() {
        for version in [b"2.0".as_slice(), b"3.3.0 NVIDIA", b"4.6.0 driver"] {
            assert!(supports_gl2(version));
        }
        for version in [b"1.1".as_slice(), b"", b"2.invalid", b"OpenGL ES 3.0"] {
            assert!(!supports_gl2(version));
        }
    }
}
