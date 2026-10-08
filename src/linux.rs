//! Dynamically loaded EGL/GLES2 for borrowed Xlib, Xcb and Wayland windows.
//!
//! The host window/display must outlive this context. Create, use and drop on
//! the window's owning thread, and call resize before the first Wayland frame.
//! EGL display initialization is shared among this module's contexts only;
//! pre-initialized displays are borrowed and never terminated here. Hosts must
//! not independently terminate a display while these contexts live.
//! ABI references: Khronos EGL 1.5, EXT_platform_base, EXT_platform_x11,
//! EXT_platform_xcb, KHR_platform_wayland; Wayland's wayland-egl-core.h.

use crate::gl::GlContext;
use libloading::Library;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::ffi::{CStr, c_char, c_int, c_uint, c_ulong, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use std::sync::Mutex;

type Handle = *mut c_void;
type Boolean = c_uint;
type Proc = unsafe extern "C" fn();
type GetProcAddress = unsafe extern "C" fn(*const c_char) -> Option<Proc>;
type GetDisplay = unsafe extern "C" fn(Handle) -> Handle;
type Initialize = unsafe extern "C" fn(Handle, *mut c_int, *mut c_int) -> Boolean;
type Terminate = unsafe extern "C" fn(Handle) -> Boolean;
type BindApi = unsafe extern "C" fn(c_uint) -> Boolean;
type QueryApi = unsafe extern "C" fn() -> c_uint;
type QueryString = unsafe extern "C" fn(Handle, c_int) -> *const c_char;
type ChooseConfig =
    unsafe extern "C" fn(Handle, *const c_int, *mut Handle, c_int, *mut c_int) -> Boolean;
type GetConfigAttrib = unsafe extern "C" fn(Handle, Handle, c_int, *mut c_int) -> Boolean;
type CreateContext = unsafe extern "C" fn(Handle, Handle, Handle, *const c_int) -> Handle;
type CreateWindowSurface = unsafe extern "C" fn(Handle, Handle, c_ulong, *const c_int) -> Handle;
type DestroyResource = unsafe extern "C" fn(Handle, Handle) -> Boolean;
type MakeCurrent = unsafe extern "C" fn(Handle, Handle, Handle, Handle) -> Boolean;
type GetCurrentContext = unsafe extern "C" fn() -> Handle;
type GetCurrentDisplay = unsafe extern "C" fn() -> Handle;
type GetCurrentSurface = unsafe extern "C" fn(c_int) -> Handle;
type GlGetString = unsafe extern "C" fn(c_uint) -> *const c_char;
type SwapBuffers = unsafe extern "C" fn(Handle, Handle) -> Boolean;
type SwapInterval = unsafe extern "C" fn(Handle, c_int) -> Boolean;
type GetPlatformDisplay = unsafe extern "C" fn(c_uint, Handle, *const isize) -> Handle;
type GetPlatformDisplayExt = unsafe extern "C" fn(c_uint, Handle, *const c_int) -> Handle;
type CreatePlatformSurface = unsafe extern "C" fn(Handle, Handle, Handle, *const isize) -> Handle;
type CreatePlatformSurfaceExt =
    unsafe extern "C" fn(Handle, Handle, Handle, *const c_int) -> Handle;
type WaylandCreate = unsafe extern "C" fn(Handle, c_int, c_int) -> Handle;
type WaylandResize = unsafe extern "C" fn(Handle, c_int, c_int, c_int, c_int);
type WaylandDestroy = unsafe extern "C" fn(Handle);

const EGL_NONE: c_int = 0x3038;
const EGL_EXTENSIONS: c_int = 0x3055;
const EGL_VERSION: c_int = 0x3054;
const EGL_OPENGL_ES_API: c_uint = 0x30A0;
const EGL_CONTEXT_CLIENT_VERSION: c_int = 0x3098;
const EGL_RED_SIZE: c_int = 0x3024;
const EGL_GREEN_SIZE: c_int = 0x3023;
const EGL_BLUE_SIZE: c_int = 0x3022;
const EGL_ALPHA_SIZE: c_int = 0x3021;
const EGL_NATIVE_VISUAL_ID: c_int = 0x302E;
const EGL_PLATFORM_X11: c_uint = 0x31D5;
const EGL_PLATFORM_X11_SCREEN: c_int = 0x31D6;
const EGL_PLATFORM_XCB: c_uint = 0x31DC;
const EGL_PLATFORM_XCB_SCREEN: c_int = 0x31DE;
const EGL_PLATFORM_WAYLAND: c_uint = 0x31D8;
const CONFIG_ATTRIBUTES: [c_int; 13] = [
    EGL_RED_SIZE,
    8,
    EGL_GREEN_SIZE,
    8,
    EGL_BLUE_SIZE,
    8,
    EGL_ALPHA_SIZE,
    0,
    0x3033,
    0x0004, // EGL_SURFACE_TYPE, EGL_WINDOW_BIT
    0x3040,
    0x0004, // EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT
    EGL_NONE,
];

struct Egl {
    library: Library,
    gles: Option<Library>,
    get_proc_address: GetProcAddress,
    get_display: GetDisplay,
    initialize: Initialize,
    terminate: Terminate,
    bind_api: BindApi,
    query_api: QueryApi,
    query_string: QueryString,
    choose_config: ChooseConfig,
    get_config_attrib: GetConfigAttrib,
    create_context: CreateContext,
    create_window_surface: CreateWindowSurface,
    destroy_context: DestroyResource,
    destroy_surface: DestroyResource,
    make_current: MakeCurrent,
    get_current_context: GetCurrentContext,
    get_current_display: GetCurrentDisplay,
    get_current_surface: GetCurrentSurface,
    swap_buffers: SwapBuffers,
    swap_interval: SwapInterval,
}

fn open_library(names: &[&str]) -> Option<Library> {
    names
        .iter()
        .find_map(|name| unsafe { Library::new(name).ok() })
}

// T must be the function-pointer type declared for this ABI symbol.
unsafe fn symbol<T: Copy>(library: &Library, name: &CStr) -> Option<T> {
    unsafe {
        library
            .get::<T>(name.to_bytes_with_nul())
            .ok()
            .map(|value| *value)
    }
}

impl Egl {
    fn load() -> Option<Self> {
        let library = open_library(&["libEGL.so.1", "libEGL.so"])?;
        // GLES is optional: some EGL implementations resolve all core GL entry
        // points through eglGetProcAddress, while others require the DSO fallback.
        unsafe {
            Some(Self {
                get_proc_address: symbol(&library, c"eglGetProcAddress")?,
                get_display: symbol(&library, c"eglGetDisplay")?,
                initialize: symbol(&library, c"eglInitialize")?,
                terminate: symbol(&library, c"eglTerminate")?,
                bind_api: symbol(&library, c"eglBindAPI")?,
                query_api: symbol(&library, c"eglQueryAPI")?,
                query_string: symbol(&library, c"eglQueryString")?,
                choose_config: symbol(&library, c"eglChooseConfig")?,
                get_config_attrib: symbol(&library, c"eglGetConfigAttrib")?,
                create_context: symbol(&library, c"eglCreateContext")?,
                create_window_surface: symbol(&library, c"eglCreateWindowSurface")?,
                destroy_context: symbol(&library, c"eglDestroyContext")?,
                destroy_surface: symbol(&library, c"eglDestroySurface")?,
                make_current: symbol(&library, c"eglMakeCurrent")?,
                get_current_context: symbol(&library, c"eglGetCurrentContext")?,
                get_current_display: symbol(&library, c"eglGetCurrentDisplay")?,
                get_current_surface: symbol(&library, c"eglGetCurrentSurface")?,
                swap_buffers: symbol(&library, c"eglSwapBuffers")?,
                swap_interval: symbol(&library, c"eglSwapInterval")?,
                gles: open_library(&["libGLESv2.so.2", "libGLESv2.so"]),
                library,
            })
        }
    }

    unsafe fn extension_symbol<T: Copy>(&self, name: &CStr) -> Option<T> {
        if let Some(value) = unsafe { symbol(&self.library, name) } {
            return Some(value);
        }
        if std::mem::size_of::<T>() != std::mem::size_of::<Proc>() {
            return None;
        }
        let proc = unsafe { (self.get_proc_address)(name.as_ptr()) }?;
        Some(unsafe { std::mem::transmute_copy(&proc) })
    }

    fn client_extension(&self, name: &str) -> bool {
        let extensions = unsafe { (self.query_string)(ptr::null_mut(), EGL_EXTENSIONS) };
        if extensions.is_null() {
            return false;
        }
        let bytes = unsafe { CStr::from_ptr(extensions) }.to_bytes();
        extension_present(bytes, name)
    }

    fn platform_display(
        &self,
        platform: c_uint,
        native: Handle,
        screen: Option<(c_int, c_int)>,
        khr_extension: &str,
        ext_extension: &str,
    ) -> Option<(Handle, PlatformApi)> {
        let khr = !khr_extension.is_empty() && self.client_extension(khr_extension);
        let ext =
            self.client_extension(ext_extension) && self.client_extension("EGL_EXT_platform_base");
        if !khr && !ext {
            return None;
        }
        // Core attributes are EGLAttrib (pointer-sized), EXT attributes EGLint.
        // Exported core functions are tried first, including EXT's Xcb platform.
        if let Some(api) = PlatformApi::core(self) {
            let display = api.display(platform, native, screen);
            if !display.is_null() {
                return Some((display, api));
            }
        }
        if ext {
            let api = PlatformApi::ext(self)?;
            let display = api.display(platform, native, screen);
            if !display.is_null() {
                return Some((display, api));
            }
        }
        None
    }

    fn configs(&self, display: Handle) -> Option<Vec<Handle>> {
        let mut count = 0;
        if unsafe {
            (self.choose_config)(
                display,
                CONFIG_ATTRIBUTES.as_ptr(),
                ptr::null_mut(),
                0,
                &mut count,
            )
        } == 0
            || !(1..=4096).contains(&count)
        {
            return None;
        }
        let mut configs = Vec::new();
        configs.try_reserve_exact(count as usize).ok()?;
        configs.resize(count as usize, ptr::null_mut());
        let mut returned = 0;
        if unsafe {
            (self.choose_config)(
                display,
                CONFIG_ATTRIBUTES.as_ptr(),
                configs.as_mut_ptr(),
                count,
                &mut returned,
            )
        } == 0
            || returned <= 0
            || returned > count
        {
            return None;
        }
        configs.truncate(returned as usize);
        // Texture alpha and premultiplied blending do not require an alpha
        // channel in an opaque window framebuffer. Prefer alpha8 without
        // excluding alpha0 configurations for ordinary 24-bit X11 visuals.
        configs.sort_by_key(|config| {
            self.config_attribute(display, *config, EGL_ALPHA_SIZE) != Some(8)
        });
        Some(configs)
    }

    fn config_attribute(&self, display: Handle, config: Handle, attribute: c_int) -> Option<c_int> {
        let mut value = 0;
        (unsafe { (self.get_config_attrib)(display, config, attribute, &mut value) } != 0)
            .then_some(value)
    }

    fn config_matches(&self, display: Handle, config: Handle, visual: Option<c_ulong>) -> bool {
        if config.is_null() {
            return false;
        }
        for attribute in [EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE] {
            if self.config_attribute(display, config, attribute) != Some(8) {
                return false;
            }
        }
        if let Some(visual) = visual {
            if self
                .config_attribute(display, config, EGL_NATIVE_VISUAL_ID)
                .is_none_or(|value| (value as u32 as c_ulong) != visual)
            {
                return false;
            }
        }
        true
    }
}

fn extension_present(extensions: &[u8], name: &str) -> bool {
    !name.is_empty()
        && extensions
            .split(|byte| byte.is_ascii_whitespace())
            .any(|extension| extension == name.as_bytes())
}

#[derive(Clone, Copy)]
enum PlatformApi {
    Core {
        display: GetPlatformDisplay,
        surface: CreatePlatformSurface,
    },
    Ext {
        display: GetPlatformDisplayExt,
        surface: CreatePlatformSurfaceExt,
    },
}

impl PlatformApi {
    fn core(egl: &Egl) -> Option<Self> {
        unsafe {
            Some(Self::Core {
                display: symbol(&egl.library, c"eglGetPlatformDisplay")?,
                surface: symbol(&egl.library, c"eglCreatePlatformWindowSurface")?,
            })
        }
    }

    fn ext(egl: &Egl) -> Option<Self> {
        unsafe {
            Some(Self::Ext {
                display: egl.extension_symbol(c"eglGetPlatformDisplayEXT")?,
                surface: egl.extension_symbol(c"eglCreatePlatformWindowSurfaceEXT")?,
            })
        }
    }

    fn display(self, platform: c_uint, native: Handle, screen: Option<(c_int, c_int)>) -> Handle {
        let (attribute, value) = screen.unwrap_or((EGL_NONE, 0));
        unsafe {
            match self {
                Self::Core { display, .. } => display(
                    platform,
                    native,
                    [attribute as isize, value as isize, EGL_NONE as isize].as_ptr(),
                ),
                Self::Ext { display, .. } => {
                    display(platform, native, [attribute, value, EGL_NONE].as_ptr())
                }
            }
        }
    }

    unsafe fn surface(self, display: Handle, config: Handle, native: Handle) -> Handle {
        unsafe {
            match self {
                Self::Core { surface, .. } => surface(display, config, native, ptr::null()),
                Self::Ext { surface, .. } => surface(display, config, native, ptr::null()),
            }
        }
    }
}

struct WaylandWindow {
    handle: Handle,
    resize: WaylandResize,
    destroy: WaylandDestroy,
    _library: Library,
}

impl WaylandWindow {
    fn new(surface: Handle) -> Option<Self> {
        let library = open_library(&["libwayland-egl.so.1", "libwayland-egl.so"])?;
        unsafe {
            let create: WaylandCreate = symbol(&library, c"wl_egl_window_create")?;
            let resize = symbol(&library, c"wl_egl_window_resize")?;
            let destroy = symbol(&library, c"wl_egl_window_destroy")?;
            let handle = create(surface, 1, 1);
            if handle.is_null() {
                return None;
            }
            Some(Self {
                handle,
                resize,
                destroy,
                _library: library,
            })
        }
    }
}

impl Drop for WaylandWindow {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.handle) };
    }
}

type DisplayKey = (usize, usize);
struct DisplayEntry {
    key: DisplayKey,
    users: usize,
    owned: bool,
}
static DISPLAYS: Mutex<Vec<DisplayEntry>> = Mutex::new(Vec::new());

fn retain_display(egl: &Egl, display: Handle) -> Option<DisplayKey> {
    if display.is_null() {
        return None;
    }
    let key = (display as usize, egl.initialize as usize);
    let mut displays = DISPLAYS.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(entry) = displays.iter_mut().find(|entry| entry.key == key) {
        entry.users = entry.users.checked_add(1)?;
        return Some(key);
    }
    displays.try_reserve(1).ok()?;
    let owned = unsafe { (egl.query_string)(display, EGL_VERSION) }.is_null();
    let (mut major, mut minor) = (0, 0);
    if unsafe { (egl.initialize)(display, &mut major, &mut minor) } == 0 {
        return None;
    }
    if (major, minor) < (1, 3) {
        if owned {
            unsafe { (egl.terminate)(display) };
        }
        return None;
    }
    displays.push(DisplayEntry {
        key,
        users: 1,
        owned,
    });
    Some(key)
}

fn release_display(egl: &Egl, display: Handle, key: DisplayKey) {
    let mut displays = DISPLAYS.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(index) = displays.iter().position(|entry| entry.key == key) {
        let entry = &mut displays[index];
        if entry.users > 1 {
            entry.users -= 1;
        } else {
            // Serialize termination against initialization on other threads.
            if entry.owned {
                unsafe { (egl.terminate)(display) };
            }
            displays.swap_remove(index);
        }
    }
}

struct ApiBinding {
    bind: BindApi,
    make_current: MakeCurrent,
    previous: c_uint,
    display: Handle,
    context: Handle,
    draw: Handle,
    read: Handle,
    restore: bool,
}

impl Drop for ApiBinding {
    fn drop(&mut self) {
        if self.restore {
            unsafe {
                if (self.bind)(self.previous) != 0 && !self.context.is_null() {
                    (self.make_current)(self.display, self.draw, self.read, self.context);
                }
            }
        }
    }
}

pub(crate) struct NativeContext {
    egl: Egl,
    display: Handle,
    display_key: DisplayKey,
    context: Handle,
    surface: Handle,
    wayland: Option<WaylandWindow>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl NativeContext {
    pub(crate) fn new(
        window: &impl HasWindowHandle,
        display: &impl HasDisplayHandle,
    ) -> Option<Self> {
        let window = window.window_handle().ok()?;
        let display = display.display_handle().ok()?;
        let egl = Egl::load()?;
        let (display, platform, visual, wayland) = match (window.as_raw(), display.as_raw()) {
            (RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display))
                if window.window != 0 && display.screen >= 0 =>
            {
                let native = display
                    .display
                    .map_or(ptr::null_mut(), |display| display.as_ptr());
                let platform = egl.platform_display(
                    EGL_PLATFORM_X11,
                    native,
                    Some((EGL_PLATFORM_X11_SCREEN, display.screen)),
                    "EGL_KHR_platform_x11",
                    "EGL_EXT_platform_x11",
                );
                let (display, api) = if let Some((display, api)) = platform {
                    (display, Some(api))
                } else if display.screen == 0 {
                    (unsafe { (egl.get_display)(native) }, None)
                } else {
                    return None;
                };
                (
                    display,
                    api,
                    (window.visual_id != 0).then_some(window.visual_id),
                    None,
                )
            }
            (RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display))
                if display.screen >= 0 =>
            {
                let native = display
                    .connection
                    .map_or(ptr::null_mut(), |connection| connection.as_ptr());
                let (display, api) = egl.platform_display(
                    EGL_PLATFORM_XCB,
                    native,
                    Some((EGL_PLATFORM_XCB_SCREEN, display.screen)),
                    "",
                    "EGL_EXT_platform_xcb",
                )?;
                (
                    display,
                    Some(api),
                    window.visual_id.map(|visual| visual.get() as c_ulong),
                    None,
                )
            }
            (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
                let (display, api) = egl.platform_display(
                    EGL_PLATFORM_WAYLAND,
                    display.display.as_ptr(),
                    None,
                    "EGL_KHR_platform_wayland",
                    "EGL_EXT_platform_wayland",
                )?;
                (
                    display,
                    Some(api),
                    None,
                    Some(WaylandWindow::new(window.surface.as_ptr())?),
                )
            }
            _ => return None,
        };
        let display_key = retain_display(&egl, display)?;
        let mut this = Self {
            egl,
            display,
            display_key,
            context: ptr::null_mut(),
            surface: ptr::null_mut(),
            wayland,
            _thread_bound: PhantomData,
        };
        let mut binding = ApiBinding {
            bind: this.egl.bind_api,
            make_current: this.egl.make_current,
            previous: unsafe { (this.egl.query_api)() },
            display: unsafe { (this.egl.get_current_display)() },
            context: unsafe { (this.egl.get_current_context)() },
            draw: unsafe { (this.egl.get_current_surface)(0x3059) }, // EGL_DRAW
            read: unsafe { (this.egl.get_current_surface)(0x305A) }, // EGL_READ
            restore: true,
        };
        if unsafe { (this.egl.bind_api)(EGL_OPENGL_ES_API) } == 0 {
            return None;
        }
        let configs = this.egl.configs(display)?;
        for config in configs {
            if !this.egl.config_matches(display, config, visual) {
                continue;
            }
            // Platform APIs take addresses of X11 integer IDs, not casts of IDs
            // to pointers. Legacy Xlib instead takes the native XID by value.
            let surface = unsafe {
                match window.as_raw() {
                    RawWindowHandle::Xlib(window) => match platform {
                        Some(api) => {
                            let mut xid = window.window;
                            api.surface(display, config, (&mut xid as *mut c_ulong).cast())
                        }
                        None => (this.egl.create_window_surface)(
                            display,
                            config,
                            window.window,
                            ptr::null(),
                        ),
                    },
                    RawWindowHandle::Xcb(window) => {
                        let mut xid = window.window.get();
                        platform?.surface(display, config, (&mut xid as *mut u32).cast())
                    }
                    RawWindowHandle::Wayland(_) => {
                        platform?.surface(display, config, this.wayland.as_ref()?.handle)
                    }
                    _ => return None,
                }
            };
            if surface.is_null() {
                continue;
            }
            this.surface = surface;
            this.context = unsafe {
                (this.egl.create_context)(
                    display,
                    config,
                    ptr::null_mut(),
                    [EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE].as_ptr(),
                )
            };
            if !this.context.is_null() && this.make_current() {
                // EGL availability alone does not imply GPU acceleration.
                if !this.accelerated_renderer() {
                    return None;
                }
                // Interval zero is best-effort: implementations may clamp to
                // EGL_MIN_SWAP_INTERVAL, so the host still owns frame pacing.
                unsafe { (this.egl.swap_interval)(display, 0) };
                binding.restore = false;
                drop(binding);
                return Some(this);
            }
            this.destroy_resources();
        }
        None
    }

    fn accelerated_renderer(&self) -> bool {
        let address = unsafe { self.get_proc_address(c"glGetString") };
        if address.is_null() {
            return false;
        }
        let get_string: GlGetString = unsafe { std::mem::transmute(address) };
        let renderer = unsafe { get_string(0x1F01) }; // GL_RENDERER
        if renderer.is_null() {
            return false;
        }
        let renderer = unsafe { CStr::from_ptr(renderer) }.to_bytes();
        !renderer.is_empty() && !software_renderer(renderer)
    }

    fn destroy_resources(&mut self) {
        unsafe {
            if !self.context.is_null() {
                if (self.egl.get_current_context)() == self.context {
                    (self.egl.make_current)(
                        self.display,
                        ptr::null_mut(),
                        ptr::null_mut(),
                        ptr::null_mut(),
                    );
                }
                (self.egl.destroy_context)(self.display, self.context);
                self.context = ptr::null_mut();
            }
            if !self.surface.is_null() {
                (self.egl.destroy_surface)(self.display, self.surface);
                self.surface = ptr::null_mut();
            }
        }
    }
}

impl GlContext for NativeContext {
    fn make_current(&self) -> bool {
        unsafe {
            if self.context.is_null() || self.surface.is_null() {
                return false;
            }
            if (self.egl.query_api)() == EGL_OPENGL_ES_API
                && (self.egl.get_current_context)() == self.context
            {
                return true;
            }
            (self.egl.bind_api)(EGL_OPENGL_ES_API) != 0
                && (self.egl.make_current)(self.display, self.surface, self.surface, self.context)
                    != 0
        }
    }

    fn swap_buffers(&self) -> bool {
        self.make_current() && unsafe { (self.egl.swap_buffers)(self.display, self.surface) != 0 }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if let Some(window) = &self.wayland {
            unsafe {
                (window.resize)(
                    window.handle,
                    wayland_dimension(width),
                    wayland_dimension(height),
                    0,
                    0,
                )
            };
        }
        // X11 EGL window surfaces follow native window resize; viewport updates
        // are performed by the parent's shared GL renderer for all platforms.
    }

    unsafe fn get_proc_address(&self, name: &CStr) -> *const c_void {
        if let Some(proc) = unsafe { (self.egl.get_proc_address)(name.as_ptr()) } {
            return proc as *const () as *const c_void;
        }
        self.egl
            .gles
            .as_ref()
            .and_then(|library| unsafe { symbol::<Proc>(library, name) })
            .map_or(ptr::null(), |proc| proc as *const () as *const c_void)
    }
}

fn wayland_dimension(value: u32) -> c_int {
    value.clamp(1, c_int::MAX as u32) as c_int
}

// Renderer strings are driver descriptions, not a hardware attestation. Reject
// known CPU implementations, including software Vulkan behind GLES translation.
fn software_renderer(renderer: &[u8]) -> bool {
    [
        b"llvmpipe".as_slice(),
        b"softpipe",
        b"swrast",
        b"software rasterizer",
        b"software renderer",
        b"swiftshader",
        b"lavapipe",
        b"mesa x11",
        b"swr (",
    ]
    .iter()
    .any(|marker| {
        renderer
            .windows(marker.len())
            .any(|part| part.eq_ignore_ascii_case(marker))
    })
}

impl Drop for NativeContext {
    fn drop(&mut self) {
        // Do not detach a sibling's current context, or the host's desktop GL
        // binding. Function pointers remain backed by owned DSOs throughout.
        let previous = unsafe { (self.egl.query_api)() };
        unsafe { (self.egl.bind_api)(EGL_OPENGL_ES_API) };
        self.destroy_resources();
        // EGL must release the native wrapper before libwayland-egl destroys it.
        release_display(&self.egl, self.display, self.display_key);
        drop(self.wayland.take());
        unsafe { (self.egl.bind_api)(previous) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    mod lifecycle {
        use super::*;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        static BORROWED: AtomicBool = AtomicBool::new(false);
        static INITIALIZES: AtomicUsize = AtomicUsize::new(0);
        static TERMINATES: AtomicUsize = AtomicUsize::new(0);
        static CURRENT: AtomicUsize = AtomicUsize::new(0);
        static DETACHES: AtomicUsize = AtomicUsize::new(0);
        static BINDS: AtomicUsize = AtomicUsize::new(0);
        static MAKES_CURRENT: AtomicUsize = AtomicUsize::new(0);
        static CONTEXTS_DESTROYED: AtomicUsize = AtomicUsize::new(0);
        static SURFACES_DESTROYED: AtomicUsize = AtomicUsize::new(0);

        unsafe extern "C" fn get_proc(_: *const c_char) -> Option<Proc> {
            None
        }
        unsafe extern "C" fn get_display(native: Handle) -> Handle {
            native
        }
        unsafe extern "C" fn initialize(
            _: Handle,
            major: *mut c_int,
            minor: *mut c_int,
        ) -> Boolean {
            INITIALIZES.fetch_add(1, Ordering::SeqCst);
            unsafe {
                *major = 1;
                *minor = 5;
            }
            1
        }
        unsafe extern "C" fn terminate(_: Handle) -> Boolean {
            TERMINATES.fetch_add(1, Ordering::SeqCst);
            1
        }
        unsafe extern "C" fn bind(_: c_uint) -> Boolean {
            BINDS.fetch_add(1, Ordering::SeqCst);
            1
        }
        unsafe extern "C" fn query_api() -> c_uint {
            EGL_OPENGL_ES_API
        }
        unsafe extern "C" fn query_string(_: Handle, name: c_int) -> *const c_char {
            if name == EGL_VERSION && BORROWED.load(Ordering::SeqCst) {
                c"1.5".as_ptr()
            } else {
                ptr::null()
            }
        }
        unsafe extern "C" fn choose(
            _: Handle,
            _: *const c_int,
            _: *mut Handle,
            _: c_int,
            _: *mut c_int,
        ) -> Boolean {
            0
        }
        unsafe extern "C" fn config(
            _: Handle,
            config: Handle,
            name: c_int,
            value: *mut c_int,
        ) -> Boolean {
            let result = match name {
                EGL_RED_SIZE | EGL_GREEN_SIZE | EGL_BLUE_SIZE => {
                    if config as usize == 13 {
                        5
                    } else {
                        8
                    }
                }
                EGL_ALPHA_SIZE => {
                    if config as usize == 11 {
                        0
                    } else {
                        8
                    }
                }
                EGL_NATIVE_VISUAL_ID => {
                    if config as usize == 11 {
                        44
                    } else {
                        45
                    }
                }
                _ => return 0,
            };
            unsafe {
                *value = result;
            }
            1
        }
        unsafe extern "C" fn create_context(
            _: Handle,
            _: Handle,
            _: Handle,
            _: *const c_int,
        ) -> Handle {
            ptr::null_mut()
        }
        unsafe extern "C" fn create_surface(
            _: Handle,
            _: Handle,
            _: c_ulong,
            _: *const c_int,
        ) -> Handle {
            ptr::null_mut()
        }
        unsafe extern "C" fn destroy_context(_: Handle, _: Handle) -> Boolean {
            CONTEXTS_DESTROYED.fetch_add(1, Ordering::SeqCst);
            1
        }
        unsafe extern "C" fn destroy_surface(_: Handle, _: Handle) -> Boolean {
            SURFACES_DESTROYED.fetch_add(1, Ordering::SeqCst);
            1
        }
        unsafe extern "C" fn make_current(
            _: Handle,
            _: Handle,
            _: Handle,
            context: Handle,
        ) -> Boolean {
            MAKES_CURRENT.fetch_add(1, Ordering::SeqCst);
            CURRENT.store(context as usize, Ordering::SeqCst);
            if context.is_null() {
                DETACHES.fetch_add(1, Ordering::SeqCst);
            }
            1
        }
        unsafe extern "C" fn current_context() -> Handle {
            CURRENT.load(Ordering::SeqCst) as Handle
        }
        unsafe extern "C" fn current_display() -> Handle {
            42usize as Handle
        }
        unsafe extern "C" fn current_surface(_: c_int) -> Handle {
            ptr::null_mut()
        }
        unsafe extern "C" fn swap(_: Handle, _: Handle) -> Boolean {
            1
        }
        unsafe extern "C" fn interval(_: Handle, _: c_int) -> Boolean {
            1
        }

        fn egl() -> Egl {
            // Own a real loader handle, but all EGL calls go to typed test stubs.
            Egl {
                library: libloading::os::unix::Library::this().into(),
                gles: None,
                get_proc_address: get_proc,
                get_display,
                initialize,
                terminate,
                bind_api: bind,
                query_api,
                query_string,
                choose_config: choose,
                get_config_attrib: config,
                create_context,
                create_window_surface: create_surface,
                destroy_context,
                destroy_surface,
                make_current,
                get_current_context: current_context,
                get_current_display: current_display,
                get_current_surface: current_surface,
                swap_buffers: swap,
                swap_interval: interval,
            }
        }

        fn context(id: usize) -> NativeContext {
            let egl = egl();
            let display = 42usize as Handle;
            let display_key = retain_display(&egl, display).unwrap();
            NativeContext {
                egl,
                display,
                display_key,
                context: id as Handle,
                surface: id as Handle,
                wayland: None,
                _thread_bound: PhantomData,
            }
        }

        #[test]
        fn sibling_and_borrowed_displays_survive_until_their_owner_releases() {
            for borrowed in [false, true] {
                BORROWED.store(borrowed, Ordering::SeqCst);
                for count in [
                    &INITIALIZES,
                    &TERMINATES,
                    &DETACHES,
                    &BINDS,
                    &MAKES_CURRENT,
                    &CONTEXTS_DESTROYED,
                    &SURFACES_DESTROYED,
                ] {
                    count.store(0, Ordering::SeqCst);
                }
                let first = context(1);
                let second = context(2);
                CURRENT.store(2, Ordering::SeqCst);
                assert_eq!(INITIALIZES.load(Ordering::SeqCst), 1);
                assert!(second.make_current());
                assert_eq!(BINDS.load(Ordering::SeqCst), 0);
                assert_eq!(MAKES_CURRENT.load(Ordering::SeqCst), 0);
                drop(first);
                assert_eq!(TERMINATES.load(Ordering::SeqCst), 0);
                assert_eq!(CURRENT.load(Ordering::SeqCst), 2);
                assert_eq!(DETACHES.load(Ordering::SeqCst), 0);
                drop(second);
                assert_eq!(TERMINATES.load(Ordering::SeqCst), usize::from(!borrowed));
                assert_eq!(CURRENT.load(Ordering::SeqCst), 0);
                assert_eq!(DETACHES.load(Ordering::SeqCst), 1);
                assert_eq!(CONTEXTS_DESTROYED.load(Ordering::SeqCst), 2);
                assert_eq!(SURFACES_DESTROYED.load(Ordering::SeqCst), 2);
            }
        }

        #[test]
        fn opaque_rgb8_config_requires_matching_native_visual() {
            let egl = egl();
            let display = 42usize as Handle;
            let opaque = 11usize as Handle;
            assert_eq!(
                egl.config_attribute(display, opaque, EGL_ALPHA_SIZE),
                Some(0)
            );
            assert!(egl.config_matches(display, opaque, Some(44)));
            assert!(!egl.config_matches(display, opaque, Some(45)));
            assert!(!egl.config_matches(display, 12usize as Handle, Some(44)));
            assert!(!egl.config_matches(display, 13usize as Handle, None));
        }
    }

    #[test]
    fn extension_names_are_whole_tokens() {
        let extensions =
            b"EGL_EXT_platform_base EGL_EXT_platform_xcb_extra EGL_KHR_platform_wayland";
        assert!(extension_present(extensions, "EGL_EXT_platform_base"));
        assert!(!extension_present(extensions, "EGL_EXT_platform_xcb"));
        assert!(!extension_present(extensions, ""));
        assert!(!extension_present(b"", ""));
        assert!(extension_present(extensions, "EGL_KHR_platform_wayland"));
    }

    #[test]
    fn wayland_sizes_are_positive_c_ints() {
        assert_eq!(wayland_dimension(0), 1);
        assert_eq!(wayland_dimension(1920), 1920);
        assert_eq!(wayland_dimension(u32::MAX), c_int::MAX);
    }

    #[test]
    fn known_cpu_renderers_are_not_hardware_backends() {
        for renderer in [
            "llvmpipe (LLVM 18.0, 256 bits)",
            "SOFTPIPE",
            "Mesa swrast",
            "Software Rasterizer",
            "Mesa X11",
            "SWR (Intel OpenSWR)",
            "ANGLE (Google, Vulkan SwiftShader Device)",
            "zink (Vulkan lavapipe)",
        ] {
            assert!(software_renderer(renderer.as_bytes()), "{renderer}");
        }
        for renderer in [
            "Mesa Intel(R) UHD Graphics 630",
            "AMD Radeon RX 6600",
            "NVIDIA GeForce RTX 3070",
        ] {
            assert!(!software_renderer(renderer.as_bytes()), "{renderer}");
        }
    }
}
