# Web Core - Accelerate

This crate is for centralizing gpu acceleration for Web Core Browser Engine
Shared surface presentation and software acceleration for renderers and media
applications. The crate has no browser, DOM, layout, codec, winit, or wgpu
dependency.

## Current Backends

- macOS: retained Core Animation layers for framebuffers, media, and foreground
  overlays. This is native presentation, not GPU rasterization or a Metal compute
  implementation.
- Windows: hardware WGL/OpenGL presentation on the host HWND.
- Linux: EGL/OpenGL ES presentation for X11 (Xlib/Xcb) and Wayland.
- Fourth backend, software: premultiplied RGBA drawing with clipped bulk row copies for
  opaque integer-aligned surfaces, tiny-skia fallback for transforms/blending,
  and ARM64 NEON alpha classification. It remains available on all platforms.

Windows and Linux share one texture renderer: cached RGBA uploads, linear GPU
scaling, cover cropping, rounded media clips, premultiplied blending, tinting,
and foreground overlays. High-precision shader coordinates preserve individual
pixels in large/HiDPI framebuffers; GLES drivers without fragment high precision
use the software fallback. Contexts, shaders, textures, and window/display handles
are validated before selecting a native backend; initialization or presentation
failure falls back to software. Windows rejects generic GDI OpenGL, and Linux
rejects known software Mesa renderers rather than reporting them as hardware.

`WindowSurface` accepts an owned `Arc` of a `raw-window-handle` host window/display,
retaining it until native layers are removed. Native initialization must occur on
the window's UI thread; unsupported or unavailable native presentation reports
`Backend::Software`. A failed framebuffer presentation disables its native
backend. Call `resize(physical_width, physical_height, scale_factor)` when the
host size or display scale changes. A `false` presentation result tells the host to present through its own
software window presenter; accelerate does not create windows.

`Preference::Software` forces fallback. `ACCELERATE_DISABLE_GPU` disables native
presentation. Optional `Diagnostics` callbacks preserve host profiling without
coupling the crate to a particular logger or application.

## Software Drawing

`RgbaSurface` is the shared decoder/compositor boundary. It validates dimensions
and retains premultiplied RGBA through an `Arc` without copying pixels. Use
`draw()` for software composition, `layer_frame()` for configurable native media
composition, or `WindowSurface::present_surface()` for a simple native layer.
The software pixel primitives also include bit-exact RGBA premultiplication,
using NEON batches on ARM64 and a portable scalar path elsewhere.

`fonts::blit_mask` blends cached grayscale glyphs in clipped row spans without
per-pixel renderer callbacks. Explicit `BlendRounding` preserves the existing
Canvas and browser edge-blending conventions. Shaping, font-container decoding,
and glyph rasterization remain outside accelerate.

`video::yuv420_to_rgba` converts padded 8-bit planar video using paired-row NEON
on ARM64 and portable scalar conversion elsewhere. BT.601/BT.709 and the
decoder's low-luma policy are explicit, so sharing the conversion does not change
codec output. Entropy decoding, prediction, and filtering remain in webmedia.

```rust
use accelerate::draw_surface;
use tiny_skia::{Pixmap, PixmapPaint, Transform};

let mut target = Pixmap::new(640, 480).unwrap();
let source = Pixmap::new(32, 32).unwrap();
draw_surface(
    &mut target,
    source.as_ref(),
    10,
    20,
    &PixmapPaint::default(),
    Transform::identity(),
    None,
);
```

The API currently covers presentation and software pixel operations, not arbitrary GPU
shader execution. Device, queue, and compute APIs can be added independently of
browser compositor policy.

## Validation

`backend-check` includes both EGL and WGL source modules for type checking on a
different development OS. This does not substitute for on-device presentation
testing. Do not enable this feature in production; normal builds select only the
host OS backend.

## License

GPL-3.0-or-later or a separately obtained commercial license.
