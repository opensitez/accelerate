//! CPU YUV conversion shared by media decoders; no GPU codec operations.

/// An 8-bit planar image with an independently padded row stride.
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum YuvMatrix {
    Bt601,
    Bt709,
}

/// Preserve the decoder's limited-range luma policy before chroma is added.
#[derive(Clone, Copy, Debug)]
pub enum LumaPolicy {
    /// VPx: negative Y-16 participates in the conversion.
    Unclamped,
    /// H.264/Y4M: clamp Y-16 to zero before multiplying by 298.
    Clamped,
}

fn valid_plane(plane: Plane<'_>, columns: usize, rows: usize) -> bool {
    plane.stride >= columns
        && (rows - 1)
            .checked_mul(plane.stride)
            .and_then(|offset| offset.checked_add(columns))
            .is_some_and(|required| required <= plane.data.len())
}

/// Convert planar 4:2:0 to tightly packed opaque RGBA. Chroma dimensions round
/// up for odd sizes. Padding after the final visible row need not be present.
/// Returns `None` for invalid strides, short planes, or dimension overflow.
pub fn yuv420_to_rgba(
    width: usize,
    height: usize,
    y: Plane<'_>,
    u: Plane<'_>,
    v: Plane<'_>,
    matrix: YuvMatrix,
    luma: LumaPolicy,
) -> Option<Vec<u8>> {
    let bytes = width.checked_mul(height)?.checked_mul(4)?;
    if bytes == 0 {
        return Some(Vec::new());
    }
    if !valid_plane(y, width, height)
        || !valid_plane(u, width.div_ceil(2), height.div_ceil(2))
        || !valid_plane(v, width.div_ceil(2), height.div_ceil(2))
    {
        return None;
    }
    Some(match (matrix, luma) {
        (YuvMatrix::Bt601, LumaPolicy::Unclamped) => {
            convert::<true, false, false>(width, height, y, u, v)
        }
        (YuvMatrix::Bt601, LumaPolicy::Clamped) => {
            convert::<true, false, true>(width, height, y, u, v)
        }
        (YuvMatrix::Bt709, LumaPolicy::Unclamped) => {
            convert::<true, true, false>(width, height, y, u, v)
        }
        (YuvMatrix::Bt709, LumaPolicy::Clamped) => {
            convert::<true, true, true>(width, height, y, u, v)
        }
    })
}

fn convert<const VECTOR: bool, const BT709: bool, const CLAMP: bool>(
    width: usize,
    height: usize,
    y: Plane<'_>,
    u: Plane<'_>,
    v: Plane<'_>,
) -> Vec<u8> {
    let mut rgba = vec![0; width * height * 4];
    let row_bytes = width * 4;
    let pair_bytes = row_bytes.checked_mul(2).unwrap_or(rgba.len());
    for (pair, output) in rgba.chunks_mut(pair_bytes).enumerate() {
        let row = pair * 2;
        let first = &y.data[row * y.stride..][..width];
        let second = if row + 1 < height {
            &y.data[(row + 1) * y.stride..][..width]
        } else {
            &[]
        };
        let cb = &u.data[pair * u.stride..][..width.div_ceil(2)];
        let cr = &v.data[pair * v.stride..][..width.div_ceil(2)];
        let (first_target, second_target) = output.split_at_mut(row_bytes);
        #[cfg(target_arch = "aarch64")]
        let processed = if VECTOR {
            // AArch64 guarantees NEON. Plane bounds were checked before allocation.
            if second.is_empty() {
                unsafe {
                    rows_neon::<BT709, CLAMP, false>(
                        first,
                        second,
                        cb,
                        cr,
                        first_target,
                        second_target,
                    )
                }
            } else {
                unsafe {
                    rows_neon::<BT709, CLAMP, true>(
                        first,
                        second,
                        cb,
                        cr,
                        first_target,
                        second_target,
                    )
                }
            }
        } else {
            0
        };
        #[cfg(not(target_arch = "aarch64"))]
        let processed = 0;
        rows_scalar::<BT709, CLAMP>(
            first,
            second,
            cb,
            cr,
            first_target,
            second_target,
            processed,
        );
    }
    rgba
}

fn rows_scalar<const BT709: bool, const CLAMP: bool>(
    first: &[u8],
    second: &[u8],
    cb: &[u8],
    cr: &[u8],
    first_target: &mut [u8],
    second_target: &mut [u8],
    start: usize,
) {
    #[inline]
    fn write<const CLAMP: bool>(
        samples: &[u8],
        target: &mut [u8],
        red: i32,
        green: i32,
        blue: i32,
    ) {
        for (pixel, &sample) in target.chunks_exact_mut(4).zip(samples) {
            let y = i32::from(sample) - 16;
            let luma = 298 * if CLAMP { y.max(0) } else { y };
            pixel[0] = ((luma + red) >> 8).clamp(0, 255) as u8;
            pixel[1] = ((luma + green) >> 8).clamp(0, 255) as u8;
            pixel[2] = ((luma + blue) >> 8).clamp(0, 255) as u8;
            pixel[3] = 255;
        }
    }
    for x in (start..first.len()).step_by(2) {
        let u = i32::from(cb[x / 2]) - 128;
        let v = i32::from(cr[x / 2]) - 128;
        let red = (if BT709 { 459 } else { 409 }) * v + 128;
        let green = -(if BT709 { 55 } else { 100 }) * u - (if BT709 { 136 } else { 208 }) * v + 128;
        let blue = (if BT709 { 541 } else { 516 }) * u + 128;
        let end = (x + 2).min(first.len());
        write::<CLAMP>(
            &first[x..end],
            &mut first_target[x * 4..end * 4],
            red,
            green,
            blue,
        );
        if !second.is_empty() {
            write::<CLAMP>(
                &second[x..end],
                &mut second_target[x * 4..end * 4],
                red,
                green,
                blue,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rows_neon<const BT709: bool, const CLAMP: bool, const PAIR: bool>(
    first: &[u8],
    second: &[u8],
    cb: &[u8],
    cr: &[u8],
    first_target: &mut [u8],
    second_target: &mut [u8],
) -> usize {
    use std::arch::aarch64::*;
    let chroma = |u, v| {
        let u = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u)), vdupq_n_s16(128));
        let v = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v)), vdupq_n_s16(128));
        let terms = |u, v| {
            (
                vmull_n_s16(v, if BT709 { 459 } else { 409 }),
                vsubq_s32(
                    vnegq_s32(vmull_n_s16(u, if BT709 { 55 } else { 100 })),
                    vmull_n_s16(v, if BT709 { 136 } else { 208 }),
                ),
                vmull_n_s16(u, if BT709 { 541 } else { 516 }),
            )
        };
        [
            terms(vget_low_s16(u), vget_low_s16(v)),
            terms(vget_high_s16(u), vget_high_s16(v)),
        ]
    };
    let convert = |luma, chroma: &[(int32x4_t, int32x4_t, int32x4_t); 2]| {
        let mut luma = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(luma)), vdupq_n_s16(16));
        if CLAMP {
            luma = vmaxq_s16(luma, vdupq_n_s16(0));
        }
        let four = |luma, terms: &(int32x4_t, int32x4_t, int32x4_t)| {
            let base = vmlal_n_s16(vdupq_n_s32(128), luma, 298);
            let rounded = |value| vqmovun_s32(vshrq_n_s32::<8>(vaddq_s32(base, value)));
            (rounded(terms.0), rounded(terms.1), rounded(terms.2))
        };
        let lo = four(vget_low_s16(luma), &chroma[0]);
        let hi = four(vget_high_s16(luma), &chroma[1]);
        uint8x8x4_t(
            vqmovn_u16(vcombine_u16(lo.0, hi.0)),
            vqmovn_u16(vcombine_u16(lo.1, hi.1)),
            vqmovn_u16(vcombine_u16(lo.2, hi.2)),
            vdup_n_u8(255),
        )
    };
    let processed = first.len() / 16 * 16;
    for x in (0..processed).step_by(16) {
        // Load each chroma block once, retaining wide products across both rows.
        let first_y = unsafe { vld1q_u8(first[x..x + 16].as_ptr()) };
        let second_y = if PAIR {
            unsafe { vld1q_u8(second[x..x + 16].as_ptr()) }
        } else {
            vdupq_n_u8(0)
        };
        let u = unsafe { vld1_u8(cb[x / 2..x / 2 + 8].as_ptr()) };
        let v = unsafe { vld1_u8(cr[x / 2..x / 2 + 8].as_ptr()) };
        // Finish the low half for both rows before computing the high products:
        // this matches the original paired VPx kernel's register lifetimes.
        let lo = chroma(vzip1_u8(u, u), vzip1_u8(v, v));
        unsafe {
            vst4_u8(
                first_target[x * 4..x * 4 + 32].as_mut_ptr(),
                convert(vget_low_u8(first_y), &lo),
            );
            if PAIR {
                vst4_u8(
                    second_target[x * 4..x * 4 + 32].as_mut_ptr(),
                    convert(vget_low_u8(second_y), &lo),
                );
            }
        }
        let hi = chroma(vzip2_u8(u, u), vzip2_u8(v, v));
        unsafe {
            vst4_u8(
                first_target[x * 4 + 32..x * 4 + 64].as_mut_ptr(),
                convert(vget_high_u8(first_y), &hi),
            );
            if PAIR {
                vst4_u8(
                    second_target[x * 4 + 32..x * 4 + 64].as_mut_ptr(),
                    convert(vget_high_u8(second_y), &hi),
                );
            }
        }
    }
    let mut x = processed;
    if x + 8 <= first.len() {
        // An eight-pixel tail needs only four chroma bytes, including on a
        // minimally allocated final chroma row. Never load past that row.
        let u = vcreate_u8(u32::from_le_bytes(cb[x / 2..x / 2 + 4].try_into().unwrap()) as u64);
        let v = vcreate_u8(u32::from_le_bytes(cr[x / 2..x / 2 + 4].try_into().unwrap()) as u64);
        let terms = chroma(vzip1_u8(u, u), vzip1_u8(v, v));
        let first_y = unsafe { vld1_u8(first[x..x + 8].as_ptr()) };
        unsafe {
            vst4_u8(
                first_target[x * 4..x * 4 + 32].as_mut_ptr(),
                convert(first_y, &terms),
            );
            if PAIR {
                let second_y = vld1_u8(second[x..x + 8].as_ptr());
                vst4_u8(
                    second_target[x * 4..x * 4 + 32].as_mut_ptr(),
                    convert(second_y, &terms),
                );
            }
        }
        x += 8;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(
        width: usize,
        height: usize,
        y: Plane<'_>,
        u: Plane<'_>,
        v: Plane<'_>,
        bt709: bool,
        clamp: bool,
    ) -> Vec<u8> {
        let mut result = Vec::new();
        for row in 0..height {
            for x in 0..width {
                let luma = i32::from(y.data[row * y.stride + x]) - 16;
                let c = 298 * if clamp { luma.max(0) } else { luma };
                let u = i32::from(u.data[row / 2 * u.stride + x / 2]) - 128;
                let v = i32::from(v.data[row / 2 * v.stride + x / 2]) - 128;
                result.extend([
                    ((c + (if bt709 { 459 } else { 409 }) * v + 128) >> 8).clamp(0, 255) as u8,
                    ((c - (if bt709 { 55 } else { 100 }) * u - (if bt709 { 136 } else { 208 }) * v
                        + 128)
                        >> 8)
                        .clamp(0, 255) as u8,
                    ((c + (if bt709 { 541 } else { 516 }) * u + 128) >> 8).clamp(0, 255) as u8,
                    255,
                ]);
            }
        }
        result
    }

    #[test]
    fn yuv420_policies_match_reference_with_odd_strides_and_tails() {
        for width in 1_usize..=65 {
            for height in 1_usize..=5 {
                let ys = width + 3;
                let us = width.div_ceil(2) + 5;
                let vs = width.div_ceil(2) + 7;
                let ch = height.div_ceil(2);
                let y: Vec<_> = (0..(height - 1) * ys + width)
                    .map(|i| (i * 53 + 7) as u8)
                    .collect();
                let u: Vec<_> = (0..(ch - 1) * us + width.div_ceil(2))
                    .map(|i| (i * 37 + 3) as u8)
                    .collect();
                let v: Vec<_> = (0..(ch - 1) * vs + width.div_ceil(2))
                    .map(|i| (i * 71 + 29) as u8)
                    .collect();
                let y = Plane {
                    data: &y,
                    stride: ys,
                };
                let u = Plane {
                    data: &u,
                    stride: us,
                };
                let v = Plane {
                    data: &v,
                    stride: vs,
                };
                for (matrix, bt709) in [(YuvMatrix::Bt601, false), (YuvMatrix::Bt709, true)] {
                    for (policy, clamp) in
                        [(LumaPolicy::Unclamped, false), (LumaPolicy::Clamped, true)]
                    {
                        let actual =
                            yuv420_to_rgba(width, height, y, u, v, matrix, policy).unwrap();
                        assert_eq!(
                            actual,
                            reference(width, height, y, u, v, bt709, clamp),
                            "{width}x{height} {matrix:?} {policy:?}"
                        );
                    }
                }
                assert_eq!(
                    convert::<false, false, false>(width, height, y, u, v),
                    reference(width, height, y, u, v, false, false)
                );
            }
        }
    }

    #[test]
    fn yuv420_low_luma_policies_remain_distinct() {
        let y = Plane {
            data: &[0],
            stride: 1,
        };
        let chroma = Plane {
            data: &[255],
            stride: 1,
        };
        for (matrix, unclamped, clamped) in [
            (YuvMatrix::Bt601, [184, 0, 237, 255], [203, 0, 255, 255]),
            (YuvMatrix::Bt709, [209, 0, 250, 255], [228, 0, 255, 255]),
        ] {
            assert_eq!(
                yuv420_to_rgba(1, 1, y, chroma, chroma, matrix, LumaPolicy::Unclamped).unwrap(),
                unclamped
            );
            assert_eq!(
                yuv420_to_rgba(1, 1, y, chroma, chroma, matrix, LumaPolicy::Clamped).unwrap(),
                clamped
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn yuv420_neon_rows_preserve_output_guards() {
        fn check<const BT709: bool, const CLAMP: bool>(width: usize) {
            let first: Vec<_> = (0..width).map(|i| (i * 53 + 7) as u8).collect();
            let second: Vec<_> = (0..width).map(|i| (i * 71 + 3) as u8).collect();
            let u: Vec<_> = (0..width.div_ceil(2))
                .map(|i| (i * 37 + 13) as u8)
                .collect();
            let v: Vec<_> = (0..width.div_ceil(2))
                .map(|i| (i * 47 + 29) as u8)
                .collect();
            let mut first_target = vec![91; width * 4 + 8];
            let mut second_target = first_target.clone();
            let end = width * 4 + 4;
            let processed = unsafe {
                rows_neon::<BT709, CLAMP, true>(
                    &first,
                    &second,
                    &u,
                    &v,
                    &mut first_target[4..end],
                    &mut second_target[4..end],
                )
            };
            rows_scalar::<BT709, CLAMP>(
                &first,
                &second,
                &u,
                &v,
                &mut first_target[4..end],
                &mut second_target[4..end],
                processed,
            );
            let mut expected_first = vec![0; width * 4];
            let mut expected_second = expected_first.clone();
            rows_scalar::<BT709, CLAMP>(
                &first,
                &second,
                &u,
                &v,
                &mut expected_first,
                &mut expected_second,
                0,
            );
            assert_eq!(&first_target[4..end], expected_first);
            assert_eq!(&second_target[4..end], expected_second);
            for target in [first_target, second_target] {
                assert!(
                    target[..4]
                        .iter()
                        .chain(&target[end..])
                        .all(|&byte| byte == 91)
                );
            }
            let mut single_target = vec![91; width * 4 + 8];
            let mut unused_target = [91; 8];
            let processed = unsafe {
                rows_neon::<BT709, CLAMP, false>(
                    &first,
                    &[],
                    &u,
                    &v,
                    &mut single_target[4..end],
                    &mut unused_target,
                )
            };
            rows_scalar::<BT709, CLAMP>(
                &first,
                &[],
                &u,
                &v,
                &mut single_target[4..end],
                &mut unused_target,
                processed,
            );
            assert_eq!(&single_target[4..end], expected_first);
            assert!(
                single_target[..4]
                    .iter()
                    .chain(&single_target[end..])
                    .all(|&byte| byte == 91)
            );
            assert_eq!(unused_target, [91; 8]);
        }
        for width in 1..=65 {
            check::<false, false>(width);
            check::<false, true>(width);
            check::<true, false>(width);
            check::<true, true>(width);
        }
    }

    #[test]
    fn yuv420_rejects_short_planes_bad_strides_and_overflow() {
        let valid_y = Plane {
            data: &[128; 6],
            stride: 3,
        };
        let valid = Plane {
            data: &[128; 4],
            stride: 2,
        };
        let short = Plane {
            data: &[128],
            stride: 2,
        };
        let narrow = Plane {
            data: &[128; 4],
            stride: 1,
        };
        for planes in [
            (short, valid, valid),
            (narrow, valid, valid),
            (valid_y, short, valid),
            (valid_y, valid, short),
        ] {
            assert!(
                yuv420_to_rgba(
                    3,
                    2,
                    planes.0,
                    planes.1,
                    planes.2,
                    YuvMatrix::Bt601,
                    LumaPolicy::Unclamped
                )
                .is_none()
            );
        }
        assert!(
            yuv420_to_rgba(
                3,
                2,
                valid_y,
                valid,
                valid,
                YuvMatrix::Bt601,
                LumaPolicy::Unclamped
            )
            .is_some()
        );
        assert!(
            yuv420_to_rgba(
                usize::MAX,
                2,
                valid,
                valid,
                valid,
                YuvMatrix::Bt601,
                LumaPolicy::Unclamped
            )
            .is_none()
        );
        assert_eq!(
            yuv420_to_rgba(
                0,
                0,
                short,
                short,
                short,
                YuvMatrix::Bt601,
                LumaPolicy::Unclamped
            ),
            Some(Vec::new())
        );
    }
}
