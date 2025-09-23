use crate::atom::util::parser::{FIXED_POINT_16X16_SCALE, FIXED_POINT_8X8_SCALE};

pub fn fixed_point_16x16(val: f32) -> Vec<u8> {
    let fixed = (val * FIXED_POINT_16X16_SCALE) as u32;
    fixed.to_be_bytes().to_vec()
}

pub fn fixed_point_8x8(val: f32) -> Vec<u8> {
    let fixed = (val * FIXED_POINT_8X8_SCALE) as u16;
    fixed.to_be_bytes().to_vec()
}
