use crate::atom::util::parser::{FIXED_POINT_16X16_SCALE, FIXED_POINT_8X8_SCALE};

pub fn be_u32(value: u32) -> Vec<u8> {
    value.to_be_bytes().to_vec()
}

pub fn fixed_point_16x16(val: f32) -> Vec<u8> {
    let fixed = (val * FIXED_POINT_16X16_SCALE) as u32;
    fixed.to_be_bytes().to_vec()
}

pub fn fixed_point_8x8(val: f32) -> Vec<u8> {
    let fixed = (val * FIXED_POINT_8X8_SCALE) as u16;
    fixed.to_be_bytes().to_vec()
}

pub fn prepend_size<Size, F>(f: F) -> Vec<u8>
where
    Size: SerializeSize,
    F: FnOnce() -> Vec<u8>,
{
    let inner = f();
    let mut size = Size::serialize_size(inner.len());
    size.extend(inner);
    size
}

pub fn pascal_string(s: String) -> Vec<u8> {
    let mut data = s.into_bytes();
    let len = u8::try_from(data.len()).expect("string byte len must fit in u8");
    data.insert(0, len);
    data
}

/// Serialize u8(size)
#[derive(Debug, Clone)]
pub struct SizeU8;

/// Serialize be_u32(size)
#[derive(Debug, Clone)]
pub struct SizeU32;

/// Serialize be_u64(size)
#[derive(Debug, Clone)]
pub struct SizeU64;

/// Serialize be_u32(size), or be_u32(1) + be_u64(size)
#[derive(Debug, Clone)]
pub struct SizeU32OrU64;

pub trait SerializeSize {
    fn serialize_size(size: usize) -> Vec<u8>;
}

const U8_BYTE_SIZE: usize = 1;
impl SerializeSize for SizeU8 {
    fn serialize_size(size: usize) -> Vec<u8> {
        vec![(size + U8_BYTE_SIZE) as u8]
    }
}

const U32_BYTE_SIZE: usize = 4;
impl SerializeSize for SizeU32 {
    fn serialize_size(size: usize) -> Vec<u8> {
        ((size + U32_BYTE_SIZE) as u32).to_be_bytes().to_vec()
    }
}

const U64_BYTE_SIZE: usize = 8;
impl SerializeSize for SizeU64 {
    fn serialize_size(size: usize) -> Vec<u8> {
        ((size + U64_BYTE_SIZE) as u64).to_be_bytes().to_vec()
    }
}

impl SerializeSize for SizeU32OrU64 {
    fn serialize_size(size: usize) -> Vec<u8> {
        if size + U32_BYTE_SIZE <= u32::MAX as usize {
            SizeU32::serialize_size(size)
        } else {
            let mut output = vec![1];
            output.extend(SizeU64::serialize_size(size + 1));
            output
        }
    }
}
