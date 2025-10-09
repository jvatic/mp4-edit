use std::{collections::VecDeque, marker::PhantomData};

use crate::atom::util::{
    parser::{FIXED_POINT_16X16_SCALE, FIXED_POINT_8X8_SCALE},
    ColorRgb,
};

pub fn be_u32(value: u32) -> [u8; 4] {
    value.to_be_bytes()
}

pub fn be_u24(value: u32) -> [u8; 3] {
    [(value >> 16) as u8, (value >> 8) as u8, value as u8]
}

pub fn fixed_point_16x16(val: f32) -> [u8; 4] {
    let fixed = (val * FIXED_POINT_16X16_SCALE) as u32;
    fixed.to_be_bytes()
}

pub fn fixed_point_8x8(val: f32) -> [u8; 2] {
    let fixed = (val * FIXED_POINT_8X8_SCALE) as u16;
    fixed.to_be_bytes()
}

/// Prepends the size of `f()` + `Size`, according to `Size`
pub fn prepend_size_inclusive<Size, F>(f: F) -> Vec<u8>
where
    SizeInclusive<Size>: SerializeSize,
    F: FnOnce() -> Vec<u8>,
{
    let inner = f();
    let mut size = <SizeInclusive<Size> as SerializeSize>::serialize_size(inner.len());
    size.extend(inner);
    size
}

/// Prepends the size of `f()`, according to `Size`
pub fn prepend_size_exclusive<Size, F>(f: F) -> Vec<u8>
where
    SizeExclusive<Size>: SerializeSize,
    F: FnOnce() -> Vec<u8>,
{
    let inner = f();
    let mut size = <SizeExclusive<Size> as SerializeSize>::serialize_size(inner.len());
    size.extend(inner);
    size
}

pub fn pascal_string(s: String) -> Vec<u8> {
    prepend_size_exclusive::<SizeU8, _>(move || s.into_bytes())
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

/// Serialize size as a Variable Length Quantity, using at most `S` bytes
#[derive(Debug, Clone)]
pub struct SizeVLQ<S>(PhantomData<S>);

#[derive(Debug, Clone)]
pub struct SizeInclusive<S>(PhantomData<S>);

#[derive(Debug, Clone)]
pub struct SizeExclusive<S>(PhantomData<S>);

pub trait SerializeSize {
    fn serialize_size(size: usize) -> Vec<u8>;
}

const U8_BYTE_SIZE: usize = 1;
impl SerializeSize for SizeInclusive<SizeU8> {
    fn serialize_size(size: usize) -> Vec<u8> {
        vec![(size + U8_BYTE_SIZE) as u8]
    }
}

const U32_BYTE_SIZE: usize = 4;
impl SerializeSize for SizeInclusive<SizeU32> {
    fn serialize_size(size: usize) -> Vec<u8> {
        ((size + U32_BYTE_SIZE) as u32).to_be_bytes().to_vec()
    }
}

const U64_BYTE_SIZE: usize = 8;
impl SerializeSize for SizeInclusive<SizeU64> {
    fn serialize_size(size: usize) -> Vec<u8> {
        ((size + U64_BYTE_SIZE) as u64).to_be_bytes().to_vec()
    }
}

impl SerializeSize for SizeInclusive<SizeU32OrU64> {
    fn serialize_size(size: usize) -> Vec<u8> {
        if size + U32_BYTE_SIZE <= u32::MAX as usize {
            SizeInclusive::<SizeU32>::serialize_size(size)
        } else {
            let mut output = vec![1];
            output.extend(SizeInclusive::<SizeU64>::serialize_size(size + 1));
            output
        }
    }
}

impl SerializeSize for SizeExclusive<SizeU8> {
    fn serialize_size(size: usize) -> Vec<u8> {
        vec![size as u8]
    }
}

impl SerializeSize for SizeExclusive<SizeVLQ<SizeU32>> {
    fn serialize_size(size: usize) -> Vec<u8> {
        variable_length_quantity::<U32_BYTE_SIZE>(size)
    }
}

fn variable_length_quantity<const N: usize>(mut length: usize) -> Vec<u8> {
    let mut data = VecDeque::with_capacity(N);
    for _ in 0..N {
        let mut byte = (length & 0b0111_1111) as u8;
        length >>= 7;

        if !data.is_empty() {
            byte |= 0b1000_0000;
        }

        data.push_front(byte);

        if length == 0 {
            break;
        }
    }

    data.into()
}

pub fn color_rgb(color: ColorRgb) -> [u8; 6] {
    let mut data = Vec::with_capacity(6);
    data.extend(color.red.to_be_bytes());
    data.extend(color.green.to_be_bytes());
    data.extend(color.blue.to_be_bytes());
    data.try_into().expect("color_rgb is 6 bytes")
}

pub mod bits {
    use std::fmt;

    pub struct Packer {
        full_bytes: Vec<u8>,
        partial_byte: u8,
        bit_offset: u8,
    }

    impl fmt::Debug for Packer {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Packer")
                .field(
                    "full_bytes",
                    &self
                        .full_bytes
                        .iter()
                        .map(|b| format!("{b:08b}"))
                        .collect::<Vec<_>>(),
                )
                .field("partial_byte", &format!("{:08b}", self.partial_byte))
                .field("bit_offset", &self.bit_offset)
                .finish()
        }
    }

    const BYTE: u8 = 8;

    impl From<Vec<u8>> for Packer {
        fn from(value: Vec<u8>) -> Self {
            Packer {
                full_bytes: value,
                partial_byte: 0,
                bit_offset: 0,
            }
        }
    }

    impl From<Packer> for Vec<u8> {
        fn from(packer: Packer) -> Self {
            let mut res = packer.full_bytes;
            if packer.bit_offset > 0 {
                res.push(packer.partial_byte);
            }
            res
        }
    }

    impl Packer {
        pub fn new() -> Self {
            Packer {
                full_bytes: Vec::new(),
                partial_byte: 0u8,
                bit_offset: 0u8,
            }
        }

        pub fn push_bool(&mut self, b: bool) {
            self.push_n::<1>(b as u8);
        }

        pub fn push_n<const N: u8>(&mut self, bits: u8) {
            debug_assert!(N <= 8 && N > 0, "N must be <= 8 and > 0");

            let n = N;
            let available = BYTE - self.bit_offset;

            if n <= available {
                // All bits fit in current partial byte
                let shift_left = available - n;
                self.partial_byte |= bits << shift_left;
                self.bit_offset += n;

                if n == available {
                    self.full_bytes.push(self.partial_byte);
                    self.partial_byte = 0;
                    self.bit_offset = 0;
                }
            } else {
                // We need to split it across two bytes
                let shift_right = n - available;
                let first =
                    self.partial_byte | trim_higher_bits(BYTE - available, bits >> shift_right);
                self.full_bytes.push(first);

                let shift_left = BYTE - n + available;
                let second = bits << shift_left;
                self.partial_byte = second;
                self.bit_offset = n - available;
            }
        }

        pub fn push_n_u32<const N: u8>(&mut self, bits: u32) {
            let n_bytes = N / BYTE;
            let n_bits = N % BYTE;
            for bn in 1..=n_bytes {
                let shift_right = (n_bytes * BYTE) - (bn * BYTE);
                self.push_n::<BYTE>((bits >> shift_right) as u8);
            }
            if n_bits == 0 {
                return;
            }
            let bits = (bits >> (n_bytes * BYTE)) as u8;
            match n_bits {
                1 => self.push_n::<1>(bits),
                2 => self.push_n::<2>(bits),
                3 => self.push_n::<3>(bits),
                4 => self.push_n::<4>(bits),
                5 => self.push_n::<5>(bits),
                6 => self.push_n::<6>(bits),
                7 => self.push_n::<7>(bits),
                _ => unreachable!(),
            }
        }

        pub fn push_bytes(&mut self, bytes: Vec<u8>) {
            if self.bit_offset == 0 {
                self.full_bytes.extend(bytes);
                return;
            }

            for byte in bytes {
                self.push_n::<8>(byte);
            }
        }
    }

    fn trim_higher_bits(n: u8, bits: u8) -> u8 {
        let bits = bits << n;
        bits >> n
    }

    #[cfg(test)]
    mod tests {
        use crate::atom::{test_utils::assert_bytes_equal, util::serializer::bits::Packer};

        macro_rules! test_push_n {
            ($name:ident($fn_name:ident), {
                full_bytes: $full_bytes:expr,
                partial_byte: $partial_byte:expr,
                bit_offset: $bit_offset:expr,
                push_bits: $push_bits:expr,
                push_bits_n: $push_bits_n:expr,
                expect_partial_byte: $expect_partial_byte:expr,
                expect_bit_offset: $expect_bit_offset:expr,
                expect_full_bytes: $expect_full_bytes:expr,
            }) => {
                test_push_n!($name($fn_name), {
                    rounds: 1,
                    full_bytes: $full_bytes,
                    partial_byte: $partial_byte,
                    bit_offset: $bit_offset,
                    push_bits: $push_bits,
                    push_bits_n: $push_bits_n,
                    expect_partial_byte: $expect_partial_byte,
                    expect_bit_offset: $expect_bit_offset,
                    expect_full_bytes: $expect_full_bytes,
                });
            };

            ($name:ident($fn_name:ident), {
                rounds: $rounds:literal,
                full_bytes: $full_bytes:expr,
                partial_byte: $partial_byte:expr,
                bit_offset: $bit_offset:expr,
                push_bits: $push_bits:expr,
                push_bits_n: $push_bits_n:expr,
                expect_partial_byte: $expect_partial_byte:expr,
                expect_bit_offset: $expect_bit_offset:expr,
                expect_full_bytes: $expect_full_bytes:expr,
            }) => {
                #[test]
                fn $name() {
                    let mut packer = Packer {
                        full_bytes: $full_bytes,
                        partial_byte: $partial_byte,
                        bit_offset: $bit_offset,
                    };
                    let rounds = $rounds;
                    for n in 1..=rounds {
                        eprintln!("round {n}");
                        packer.$fn_name::<$push_bits_n>($push_bits);
                    }
                    assert_bits_eq(packer.partial_byte, $expect_partial_byte);
                    assert_eq!(packer.bit_offset, $expect_bit_offset);
                    assert_bytes_equal(&packer.full_bytes, &$expect_full_bytes);

                    // ensure converting packer into a Vec handles partial byte
                    if $expect_bit_offset > 0 {
                        let mut expected_full_bytes = $expect_full_bytes;
                        expected_full_bytes.push($expect_partial_byte);
                        let full_bytes: Vec<_> = packer.into();
                        assert_bytes_equal(&full_bytes, &expected_full_bytes);
                    } else {
                        let expected_full_bytes = $expect_full_bytes;
                        let full_bytes: Vec<_> = packer.into();
                        assert_bytes_equal(&full_bytes, &expected_full_bytes);
                    }
                }
            };
        }

        test_push_n!(test_push_1_first_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b0000_0000,
            bit_offset: 0,
            push_bits: 0b0000_0001,
            push_bits_n: 1,
            expect_partial_byte: 0b1000_0000,
            expect_bit_offset: 1,
            expect_full_bytes: vec![],
        });

        test_push_n!(test_push_1_second_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1000_0000,
            bit_offset: 1,
            push_bits: 0b0000_0001,
            push_bits_n: 1,
            expect_partial_byte: 0b1100_0000,
            expect_bit_offset: 2,
            expect_full_bytes: vec![],
        });

        test_push_n!(test_push_1_third_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1100_0000,
            bit_offset: 2,
            push_bits: 0b0000_0001,
            push_bits_n: 1,
            expect_partial_byte: 0b1110_0000,
            expect_bit_offset: 3,
            expect_full_bytes: vec![],
        });

        test_push_n!(test_push_1_fourth_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1110_0000,
            bit_offset: 3,
            push_bits: 0b0000_0001,
            push_bits_n: 1,
            expect_partial_byte: 0b1111_0000,
            expect_bit_offset: 4,
            expect_full_bytes: vec![],
        });

        test_push_n!(test_push_1_fifth_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1111_0000,
            bit_offset: 4,
            push_bits: 0b0000_0001,
            push_bits_n: 1,
            expect_partial_byte: 0b1111_1000,
            expect_bit_offset: 5,
            expect_full_bytes: vec![],
        });

        test_push_n!(test_push_1_sixth_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1111_1000,
            bit_offset: 5,
            push_bits: 0b0000_0001,
            push_bits_n: 1,
            expect_partial_byte: 0b1111_1100,
            expect_bit_offset: 6,
            expect_full_bytes: vec![],
        });

        test_push_n!(test_push_1_seventh_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1111_1100,
            bit_offset: 6,
            push_bits: 0b0000_0001,
            push_bits_n: 1,
            expect_partial_byte: 0b1111_1110,
            expect_bit_offset: 7,
            expect_full_bytes: vec![],
        });

        test_push_n!(test_push_1_eighth_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1111_1110,
            bit_offset: 7,
            push_bits: 0b0000_0001,
            push_bits_n: 1,
            expect_partial_byte: 0b0000_0000,
            expect_bit_offset: 0,
            expect_full_bytes: vec![0b1111_1111],
        });

        test_push_n!(test_push_7_eighth_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1111_1110,
            bit_offset: 7,
            push_bits: 0b0111_1111,
            push_bits_n: 7,
            expect_partial_byte: 0b1111_1100,
            expect_bit_offset: 6,
            expect_full_bytes: vec![0b1111_1111],
        });

        test_push_n!(test_push_6_seventh_bit(push_n), {
            full_bytes: vec![0b1111_1111],
            partial_byte: 0b1111_1100,
            bit_offset: 6,
            push_bits: 0b0011_1111,
            push_bits_n: 6,
            expect_partial_byte: 0b1111_0000,
            expect_bit_offset: 4,
            expect_full_bytes: vec![0b1111_1111, 0b1111_1111],
        });

        test_push_n!(test_push_6_fifth_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1111_0000,
            bit_offset: 4,
            push_bits: 0b0011_1111,
            push_bits_n: 6,
            expect_partial_byte: 0b1100_0000,
            expect_bit_offset: 2,
            expect_full_bytes: vec![0b1111_1111],
        });

        test_push_n!(test_push_7_first_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b0000_0000,
            bit_offset: 0,
            push_bits: 0b0111_1111,
            push_bits_n: 7,
            expect_partial_byte: 0b1111_1110,
            expect_bit_offset: 7,
            expect_full_bytes: vec![],
        });

        test_push_n!(test_push_8_second_bit(push_n), {
            rounds: 2,
            full_bytes: vec![0b1010_1010],
            partial_byte: 0b1000_0000,
            bit_offset: 1,
            push_bits: 0b1111_1111,
            push_bits_n: 8,
            expect_partial_byte: 0b1000_0000,
            expect_bit_offset: 1,
            expect_full_bytes: vec![0b1010_1010, 0b1111_1111, 0b1111_1111],
        });

        test_push_n!(test_push_7_incomplete_bits_fifth_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b1111_0000,
            bit_offset: 4,
            push_bits: 0b0100_1000,
            push_bits_n: 7,
            expect_partial_byte: 0b0000_0000,
            expect_bit_offset: 3,
            expect_full_bytes: vec![0b1111_1001],
        });

        test_push_n!(test_push_4_fifth_bit(push_n), {
            full_bytes: vec![],
            partial_byte: 0b0011_0000,
            bit_offset: 5,
            push_bits: 0b0000_0011,
            push_bits_n: 4,
            expect_partial_byte: 0b1000_0000,
            expect_bit_offset: 1,
            expect_full_bytes: vec![0b0011_0001],
        });

        test_push_n!(test_push_24_first_bit(push_n_u32), {
            full_bytes: vec![],
            partial_byte: 0b0000_0000,
            bit_offset: 0,
            push_bits: 0b00000000_11111111_11111111_11111111,
            push_bits_n: 24,
            expect_partial_byte: 0b0000_0000,
            expect_bit_offset: 0,
            expect_full_bytes: vec![0b1111_1111, 0b1111_1111, 0b1111_1111],
        });

        test_push_n!(test_push_24_second_bit(push_n_u32), {
            full_bytes: vec![],
            partial_byte: 0b1000_0000,
            bit_offset: 1,
            push_bits: 0b00000000_11111111_11111111_11111111,
            push_bits_n: 24,
            expect_partial_byte: 0b1000_0000,
            expect_bit_offset: 1,
            expect_full_bytes: vec![0b1111_1111, 0b1111_1111, 0b1111_1111],
        });

        test_push_n!(test_push_26_second_bit(push_n_u32), {
            full_bytes: vec![],
            partial_byte: 0b1000_0000,
            bit_offset: 1,
            push_bits: 0b00000011_11111111_11111111_11111111,
            push_bits_n: 26,
            expect_partial_byte: 0b1110_0000,
            expect_bit_offset: 3,
            expect_full_bytes: vec![0b1111_1111, 0b1111_1111, 0b1111_1111],
        });

        #[test]
        fn test_push_bytes_first_bit() {
            let mut packer = Packer {
                full_bytes: vec![0b1111_1111],
                partial_byte: 0b0000_0000,
                bit_offset: 0,
            };
            packer.push_bytes(vec![0b0000_0000, 0b0000_0000, 0b0000_0000]);
            let expected: Vec<u8> = vec![0b1111_1111, 0b0000_0000, 0b0000_0000, 0b0000_0000];
            assert_bytes_equal(&packer.full_bytes, &expected);
            assert_bits_eq(packer.partial_byte, 0b0000_0000);
            assert_eq!(packer.bit_offset, 0);
        }

        #[test]
        fn test_push_bytes_second_bit() {
            let mut packer = Packer {
                full_bytes: vec![0b1111_1111],
                partial_byte: 0b1000_0000,
                bit_offset: 1,
            };
            packer.push_bytes(vec![0b0000_0000, 0b0000_0000, 0b0000_0000]);
            let expected: Vec<u8> = vec![0b1111_1111, 0b1000_0000, 0b0000_0000, 0b0000_0000];
            assert_bytes_equal(&packer.full_bytes, &expected);
            assert_bits_eq(packer.partial_byte, 0b0000_0000);
            assert_eq!(packer.bit_offset, 1);
        }

        fn assert_bits_eq(actual: u8, expected: u8) {
            if actual == expected {
                return;
            }

            panic!("expected 0b{expected:08b}, got 0b{actual:08b}")
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::test_utils::assert_bytes_equal;

    use super::*;

    macro_rules! test_variable_length_quantity {
        ({ $( $name:ident::<$n:literal>($input:expr) => $expected:expr ),+ $(,)? }) => {
            $(#[test] fn $name() {
                let input = $input;
                let output = variable_length_quantity::<$n>(input);
                let expected: Vec<u8> = $expected;
                assert_bytes_equal(&output, &expected);
            })+
        };
    }

    test_variable_length_quantity!({
        test_u16_0::<2>(0) => vec![0x00],
        test_u16_127::<2>(127) => vec![0x7F],
        test_u16_128::<2>(128) => vec![0x81, 0x00],
        test_u16_358::<2>(358) => vec![0x82, 0x66],
        test_u32_358::<4>(358) => vec![0x82, 0x66],
        test_u64_358::<8>(358) => vec![0x82, 0x66],
    });
}
