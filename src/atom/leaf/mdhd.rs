use bon::Builder;
use std::{fmt, time::Duration};

use crate::{
    atom::{
        util::{mp4_timestamp_now, scaled_duration, unscaled_duration},
        FourCC,
    },
    parser::ParseAtomData,
    writer::SerializeAtom,
    ParseError,
};

pub const MDHD: FourCC = FourCC::new(b"mdhd");

macro_rules! define_language_code_enum {
    ($( #[$meta:meta] )* $name:ident { $( $( #[$tag:meta] )* $variant:ident => $chars:literal ),+ $(,)? }) => {
        $(#[$meta])*
        pub enum $name {
            $( $( #[$tag] )* $variant ),+,
            Other([u8; 3]),
        }

        impl $name {
            fn from_chars(chars: &[u8; 3]) -> Self {
                match chars {
                    $( $chars => Self::$variant ),+,
                    _ => Self::Other(*chars),
                }
            }

            fn as_chars(&self) -> &[u8; 3] {
                match self {
                    $( Self::$variant => $chars ),+,
                    Self::Other(chars) => chars,
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let chars = match self {
                    $( Self::$variant => $chars ),+,
                    Self::Other(chars) => &[chars[0] as u8, chars[1] as u8, chars[2] as u8],
                };
                write!(f, "{}{}{}", chars[0] as char, chars[1] as char, chars[2] as char)
            }
        }
    };
}

define_language_code_enum!(
    /// Language code (ISO 639-2/T language code)
    #[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
    #[non_exhaustive]
    LanguageCode {
        English => b"eng",
        Spanish => b"spa",
        French => b"fra",
        German => b"deu",
        Italian => b"ita",
        Japanese => b"jpn",
        Korean => b"kor",
        Chinese => b"chi",
        Russian => b"rus",
        Arabic => b"ara",
        Portuguese => b"por",
        #[default]
        Undetermined => b"und",
    }
);

impl From<u16> for LanguageCode {
    fn from(packed: u16) -> Self {
        let char1 = (((packed >> 10) & 0x1F) + 0x60) as u8;
        let char2 = (((packed >> 5) & 0x1F) + 0x60) as u8;
        let char3 = ((packed & 0x1F) + 0x60) as u8;

        let lang: [u8; 3] = [char1, char2, char3];

        Self::from_chars(&lang)
    }
}

impl From<LanguageCode> for u16 {
    fn from(value: LanguageCode) -> Self {
        let chars = value.as_chars();

        let char1_bits = (chars[0] - 0x60) & 0x1F;
        let char2_bits = (chars[1] - 0x60) & 0x1F;
        let char3_bits = (chars[2] - 0x60) & 0x1F;

        (u16::from(char1_bits) << 10) | (u16::from(char2_bits) << 5) | u16::from(char3_bits)
    }
}

#[derive(Default, Debug, Clone, Builder)]
pub struct MediaHeaderAtom {
    /// Version of the mdhd atom format (0 or 1)
    #[builder(default = 0)]
    pub version: u8,
    /// Flags for the mdhd atom (usually all zeros)
    #[builder(default = [0u8; 3])]
    pub flags: [u8; 3],
    /// Creation time (seconds since midnight, Jan. 1, 1904, UTC)
    #[builder(default = mp4_timestamp_now())]
    pub creation_time: u64,
    /// Modification time (seconds since midnight, Jan. 1, 1904, UTC)
    #[builder(default = mp4_timestamp_now())]
    pub modification_time: u64,
    /// Media timescale (number of time units per second)
    pub timescale: u32,
    /// Duration of media (in timescale units)
    pub duration: u64,
    /// Language code (ISO 639-2/T language code)
    #[builder(default = LanguageCode::Undetermined)]
    pub language: LanguageCode,
    /// Pre-defined value (should be 0)
    #[builder(default = 0)]
    pub pre_defined: u16,
}

impl MediaHeaderAtom {
    pub fn duration(&self) -> Duration {
        unscaled_duration(self.duration, u64::from(self.timescale))
    }

    pub fn update_duration<F>(&mut self, mut closure: F) -> &mut Self
    where
        F: FnMut(Duration) -> Duration,
    {
        self.duration = scaled_duration(closure(self.duration()), u64::from(self.timescale));
        self
    }
}

impl ParseAtomData for MediaHeaderAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, MDHD);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_mdhd_data.parse(stream(input))?)
    }
}

impl SerializeAtom for MediaHeaderAtom {
    fn atom_type(&self) -> FourCC {
        MDHD
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_mdhd_atom(self)
    }
}

mod serializer {
    use super::MediaHeaderAtom;

    pub fn serialize_mdhd_atom(mdhd: MediaHeaderAtom) -> Vec<u8> {
        let mut data = Vec::new();

        let version: u8 = if mdhd.version == 1
            || mdhd.creation_time > u64::from(u32::MAX)
            || mdhd.modification_time > u64::from(u32::MAX)
            || mdhd.duration > u64::from(u32::MAX)
        {
            1
        } else {
            0
        };

        let be_u32_or_u64 = |v: u64| match version {
            0 => u32::try_from(v).unwrap().to_be_bytes().to_vec(),
            1 => v.to_be_bytes().to_vec(),
            _ => unreachable!(),
        };

        data.extend(version.to_be_bytes());
        data.extend(mdhd.flags);
        data.extend(be_u32_or_u64(mdhd.creation_time));
        data.extend(be_u32_or_u64(mdhd.modification_time));
        data.extend(mdhd.timescale.to_be_bytes());
        data.extend(be_u32_or_u64(mdhd.duration));
        data.extend(u16::from(mdhd.language).to_be_bytes());
        data.extend(mdhd.pre_defined.to_be_bytes());

        data
    }
}

mod parser {
    use winnow::{
        binary::{be_u16, be_u32, be_u64},
        combinator::{seq, trace},
        error::{StrContext, StrContextValue},
        ModalResult, Parser,
    };

    use super::{LanguageCode, MediaHeaderAtom};
    use crate::atom::util::parser::{be_u32_as_u64, flags3, version, Stream};

    pub fn parse_mdhd_data(input: &mut Stream<'_>) -> ModalResult<MediaHeaderAtom> {
        let be_u32_or_u64 = |version: u8| {
            let be_u64_type_fix =
                |input: &mut Stream<'_>| -> ModalResult<u64> { be_u64.parse_next(input) };
            match version {
                0 => be_u32_as_u64,
                1 => be_u64_type_fix,
                _ => unreachable!(),
            }
        };

        trace(
            "mdhd",
            seq!(MediaHeaderAtom {
                version: version
                    .verify(|version| *version <= 1)
                    .context(StrContext::Expected(StrContextValue::Description(
                        "expected version 0 or 1"
                    ))),
                flags: flags3,
                creation_time: be_u32_or_u64(version),
                modification_time: be_u32_or_u64(version),
                timescale: be_u32,
                duration: be_u32_or_u64(version),
                language: be_u16.map(LanguageCode::from),
                pre_defined: be_u16,
            })
            .context(StrContext::Label("mdhd")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available mdhd test data files
    #[test]
    fn test_mdhd_roundtrip() {
        test_atom_roundtrip::<MediaHeaderAtom>(MDHD);
    }
}
