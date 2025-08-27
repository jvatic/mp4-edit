use anyhow::{anyhow, Context};
use bon::Builder;
use futures_io::AsyncRead;
use std::{fmt, io::Read, time::Duration};

use crate::{
    atom::{
        util::{
            async_to_sync_read,
            time::{mp4_timestamp_now, scaled_duration, unscaled_duration},
        },
        FourCC,
    },
    parser::Parse,
    writer::SerializeAtom,
    ParseError,
};

pub const MDHD: &[u8; 4] = b"mdhd";

/// Language code (ISO 639-2/T language code)
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanguageCode {
    English,
    Spanish,
    French,
    German,
    Italian,
    Japanese,
    Korean,
    Chinese,
    Russian,
    Arabic,
    Portuguese,
    #[default]
    Undetermined,
    Other([char; 3]),
}

impl From<[u8; 3]> for LanguageCode {
    fn from(value: [u8; 3]) -> Self {
        // Decode the packed language format
        let packed = u16::from_be_bytes([value[0], value[1]]);
        let char1 = (((packed >> 10) & 0x1F) + 0x60) as u8 as char;
        let char2 = (((packed >> 5) & 0x1F) + 0x60) as u8 as char;
        let char3 = ((packed & 0x1F) + 0x60) as u8 as char;

        let lang_str = format!("{char1}{char2}{char3}");

        match lang_str.as_str() {
            "eng" => LanguageCode::English,
            "spa" => LanguageCode::Spanish,
            "fra" => LanguageCode::French,
            "deu" => LanguageCode::German,
            "ita" => LanguageCode::Italian,
            "jpn" => LanguageCode::Japanese,
            "kor" => LanguageCode::Korean,
            "chi" => LanguageCode::Chinese,
            "rus" => LanguageCode::Russian,
            "ara" => LanguageCode::Arabic,
            "por" => LanguageCode::Portuguese,
            "und" => LanguageCode::Undetermined,
            _ => LanguageCode::Other([char1, char2, char3]),
        }
    }
}

impl LanguageCode {
    /// Convert the language code to packed bytes format
    fn serialize(&self) -> [u8; 2] {
        let chars = match self {
            LanguageCode::English => ['e', 'n', 'g'],
            LanguageCode::Spanish => ['s', 'p', 'a'],
            LanguageCode::French => ['f', 'r', 'a'],
            LanguageCode::German => ['d', 'e', 'u'],
            LanguageCode::Italian => ['i', 't', 'a'],
            LanguageCode::Japanese => ['j', 'p', 'n'],
            LanguageCode::Korean => ['k', 'o', 'r'],
            LanguageCode::Chinese => ['c', 'h', 'i'],
            LanguageCode::Russian => ['r', 'u', 's'],
            LanguageCode::Arabic => ['a', 'r', 'a'],
            LanguageCode::Portuguese => ['p', 'o', 'r'],
            LanguageCode::Undetermined => ['u', 'n', 'd'],
            LanguageCode::Other(chars) => *chars,
        };

        // Pack into 16-bit format (3 x 5-bit values)
        let char1_bits = (chars[0] as u8 - 0x60) & 0x1F;
        let char2_bits = (chars[1] as u8 - 0x60) & 0x1F;
        let char3_bits = (chars[2] as u8 - 0x60) & 0x1F;

        let packed =
            (u16::from(char1_bits) << 10) | (u16::from(char2_bits) << 5) | u16::from(char3_bits);
        packed.to_be_bytes()
    }
}

impl fmt::Display for LanguageCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let code = match self {
            LanguageCode::English => "eng",
            LanguageCode::Spanish => "spa",
            LanguageCode::French => "fra",
            LanguageCode::German => "deu",
            LanguageCode::Italian => "ita",
            LanguageCode::Japanese => "jpn",
            LanguageCode::Korean => "kor",
            LanguageCode::Chinese => "chi",
            LanguageCode::Russian => "rus",
            LanguageCode::Arabic => "ara",
            LanguageCode::Portuguese => "por",
            LanguageCode::Undetermined => "und",
            LanguageCode::Other(chars) => {
                return write!(f, "{}{}{}", chars[0], chars[1], chars[2]);
            }
        };
        write!(f, "{code}")
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

impl Parse for MediaHeaderAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != MDHD {
            return Err(ParseError::new_unexpected_atom(atom_type, MDHD));
        }
        parse_mdhd_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
}

fn parse_mdhd_data<R: Read>(mut reader: R) -> Result<MediaHeaderAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    let (creation_time, modification_time, timescale, duration) = match version {
        0 => {
            let mut buf4 = [0u8; 4];

            // Creation time (32-bit)
            reader.read_exact(&mut buf4).context("read creation_time")?;
            let creation_time = u64::from(u32::from_be_bytes(buf4));

            // Modification time (32-bit)
            reader
                .read_exact(&mut buf4)
                .context("read modification_time")?;
            let modification_time = u64::from(u32::from_be_bytes(buf4));

            // Timescale (32-bit)
            reader.read_exact(&mut buf4).context("read timescale")?;
            let timescale = u32::from_be_bytes(buf4);

            // Duration (32-bit)
            reader.read_exact(&mut buf4).context("read duration")?;
            let duration = u64::from(u32::from_be_bytes(buf4));

            (creation_time, modification_time, timescale, duration)
        }
        1 => {
            let mut buf4 = [0u8; 4];
            let mut buf8 = [0u8; 8];

            // Creation time (64-bit)
            reader.read_exact(&mut buf8).context("read creation_time")?;
            let creation_time = u64::from_be_bytes(buf8);

            // Modification time (64-bit)
            reader
                .read_exact(&mut buf8)
                .context("read modification_time")?;
            let modification_time = u64::from_be_bytes(buf8);

            // Timescale (32-bit)
            reader.read_exact(&mut buf4).context("read timescale")?;
            let timescale = u32::from_be_bytes(buf4);

            // Duration (64-bit)
            reader.read_exact(&mut buf8).context("read duration")?;
            let duration = u64::from_be_bytes(buf8);

            (creation_time, modification_time, timescale, duration)
        }
        v => return Err(anyhow!("unsupported version {v}")),
    };

    // Read language (2 bytes) + pre_defined (2 bytes)
    let mut lang_pre = [0u8; 4];
    reader
        .read_exact(&mut lang_pre)
        .context("read language and pre_defined")?;

    // Language is packed in first 2 bytes as 3 x 5-bit values
    let language = LanguageCode::from([lang_pre[0], lang_pre[1], 0]);
    let pre_defined = u16::from_be_bytes([lang_pre[2], lang_pre[3]]);

    // Validate timescale
    if timescale == 0 {
        return Err(anyhow!("Invalid timescale: cannot be zero"));
    }

    Ok(MediaHeaderAtom {
        version,
        flags,
        creation_time,
        modification_time,
        timescale,
        duration,
        language,
        pre_defined,
    })
}

impl SerializeAtom for MediaHeaderAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*MDHD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Determine version based on whether values fit in 32-bit
        let needs_64_bit = self.creation_time > u64::from(u32::MAX)
            || self.modification_time > u64::from(u32::MAX)
            || self.duration > u64::from(u32::MAX);

        let version = i32::from(needs_64_bit);

        // Version and flags (4 bytes)
        let version_flags = (version as u32) << 24
            | u32::from(self.flags[0]) << 16
            | u32::from(self.flags[1]) << 8
            | u32::from(self.flags[2]);
        data.extend_from_slice(&version_flags.to_be_bytes());

        match version {
            0 => {
                // Creation time (32-bit)
                data.extend_from_slice(
                    &(u32::try_from(self.creation_time).expect("creation_time should fit in u32"))
                        .to_be_bytes(),
                );
                // Modification time (32-bit)
                data.extend_from_slice(
                    &(u32::try_from(self.modification_time)
                        .expect("modification_time should fit in u32"))
                    .to_be_bytes(),
                );
                // Timescale (32-bit)
                data.extend_from_slice(&self.timescale.to_be_bytes());
                // Duration (32-bit)
                data.extend_from_slice(
                    &(u32::try_from(self.duration).expect("duration should fit in u32"))
                        .to_be_bytes(),
                );
            }
            1 => {
                // Creation time (64-bit)
                data.extend_from_slice(&self.creation_time.to_be_bytes());
                // Modification time (64-bit)
                data.extend_from_slice(&self.modification_time.to_be_bytes());
                // Timescale (32-bit)
                data.extend_from_slice(&self.timescale.to_be_bytes());
                // Duration (64-bit)
                data.extend_from_slice(&self.duration.to_be_bytes());
            }
            _ => {} // Should not happen due to validation during parsing
        }

        // Language (2 bytes) + pre_defined (2 bytes)
        let lang_bytes = self.language.serialize();
        data.extend_from_slice(&lang_bytes);
        data.extend_from_slice(&self.pre_defined.to_be_bytes());

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available mdhd test data files
    #[test]
    fn test_mdhd_roundtrip() {
        test_atom_roundtrip_sync::<MediaHeaderAtom>(MDHD);
    }
}
