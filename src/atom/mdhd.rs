use anyhow::{anyhow, Context};
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::atom::util::parse_fixed_size_atom;

pub const MDHD: &[u8; 4] = b"mdhd";

/// Language code (ISO 639-2/T language code)
#[derive(Clone)]
pub struct LanguageCode([u8; 3]);

impl fmt::Debug for LanguageCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LanguageCode({})", self)
    }
}

impl fmt::Display for LanguageCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let packed = u16::from_be_bytes([self.0[0], self.0[1]]);
        let char1 = (((packed >> 10) & 0x1F) + 0x60) as u8 as char;
        let char2 = (((packed >> 5) & 0x1F) + 0x60) as u8 as char;
        let char3 = ((packed & 0x1F) + 0x60) as u8 as char;
        write!(f, "{}{}{}", char1, char2, char3)
    }
}

#[derive(Debug, Clone)]
pub struct MediaHeaderAtom {
    /// Version of the mdhd atom format (0 or 1)
    pub version: u8,
    /// Flags for the mdhd atom (usually all zeros)
    pub flags: [u8; 3],
    /// Creation time (seconds since midnight, Jan. 1, 1904, UTC)
    pub creation_time: u64,
    /// Modification time (seconds since midnight, Jan. 1, 1904, UTC)
    pub modification_time: u64,
    /// Media timescale (number of time units per second)
    pub timescale: u32,
    /// Duration of media (in timescale units)
    pub duration: u64,
    /// Language code (ISO 639-2/T language code)
    pub language: LanguageCode,
    /// Pre-defined value (should be 0)
    pub pre_defined: u16,
}

impl MediaHeaderAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_mdhd_atom(reader)
    }

    /// Get the duration in seconds
    pub fn duration_seconds(&self) -> f64 {
        if self.timescale == 0 {
            0.0
        } else {
            self.duration as f64 / self.timescale as f64
        }
    }

    /// Get the language as a string (ISO 639-2/T format)
    pub fn language_string(&self) -> String {
        format!("{}", self.language)
    }

    /// Check if the media header has valid timescale
    pub fn has_valid_timescale(&self) -> bool {
        self.timescale > 0
    }

    /// Check if creation and modification times are valid (non-zero)
    pub fn has_valid_timestamps(&self) -> bool {
        self.creation_time > 0 && self.modification_time > 0
    }

    /// Convert Mac epoch time (1904) to Unix epoch time (1970)
    pub fn creation_time_unix(&self) -> Option<u64> {
        // Mac epoch starts Jan 1, 1904; Unix epoch starts Jan 1, 1970
        // Difference is 66 years = 2,082,844,800 seconds
        const MAC_TO_UNIX_OFFSET: u64 = 2_082_844_800;

        if self.creation_time > MAC_TO_UNIX_OFFSET {
            Some(self.creation_time - MAC_TO_UNIX_OFFSET)
        } else {
            None
        }
    }

    /// Convert Mac epoch time (1904) to Unix epoch time (1970)
    pub fn modification_time_unix(&self) -> Option<u64> {
        const MAC_TO_UNIX_OFFSET: u64 = 2_082_844_800;

        if self.modification_time > MAC_TO_UNIX_OFFSET {
            Some(self.modification_time - MAC_TO_UNIX_OFFSET)
        } else {
            None
        }
    }
}

impl TryFrom<&[u8]> for MediaHeaderAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_mdhd_atom(reader)
    }
}

fn parse_mdhd_atom<R: Read>(reader: R) -> Result<MediaHeaderAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != MDHD {
        return Err(anyhow!("Invalid atom type: {}", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_mdhd_data(&mut cursor)
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
            let creation_time = u32::from_be_bytes(buf4) as u64;

            // Modification time (32-bit)
            reader
                .read_exact(&mut buf4)
                .context("read modification_time")?;
            let modification_time = u32::from_be_bytes(buf4) as u64;

            // Timescale (32-bit)
            reader.read_exact(&mut buf4).context("read timescale")?;
            let timescale = u32::from_be_bytes(buf4);

            // Duration (32-bit)
            reader.read_exact(&mut buf4).context("read duration")?;
            let duration = u32::from_be_bytes(buf4) as u64;

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
    let language = LanguageCode([lang_pre[0], lang_pre[1], 0]); // Third byte is derived from first two
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_seconds() {
        let header = MediaHeaderAtom {
            version: 0,
            flags: [0, 0, 0],
            creation_time: 0,
            modification_time: 0,
            timescale: 1000,
            duration: 5000,
            language: LanguageCode([0, 0, 0]),
            pre_defined: 0,
        };

        assert_eq!(header.duration_seconds(), 5.0);
    }

    #[test]
    fn test_zero_timescale() {
        let header = MediaHeaderAtom {
            version: 0,
            flags: [0, 0, 0],
            creation_time: 0,
            modification_time: 0,
            timescale: 0,
            duration: 5000,
            language: LanguageCode([0, 0, 0]),
            pre_defined: 0,
        };

        assert_eq!(header.duration_seconds(), 0.0);
        assert!(!header.has_valid_timescale());
    }

    #[test]
    fn test_language_string() {
        // Example: "eng" encoded as packed 5-bit values
        // 'e' = 5, 'n' = 14, 'g' = 7
        // Packed: (5 << 10) | (14 << 5) | 7 = 5120 + 448 + 7 = 5575
        let packed_bytes = 5575u16.to_be_bytes();

        let header = MediaHeaderAtom {
            version: 0,
            flags: [0, 0, 0],
            creation_time: 0,
            modification_time: 0,
            timescale: 1000,
            duration: 0,
            language: LanguageCode([packed_bytes[0], packed_bytes[1], 0]),
            pre_defined: 0,
        };

        assert_eq!(header.language_string(), "eng");
    }

    #[test]
    fn test_unix_timestamp_conversion() {
        const MAC_TO_UNIX_OFFSET: u64 = 2_082_844_800;

        let header = MediaHeaderAtom {
            version: 0,
            flags: [0, 0, 0],
            creation_time: MAC_TO_UNIX_OFFSET + 1000, // 1000 seconds after Unix epoch
            modification_time: MAC_TO_UNIX_OFFSET + 2000, // 2000 seconds after Unix epoch
            timescale: 1000,
            duration: 0,
            language: LanguageCode([0, 0, 0]),
            pre_defined: 0,
        };

        assert_eq!(header.creation_time_unix(), Some(1000));
        assert_eq!(header.modification_time_unix(), Some(2000));
        assert!(header.has_valid_timestamps());
    }
}
