use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::{fmt, io::Read};

use crate::{
    atom::{util::async_to_sync_read, FourCC},
    parser::Parse,
    writer::SerializeAtom,
};

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

impl Parse for MediaHeaderAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, anyhow::Error> {
        if atom_type != MDHD {
            return Err(anyhow!("Invalid atom type: {}", atom_type));
        }
        parse_mdhd_data(async_to_sync_read(reader).await?)
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

impl SerializeAtom for MediaHeaderAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*MDHD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Determine version based on whether values fit in 32-bit
        let needs_64_bit = self.creation_time > u32::MAX as u64
            || self.modification_time > u32::MAX as u64
            || self.duration > u32::MAX as u64;

        let version = if needs_64_bit { 1 } else { 0 };

        // Version and flags (4 bytes)
        let version_flags = (version as u32) << 24
            | (self.flags[0] as u32) << 16
            | (self.flags[1] as u32) << 8
            | (self.flags[2] as u32);
        data.extend_from_slice(&version_flags.to_be_bytes());

        match version {
            0 => {
                // Creation time (32-bit)
                data.extend_from_slice(&(self.creation_time as u32).to_be_bytes());
                // Modification time (32-bit)
                data.extend_from_slice(&(self.modification_time as u32).to_be_bytes());
                // Timescale (32-bit)
                data.extend_from_slice(&self.timescale.to_be_bytes());
                // Duration (32-bit)
                data.extend_from_slice(&(self.duration as u32).to_be_bytes());
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
        data.extend_from_slice(&self.language.0[0..2]);
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
    fn test_ftyp_roundtrip() {
        test_atom_roundtrip_sync::<MediaHeaderAtom>(MDHD);
    }
}
