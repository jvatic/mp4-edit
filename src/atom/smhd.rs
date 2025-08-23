use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::io::Read;

use crate::{
    atom::{util::async_to_sync_read, FourCC},
    parser::Parse,
    writer::SerializeAtom,
    ParseError,
};

pub const SMHD: &[u8; 4] = b"smhd";

#[derive(Debug, Clone, Default)]
pub struct SoundMediaHeaderAtom {
    /// Version of the smhd atom format (0)
    pub version: u8,
    /// Flags for the smhd atom (usually all zeros)
    pub flags: [u8; 3],
    /// Audio balance (fixed-point 8.8 format, 0.0 = center)
    /// Negative values favor left channel, positive favor right
    pub balance: f32,
    /// Reserved field (must be 0)
    pub reserved: u16,
}

impl Parse for SoundMediaHeaderAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != SMHD {
            return Err(ParseError::new_unexpected_atom(atom_type, SMHD));
        }
        parse_smhd_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
}

fn parse_smhd_data<R: Read>(mut reader: R) -> Result<SoundMediaHeaderAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Validate version
    if version != 0 {
        return Err(anyhow!("unsupported version {}", version));
    }

    // Read balance (2 bytes, fixed-point 8.8 format)
    let mut balance_buf = [0u8; 2];
    reader
        .read_exact(&mut balance_buf)
        .context("read balance")?;
    let balance_fixed = i16::from_be_bytes(balance_buf);

    // Convert from fixed-point 8.8 to float
    // In 8.8 format, the value is multiplied by 256
    let balance = f32::from(balance_fixed) / 256.0;

    // Read reserved field (2 bytes)
    let mut reserved_buf = [0u8; 2];
    reader
        .read_exact(&mut reserved_buf)
        .context("read reserved")?;
    let reserved = u16::from_be_bytes(reserved_buf);

    // Validate that the balance is within reasonable bounds
    if !(-1.0..=1.0).contains(&balance) {
        return Err(anyhow!("Invalid balance value: {}", balance));
    }

    Ok(SoundMediaHeaderAtom {
        version,
        flags,
        balance,
        reserved,
    })
}

impl SerializeAtom for SoundMediaHeaderAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*SMHD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version and flags (4 bytes)
        let version_flags = u32::from(self.version) << 24
            | u32::from(self.flags[0]) << 16
            | u32::from(self.flags[1]) << 8
            | u32::from(self.flags[2]);
        data.extend_from_slice(&version_flags.to_be_bytes());

        // Balance (fixed-point 8.8 format, 2 bytes)
        let balance_fixed = (self.balance * 256.0) as i16;
        data.extend_from_slice(&balance_fixed.to_be_bytes());

        // Reserved (2 bytes)
        data.extend_from_slice(&self.reserved.to_be_bytes());

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available smhd test data files
    #[test]
    fn test_smhd_roundtrip() {
        test_atom_roundtrip_sync::<SoundMediaHeaderAtom>(SMHD);
    }
}
