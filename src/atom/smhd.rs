use anyhow::{anyhow, Context};
use std::io::{Cursor, Read};

use crate::atom::util::parse_fixed_size_atom;

pub const SMHD: &[u8; 4] = b"smhd";

#[derive(Debug, Clone)]
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

impl SoundMediaHeaderAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_smhd_atom(reader)
    }
}

impl TryFrom<&[u8]> for SoundMediaHeaderAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_smhd_atom(reader)
    }
}

fn parse_smhd_atom<R: Read>(reader: R) -> Result<SoundMediaHeaderAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != SMHD {
        return Err(anyhow!("Invalid atom type: {}", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_smhd_data(&mut cursor)
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
    let balance = (balance_fixed as f32) / 256.0;

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
