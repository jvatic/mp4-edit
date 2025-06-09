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

    /// Check if the audio is balanced (center)
    pub fn is_balanced(&self) -> bool {
        self.balance.abs() < f32::EPSILON
    }

    /// Check if the audio favors the left channel
    pub fn favors_left(&self) -> bool {
        self.balance < -f32::EPSILON
    }

    /// Check if the audio favors the right channel
    pub fn favors_right(&self) -> bool {
        self.balance > f32::EPSILON
    }

    /// Get the balance as a percentage (-100.0 to +100.0)
    /// -100.0 = full left, 0.0 = center, +100.0 = full right
    pub fn balance_percentage(&self) -> f32 {
        self.balance * 100.0
    }

    /// Get a human-readable description of the balance
    pub fn balance_description(&self) -> String {
        if self.is_balanced() {
            "Center".to_string()
        } else if self.favors_left() {
            format!("Left {:.1}%", self.balance.abs() * 100.0)
        } else {
            format!("Right {:.1}%", self.balance * 100.0)
        }
    }

    /// Check if the reserved field is properly set to zero
    pub fn is_valid_reserved(&self) -> bool {
        self.reserved == 0
    }

    /// Create a new SoundMediaHeader with default values (balanced audio)
    pub fn new() -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            balance: 0.0,
            reserved: 0,
        }
    }

    /// Create a new SoundMediaHeader with specified balance
    /// balance: -1.0 (full left) to +1.0 (full right), 0.0 = center
    pub fn new_with_balance(balance: f32) -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            balance: balance.clamp(-1.0, 1.0),
            reserved: 0,
        }
    }
}

impl Default for SoundMediaHeaderAtom {
    fn default() -> Self {
        Self::new()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_sound_media_header() {
        let header = SoundMediaHeaderAtom::default();
        assert_eq!(header.version, 0);
        assert_eq!(header.flags, [0, 0, 0]);
        assert!(header.is_balanced());
        assert_eq!(header.balance, 0.0);
        assert_eq!(header.reserved, 0);
        assert!(header.is_valid_reserved());
    }

    #[test]
    fn test_balance_clamping() {
        let over_right = SoundMediaHeaderAtom::new_with_balance(2.0);
        assert_eq!(over_right.balance, 1.0);

        let over_left = SoundMediaHeaderAtom::new_with_balance(-2.0);
        assert_eq!(over_left.balance, -1.0);
    }

    #[test]
    fn test_balance_descriptions() {
        let test_cases = vec![
            (0.0, "Center"),
            (-0.5, "Left 50.0%"),
            (0.75, "Right 75.0%"),
            (-1.0, "Left 100.0%"),
            (1.0, "Right 100.0%"),
        ];

        for (balance, expected) in test_cases {
            let header = SoundMediaHeaderAtom::new_with_balance(balance);
            assert_eq!(header.balance_description(), expected);
        }
    }

    #[test]
    fn test_validation() {
        let valid_header = SoundMediaHeaderAtom {
            version: 0,
            flags: [0, 0, 0],
            balance: 0.5,
            reserved: 0,
        };
        assert!(valid_header.is_valid_reserved());

        let invalid_reserved = SoundMediaHeaderAtom {
            version: 0,
            flags: [0, 0, 0],
            balance: 0.0,
            reserved: 1,
        };
        assert!(!invalid_reserved.is_valid_reserved());
    }

    #[test]
    fn test_edge_cases() {
        // Test very small values near zero
        let tiny_left = SoundMediaHeaderAtom::new_with_balance(-0.001);
        assert!(tiny_left.favors_left());
        assert!(!tiny_left.is_balanced());

        let tiny_right = SoundMediaHeaderAtom::new_with_balance(0.001);
        assert!(tiny_right.favors_right());
        assert!(!tiny_right.is_balanced());

        // Test exactly at epsilon boundary
        let epsilon_value = SoundMediaHeaderAtom::new_with_balance(f32::EPSILON / 2.0);
        assert!(epsilon_value.is_balanced());
    }
}
