use anyhow::{anyhow, Context};
use std::io::{Cursor, Read};

use crate::atom::util::parse_fixed_size_atom;

pub const GMHD: &[u8; 4] = b"gmhd";

#[derive(Debug, Clone, PartialEq)]
pub struct GenericMediaHeaderAtom {
    /// Version of the gmhd atom format (0)
    pub version: u8,
    /// Flags for the gmhd atom (usually all zeros)
    pub flags: [u8; 3],
    /// Graphics mode for compositing (usually 0 = copy)
    pub graphics_mode: u16,
    /// RGB color values for graphics mode (each component is 16-bit)
    pub opcolor: [u16; 3], // [red, green, blue]
}

impl GenericMediaHeaderAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_gmhd_atom(reader)
    }

    /// Check if using copy graphics mode (standard)
    pub fn is_copy_mode(&self) -> bool {
        self.graphics_mode == 0
    }

    /// Check if the opcolor is black (all zeros)
    pub fn is_opcolor_black(&self) -> bool {
        self.opcolor == [0, 0, 0]
    }

    /// Get the red component of opcolor as a normalized float (0.0 to 1.0)
    pub fn red_normalized(&self) -> f32 {
        self.opcolor[0] as f32 / 65535.0
    }

    /// Get the green component of opcolor as a normalized float (0.0 to 1.0)
    pub fn green_normalized(&self) -> f32 {
        self.opcolor[1] as f32 / 65535.0
    }

    /// Get the blue component of opcolor as a normalized float (0.0 to 1.0)
    pub fn blue_normalized(&self) -> f32 {
        self.opcolor[2] as f32 / 65535.0
    }

    /// Get the opcolor as normalized RGB values (0.0 to 1.0 each)
    pub fn opcolor_normalized(&self) -> [f32; 3] {
        [
            self.red_normalized(),
            self.green_normalized(),
            self.blue_normalized(),
        ]
    }

    /// Get a hex string representation of the opcolor (e.g., "#FF0000")
    pub fn opcolor_hex(&self) -> String {
        format!(
            "#{:02X}{:02X}{:02X}",
            (self.opcolor[0] >> 8) as u8,
            (self.opcolor[1] >> 8) as u8,
            (self.opcolor[2] >> 8) as u8
        )
    }

    /// Get a human-readable description of the graphics mode
    pub fn graphics_mode_description(&self) -> &'static str {
        match self.graphics_mode {
            0 => "Copy (ditherCopy)",
            1 => "Dither",
            2 => "Blend",
            3 => "Transparent",
            4 => "Straight alpha",
            5 => "Premul white alpha",
            6 => "Premul black alpha",
            7 => "Straight alpha blend",
            8 => "Composition (dither copy)",
            _ => "Unknown/Custom",
        }
    }

    /// Create a new GenericMediaHeader with default values
    pub fn new() -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            graphics_mode: 0,   // Copy mode
            opcolor: [0, 0, 0], // Black
        }
    }

    /// Create a new GenericMediaHeader with specified graphics mode and color
    pub fn new_with_mode_and_color(graphics_mode: u16, red: u16, green: u16, blue: u16) -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            graphics_mode,
            opcolor: [red, green, blue],
        }
    }

    /// Create a new GenericMediaHeader with RGB color from 8-bit values
    pub fn new_with_rgb8(graphics_mode: u16, red: u8, green: u8, blue: u8) -> Self {
        Self::new_with_mode_and_color(
            graphics_mode,
            (red as u16) << 8,
            (green as u16) << 8,
            (blue as u16) << 8,
        )
    }

    /// Check if this atom follows standard conventions
    pub fn is_standard(&self) -> bool {
        self.version == 0 && self.flags == [0, 0, 0] && self.is_copy_mode()
    }

    /// Validate the atom structure
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        if self.version != 0 {
            return Err(anyhow!("Invalid version: {} (expected 0)", self.version));
        }

        // Flags are typically all zeros but not strictly required
        // Graphics mode can be any value, but warn about unknown modes
        if self.graphics_mode > 8 {
            // This is just a warning case, not an error
        }

        Ok(())
    }
}

impl Default for GenericMediaHeaderAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl TryFrom<&[u8]> for GenericMediaHeaderAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_gmhd_atom(reader)
    }
}

fn parse_gmhd_atom<R: Read>(reader: R) -> Result<GenericMediaHeaderAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != GMHD {
        return Err(anyhow!("Invalid atom type: {}", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_gmhd_data(&mut cursor)
}

fn parse_gmhd_data<R: Read>(mut reader: R) -> Result<GenericMediaHeaderAtom, anyhow::Error> {
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

    // Read graphics mode (2 bytes)
    let mut graphics_mode_buf = [0u8; 2];
    reader
        .read_exact(&mut graphics_mode_buf)
        .context("read graphics mode")?;
    let graphics_mode = u16::from_be_bytes(graphics_mode_buf);

    // Read opcolor RGB values (6 bytes total, 2 bytes each)
    let mut opcolor = [0u16; 3];
    for i in 0..3 {
        let mut color_buf = [0u8; 2];
        reader
            .read_exact(&mut color_buf)
            .context(format!("read opcolor component {}", i))?;
        opcolor[i] = u16::from_be_bytes(color_buf);
    }

    Ok(GenericMediaHeaderAtom {
        version,
        flags,
        graphics_mode,
        opcolor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_generic_media_header() {
        let header = GenericMediaHeaderAtom::default();
        assert_eq!(header.version, 0);
        assert_eq!(header.flags, [0, 0, 0]);
        assert_eq!(header.graphics_mode, 0);
        assert_eq!(header.opcolor, [0, 0, 0]);
        assert!(header.is_copy_mode());
        assert!(header.is_opcolor_black());
        assert!(header.is_standard());
    }

    #[test]
    fn test_new_constructors() {
        let custom = GenericMediaHeaderAtom::new_with_mode_and_color(2, 65535, 32768, 0);
        assert_eq!(custom.version, 0);
        assert_eq!(custom.flags, [0, 0, 0]);
        assert_eq!(custom.graphics_mode, 2);
        assert_eq!(custom.opcolor, [65535, 32768, 0]);
        assert!(!custom.is_copy_mode());
        assert!(!custom.is_opcolor_black());

        let rgb8 = GenericMediaHeaderAtom::new_with_rgb8(1, 255, 128, 64);
        assert_eq!(rgb8.graphics_mode, 1);
        assert_eq!(rgb8.opcolor, [65280, 32768, 16384]);
    }

    #[test]
    fn test_hex_color_representation() {
        let red = GenericMediaHeaderAtom::new_with_rgb8(0, 255, 0, 0);
        assert_eq!(red.opcolor_hex(), "#FF0000");

        let green = GenericMediaHeaderAtom::new_with_rgb8(0, 0, 255, 0);
        assert_eq!(green.opcolor_hex(), "#00FF00");

        let blue = GenericMediaHeaderAtom::new_with_rgb8(0, 0, 0, 255);
        assert_eq!(blue.opcolor_hex(), "#0000FF");

        let white = GenericMediaHeaderAtom::new_with_rgb8(0, 255, 255, 255);
        assert_eq!(white.opcolor_hex(), "#FFFFFF");

        let black = GenericMediaHeaderAtom::new_with_rgb8(0, 0, 0, 0);
        assert_eq!(black.opcolor_hex(), "#000000");
    }

    #[test]
    fn test_graphics_mode_descriptions() {
        let test_cases = vec![
            (0, "Copy (ditherCopy)"),
            (1, "Dither"),
            (2, "Blend"),
            (3, "Transparent"),
            (4, "Straight alpha"),
            (5, "Premul white alpha"),
            (6, "Premul black alpha"),
            (7, "Straight alpha blend"),
            (8, "Composition (dither copy)"),
            (99, "Unknown/Custom"),
        ];

        for (mode, expected) in test_cases {
            let header = GenericMediaHeaderAtom::new_with_mode_and_color(mode, 0, 0, 0);
            assert_eq!(header.graphics_mode_description(), expected);
        }
    }

    #[test]
    fn test_validation() {
        let valid_header = GenericMediaHeaderAtom::default();
        assert!(valid_header.validate().is_ok());

        let invalid_version = GenericMediaHeaderAtom {
            version: 1,
            flags: [0, 0, 0],
            graphics_mode: 0,
            opcolor: [0, 0, 0],
        };
        assert!(invalid_version.validate().is_err());
    }

    #[test]
    fn test_standard_compliance() {
        let standard = GenericMediaHeaderAtom::default();
        assert!(standard.is_standard());

        let non_standard_mode = GenericMediaHeaderAtom::new_with_mode_and_color(1, 0, 0, 0);
        assert!(!non_standard_mode.is_standard());

        let non_standard_flags = GenericMediaHeaderAtom {
            version: 0,
            flags: [1, 0, 0],
            graphics_mode: 0,
            opcolor: [0, 0, 0],
        };
        assert!(!non_standard_flags.is_standard());
    }

    #[test]
    fn test_parse_gmhd_data() {
        // Create test data for a gmhd atom
        let mut data = Vec::new();
        data.extend_from_slice(&[0u8]); // version
        data.extend_from_slice(&[0u8, 0u8, 0u8]); // flags
        data.extend_from_slice(&2u16.to_be_bytes()); // graphics_mode = 2 (blend)
        data.extend_from_slice(&65535u16.to_be_bytes()); // red = 65535
        data.extend_from_slice(&32768u16.to_be_bytes()); // green = 32768
        data.extend_from_slice(&16384u16.to_be_bytes()); // blue = 16384

        let result = parse_gmhd_data(Cursor::new(data)).unwrap();
        assert_eq!(result.version, 0);
        assert_eq!(result.flags, [0, 0, 0]);
        assert_eq!(result.graphics_mode, 2);
        assert_eq!(result.opcolor, [65535, 32768, 16384]);
        assert_eq!(result.graphics_mode_description(), "Blend");
    }

    #[test]
    fn test_color_edge_cases() {
        // Test maximum values
        let max_color = GenericMediaHeaderAtom::new_with_mode_and_color(0, 65535, 65535, 65535);
        assert_eq!(max_color.opcolor_hex(), "#FFFFFF");
        assert!((max_color.red_normalized() - 1.0).abs() < f32::EPSILON);

        // Test minimum values
        let min_color = GenericMediaHeaderAtom::new_with_mode_and_color(0, 0, 0, 0);
        assert_eq!(min_color.opcolor_hex(), "#000000");
        assert!(min_color.red_normalized().abs() < f32::EPSILON);

        // Test mid-range values
        let mid_color = GenericMediaHeaderAtom::new_with_mode_and_color(0, 32768, 32768, 32768);
        assert_eq!(mid_color.opcolor_hex(), "#808080");
        assert!((mid_color.red_normalized() - 0.5).abs() < 0.01);
    }
}
