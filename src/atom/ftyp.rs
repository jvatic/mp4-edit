use anyhow::{anyhow, Context};
use std::io::{Cursor, Read};

use crate::atom::util::{parse_fixed_size_atom, FourCC};

pub const FTYP: &[u8; 4] = b"ftyp";

/// File Type Atom (ftyp) - ISO/IEC 14496-12
/// This atom identifies the specifications to which this file complies.
#[derive(Debug, Clone)]
pub struct FileTypeAtom {
    /// Major brand - identifies the 'best use' of the file
    pub major_brand: FourCC,
    /// Minor version - an informative integer for the minor version of the major brand
    pub minor_version: u32,
    /// Compatible brands - a list of brands compatible with this file
    pub compatible_brands: Vec<FourCC>,
}

impl FileTypeAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_ftyp_atom(reader)
    }
}

impl TryFrom<&[u8]> for FileTypeAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_ftyp_atom(reader)
    }
}

fn parse_ftyp_atom<R: Read>(reader: R) -> Result<FileTypeAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != FTYP {
        return Err(anyhow!(
            "Invalid atom type: expected ftyp, got {}",
            atom_type
        ));
    }

    let mut cursor = Cursor::new(data);
    parse_ftyp_data(&mut cursor)
}

fn parse_ftyp_data<R: Read>(mut reader: R) -> Result<FileTypeAtom, anyhow::Error> {
    // Minimum size check: major_brand (4) + minor_version (4) = 8 bytes
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).context("reading ftyp data")?;

    if buf.len() < 8 {
        return Err(anyhow!("ftyp atom too small: {} bytes", buf.len()));
    }

    // Major brand (4 bytes)
    let major_brand = FourCC([buf[0], buf[1], buf[2], buf[3]]);

    // Minor version (4 bytes)
    let minor_version = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);

    // Compatible brands (remaining bytes, must be multiple of 4)
    let remaining_bytes = &buf[8..];
    if remaining_bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "Invalid ftyp atom: compatible brands section has {} bytes, must be multiple of 4",
            remaining_bytes.len()
        ));
    }

    let mut compatible_brands = Vec::new();
    for chunk in remaining_bytes.chunks_exact(4) {
        let brand = FourCC([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if brand == &[0u8; 4] {
            continue;
        }
        compatible_brands.push(brand);
    }

    Ok(FileTypeAtom {
        major_brand,
        minor_version,
        compatible_brands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_size() {
        // Test with data too small (less than 8 bytes)
        let mut data = Vec::new();
        data.extend_from_slice(&12u32.to_be_bytes()); // 12 bytes total
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(b"mp41"); // Only 4 bytes of data, need 8 minimum

        let result = FileTypeAtom::parse(Cursor::new(data));
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_compatible_brands_size() {
        // Test with compatible brands not multiple of 4
        let mut data = Vec::new();
        data.extend_from_slice(&15u32.to_be_bytes()); // 15 bytes total
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(b"mp41"); // Major brand
        data.extend_from_slice(&0u32.to_be_bytes()); // Minor version
        data.extend_from_slice(b"mp4"); // Only 3 bytes, should be 4

        let result = FileTypeAtom::parse(Cursor::new(data));
        assert!(result.is_err());
    }
}
