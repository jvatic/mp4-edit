use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::io::Read;

use crate::{
    atom::util::{async_to_sync_read, FourCC},
    parser::Parse,
};

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

impl Parse for FileTypeAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, anyhow::Error> {
        if atom_type != FTYP {
            return Err(anyhow!(
                "Invalid atom type: expected ftyp, got {}",
                atom_type
            ));
        }
        parse_ftyp_data(async_to_sync_read(reader).await?)
    }
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

impl From<FileTypeAtom> for Vec<u8> {
    fn from(atom: FileTypeAtom) -> Self {
        let mut data = Vec::new();

        // Major brand (4 bytes)
        data.extend_from_slice(&atom.major_brand.0);

        // Minor version (4 bytes, big-endian)
        data.extend_from_slice(&atom.minor_version.to_be_bytes());

        // Compatible brands (4 bytes each)
        for brand in atom.compatible_brands {
            data.extend_from_slice(&brand.0);
        }

        data
    }
}
