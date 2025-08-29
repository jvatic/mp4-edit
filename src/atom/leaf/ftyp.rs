use anyhow::{anyhow, Context};
use bon::Builder;
use futures_io::AsyncRead;
use std::io::Read;

use crate::{
    atom::{atom_ref, util::async_to_sync_read, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    AtomData, ParseError,
};

pub const FTYP: &[u8; 4] = b"ftyp";

#[derive(Debug, Clone, Copy)]
pub struct FtypAtomRef<'a>(pub(crate) atom_ref::AtomRef<'a>);

impl<'a> FtypAtomRef<'a> {
    pub fn data(&self) -> Option<&'a FileTypeAtom> {
        self.0
            .inner()
            .and_then(|ftyp| ftyp.data.as_ref())
            .and_then(|data| match data {
                AtomData::FileType(data) => Some(data),
                _ => None,
            })
    }
}

#[derive(Debug)]
pub struct FtypAtomRefMut<'a>(pub(crate) atom_ref::AtomRefMut<'a>);

impl<'a> FtypAtomRefMut<'a> {
    pub fn as_ref(&self) -> FtypAtomRef<'_> {
        FtypAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> FtypAtomRef<'a> {
        FtypAtomRef(self.0.into_ref())
    }

    pub fn replace(&mut self, data: FileTypeAtom) {
        self.0.atom_mut().data = Some(data.into());
    }
}

/// File Type Atom (ftyp) - ISO/IEC 14496-12
/// This atom identifies the specifications to which this file complies.
#[derive(Debug, Clone, Builder)]
pub struct FileTypeAtom {
    /// Major brand - identifies the 'best use' of the file
    #[builder(into)]
    pub major_brand: FourCC,
    /// Minor version - an informative integer for the minor version of the major brand
    #[builder(default = Default::default())]
    pub minor_version: u32,
    /// Compatible brands - a list of brands compatible with this file
    #[builder(default = vec![major_brand], into)]
    pub compatible_brands: Vec<FourCC>,
}

impl Default for FileTypeAtom {
    fn default() -> Self {
        Self {
            major_brand: FourCC(*b"isom"),
            minor_version: 512,
            compatible_brands: vec![FourCC::from(*b"isom")],
        }
    }
}

impl ParseAtom for FileTypeAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != FTYP {
            return Err(ParseError::new_unexpected_atom(atom_type, FTYP));
        }
        parse_ftyp_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
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

impl SerializeAtom for FileTypeAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*FTYP)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Major brand (4 bytes)
        data.extend_from_slice(&self.major_brand.0);

        // Minor version (4 bytes, big-endian)
        data.extend_from_slice(&self.minor_version.to_be_bytes());

        // Compatible brands (4 bytes each)
        for brand in self.compatible_brands {
            data.extend_from_slice(&brand.0);
        }

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available ftyp test data files
    #[test]
    fn test_ftyp_roundtrip() {
        test_atom_roundtrip_sync::<FileTypeAtom>(FTYP);
    }
}
