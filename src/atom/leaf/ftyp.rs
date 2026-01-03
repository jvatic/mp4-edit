use bon::Builder;

use crate::{
    atom::{atom_ref, FourCC},
    parser::ParseAtomData,
    writer::SerializeAtom,
    AtomData, ParseError,
};

pub const FTYP: FourCC = FourCC::new(b"ftyp");

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

impl ParseAtomData for FileTypeAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, FTYP);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_ftyp_data.parse(stream(input))?)
    }
}

mod parser {
    use winnow::{
        binary::be_u32,
        combinator::{repeat, seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::FileTypeAtom;
    use crate::atom::util::parser::{fourcc, Stream};

    pub fn parse_ftyp_data(input: &mut Stream<'_>) -> ModalResult<FileTypeAtom> {
        trace(
            "ftyp",
            seq!(FileTypeAtom {
                major_brand: fourcc.context(StrContext::Label("major_brand")),
                minor_version: be_u32.context(StrContext::Label("minor_version")),
                compatible_brands: repeat(
                    0..,
                    fourcc.context(StrContext::Label("compatible_brand"))
                ),
            }),
        )
        .parse_next(input)
    }
}

impl SerializeAtom for FileTypeAtom {
    fn atom_type(&self) -> FourCC {
        FTYP
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
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available ftyp test data files
    #[test]
    fn test_ftyp_roundtrip() {
        test_atom_roundtrip::<FileTypeAtom>(FTYP);
    }
}
