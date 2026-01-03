use std::fmt;

use crate::{
    atom::{
        util::{parser::rest_vec, DebugList, DebugUpperHex},
        FourCC,
    },
    parser::ParseAtomData,
    writer::SerializeAtom,
    ParseError,
};

pub const FREE: FourCC = FourCC::new(b"free");
pub const SKIP: FourCC = FourCC::new(b"skip");

#[derive(Clone)]
pub struct FreeAtom {
    /// The atom type (either 'free' or 'skip')
    pub atom_type: FourCC,
    /// Size of the free space data
    pub data_size: usize,
    /// The actual free space data (usually ignored)
    pub data: Vec<u8>,
}

impl fmt::Debug for FreeAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreeAtom")
            .field("atom_type", &self.atom_type)
            .field("data_size", &self.data_size)
            .field(
                "data",
                &DebugList::new(self.data.iter().map(DebugUpperHex), 10),
            )
            .finish()
    }
}

impl ParseAtomData for FreeAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, FREE, SKIP);

        use crate::atom::util::parser::stream;
        use winnow::Parser;

        let data = rest_vec.parse(stream(input))?;
        Ok(FreeAtom {
            atom_type,
            data_size: data.len(),
            data,
        })
    }
}

impl SerializeAtom for FreeAtom {
    fn atom_type(&self) -> FourCC {
        self.atom_type
    }

    fn into_body_bytes(self) -> Vec<u8> {
        if self.data.is_empty() {
            vec![0u8; self.data_size]
        } else {
            self.data
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available free test data files
    #[test]
    fn test_free_roundtrip() {
        test_atom_roundtrip::<FreeAtom>(FREE);
    }
}
