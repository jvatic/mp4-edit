use futures_io::AsyncRead;
use std::fmt;

use crate::{
    atom::{
        util::{read_to_end, DebugList, DebugUpperHex},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const FREE: &[u8; 4] = b"free";
pub const SKIP: &[u8; 4] = b"skip";

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

impl ParseAtom for FreeAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != FREE && atom_type != SKIP {
            return Err(ParseError::new_unexpected_atom_oneof(
                atom_type,
                vec![FREE, SKIP],
            ));
        }

        let data = read_to_end(reader).await?;
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
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available free test data files
    #[test]
    fn test_free_roundtrip() {
        test_atom_roundtrip_sync::<FreeAtom>(FREE);
    }
}
