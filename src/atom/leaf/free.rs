use futures_io::AsyncRead;
use futures_util::AsyncReadExt;
use std::fmt;

use crate::{
    atom::{util::DebugEllipsis, FourCC},
    parser::{ParseAtom, ParseErrorKind},
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
            .field("data", &DebugEllipsis(Some(self.data.len())))
            .finish()
    }
}

impl ParseAtom for FreeAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        mut reader: R,
    ) -> Result<Self, ParseError> {
        // Verify this is a free or skip atom
        if atom_type != FREE {
            return Err(ParseError::new_unexpected_atom(atom_type, FREE));
        }

        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .await
            .map_err(|err| ParseError {
                kind: ParseErrorKind::Io,
                location: None,
                source: Some(Box::new(err)),
            })?;
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
        // Free atoms typically contain zero bytes or preserved data
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
