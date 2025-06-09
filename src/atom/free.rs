use anyhow::anyhow;
use futures_io::AsyncRead;
use std::fmt;

use crate::{
    atom::util::{parse_fixed_size_atom, DebugEllipsis, FourCC},
    parser::Parse,
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

impl Parse for FreeAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(reader: R) -> Result<Self, anyhow::Error> {
        let (atom_type, data) = parse_fixed_size_atom(reader).await?;

        // Verify this is a free or skip atom
        if atom_type != FREE && atom_type != SKIP {
            return Err(anyhow!(
                "Invalid atom type: {} (expected 'free' or 'skip')",
                atom_type
            ));
        }

        Ok(FreeAtom {
            atom_type,
            data_size: data.len(),
            data,
        })
    }
}
