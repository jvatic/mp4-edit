use anyhow::anyhow;
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis, FourCC};

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

impl FreeAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_free_atom(reader)
    }
}

#[derive(Debug, Clone)]
pub struct FreeSpaceSummary {
    pub atom_type: String,
    pub size: usize,
    pub is_zeroed: bool,
    pub appears_random: bool,
    pub first_bytes: Option<Vec<u8>>,
}

impl TryFrom<&[u8]> for FreeAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_free_atom(reader)
    }
}

fn parse_free_atom<R: Read>(reader: R) -> Result<FreeAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;

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
