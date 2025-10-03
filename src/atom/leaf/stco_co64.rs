use bon::bon;
use derive_more::{Deref, DerefMut};
use futures_io::AsyncRead;
use std::{
    fmt,
    ops::{Deref, Range},
};

use crate::{
    atom::{
        util::{read_to_end, DebugList, DebugUpperHex},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const STCO: &[u8; 4] = b"stco";
pub const CO64: &[u8; 4] = b"co64";

#[derive(Default, Clone, Deref, DerefMut)]
pub struct ChunkOffsets(Vec<u64>);

impl ChunkOffsets {
    pub fn into_inner(self) -> Vec<u64> {
        self.0
    }

    pub fn inner(&self) -> &[u64] {
        &self.0
    }
}

impl From<Vec<u64>> for ChunkOffsets {
    fn from(value: Vec<u64>) -> Self {
        Self(value)
    }
}

impl FromIterator<u64> for ChunkOffsets {
    fn from_iter<T: IntoIterator<Item = u64>>(iter: T) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl fmt::Debug for ChunkOffsets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&DebugList::new(self.0.iter().map(DebugUpperHex), 10), f)
    }
}

/// Chunk Offset Atom - contains file offsets of chunks
#[derive(Default, Debug, Clone)]
pub struct ChunkOffsetAtom {
    /// Version of the stco atom format (0)
    pub version: u8,
    /// Flags for the stco atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of chunk offsets
    pub chunk_offsets: ChunkOffsets,
    /// Whether this uses 64-bit offsets (co64) or 32-bit (stco)
    pub is_64bit: bool,
}

#[bon]
impl ChunkOffsetAtom {
    #[builder]
    pub fn new(
        #[builder(default = 0)] version: u8,
        #[builder(default = [0u8; 3])] flags: [u8; 3],
        #[builder(with = FromIterator::from_iter)] chunk_offsets: Vec<u64>,
        #[builder(default = false)] is_64bit: bool,
    ) -> Self {
        Self {
            version,
            flags,
            chunk_offsets: chunk_offsets.into(),
            is_64bit,
        }
    }

    /// Returns the total number of chunks
    pub fn chunk_count(&self) -> usize {
        self.chunk_offsets.len()
    }

    /// Removes the specified chunk indices.
    pub(crate) fn remove_chunk_indices(&mut self, chunk_indices_to_remove: &[Range<usize>]) {
        for range in chunk_indices_to_remove.iter().cloned() {
            self.chunk_offsets.drain(range);
        }
    }
}

impl<S: chunk_offset_atom_builder::State> ChunkOffsetAtomBuilder<S> {
    pub fn chunk_offset(
        self,
        chunk_offset: impl Into<u64>,
    ) -> ChunkOffsetAtomBuilder<chunk_offset_atom_builder::SetChunkOffsets<S>>
    where
        S::ChunkOffsets: chunk_offset_atom_builder::IsUnset,
    {
        self.chunk_offsets(vec![chunk_offset.into()])
    }
}

impl ParseAtom for ChunkOffsetAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        let data = read_to_end(reader).await?;
        match atom_type.deref() {
            STCO => parser::parse_stco_data(&data),
            CO64 => parser::parse_co64_data(&data),
            _ => return Err(ParseError::new_unexpected_atom(atom_type, STCO)),
        }
    }
}

impl SerializeAtom for ChunkOffsetAtom {
    fn atom_type(&self) -> FourCC {
        // Use the appropriate atom type based on is_64bit
        if self.is_64bit {
            FourCC(*CO64)
        } else {
            FourCC(*STCO)
        }
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_stco_co64_data(self)
    }
}

mod serializer {
    use crate::atom::{util::serializer::be_u32, ChunkOffsetAtom};

    pub fn serialize_stco_co64_data(atom: ChunkOffsetAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(atom.version);
        data.extend(atom.flags);
        data.extend(be_u32(
            atom.chunk_offsets
                .len()
                .try_into()
                .expect("chunk offsets length must fit in u32"),
        ));

        atom.chunk_offsets.0.into_iter().for_each(|offset| {
            if atom.is_64bit {
                data.extend(offset.to_be_bytes());
            } else {
                data.extend(be_u32(
                    offset.try_into().expect("chunk offset must fit in u32"),
                ))
            }
        });

        data
    }
}

mod parser {
    use winnow::{
        binary::{be_u32, be_u64},
        combinator::{empty, repeat, seq, trace},
        error::{ContextError, ErrMode, StrContext},
        Parser,
    };

    use super::{ChunkOffsetAtom, ChunkOffsets};
    use crate::atom::util::parser::{byte_array, stream, version, Stream};

    pub fn parse_stco_data(input: &[u8]) -> Result<ChunkOffsetAtom, crate::ParseError> {
        parse_stco_co64_data_inner(false)
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    pub fn parse_co64_data(input: &[u8]) -> Result<ChunkOffsetAtom, crate::ParseError> {
        parse_stco_co64_data_inner(true)
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_stco_co64_data_inner<'i>(
        is_64bit: bool,
    ) -> impl Parser<Stream<'i>, ChunkOffsetAtom, ErrMode<ContextError>> {
        trace(
            if is_64bit { "co64" } else { "stco" },
            move |input: &mut Stream<'_>| {
                seq!(ChunkOffsetAtom {
                    version: version,
                    flags: byte_array.context(StrContext::Label("flags")),
                    chunk_offsets: chunk_offsets(is_64bit)
                        .map(ChunkOffsets)
                        .context(StrContext::Label("chunk_offsets")),
                    is_64bit: empty.value(is_64bit),
                })
                .parse_next(input)
            },
        )
    }

    fn chunk_offsets<'i>(
        is_64bit: bool,
    ) -> impl Parser<Stream<'i>, Vec<u64>, ErrMode<ContextError>> {
        trace("chunk_offsets", move |input: &mut Stream<'_>| {
            let entry_count = be_u32.parse_next(input)?;
            repeat(entry_count as usize, chunk_offset(is_64bit)).parse_next(input)
        })
    }

    fn chunk_offset<'i>(is_64bit: bool) -> impl Parser<Stream<'i>, u64, ErrMode<ContextError>> {
        trace("chunk_offset", move |input: &mut Stream<'_>| {
            if is_64bit {
                be_u64.parse_next(input)
            } else {
                be_u32.map(|v| v as u64).parse_next(input)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available stco/co64 test data files
    #[test]
    fn test_stco_co64_roundtrip() {
        test_atom_roundtrip_sync::<ChunkOffsetAtom>(STCO);
        test_atom_roundtrip_sync::<ChunkOffsetAtom>(CO64);
    }
}
