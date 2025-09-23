use bon::Builder;
use futures_io::AsyncRead;

use crate::{
    atom::{util::read_to_end, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const DREF: &[u8; 4] = b"dref";

/// Data Reference Entry Types
pub mod entry_types {
    /// URL data reference
    pub const URL: &[u8; 4] = b"url ";
    /// URN data reference
    pub const URN: &[u8; 4] = b"urn ";
    /// Alias data reference (Mac OS)
    pub const ALIS: &[u8; 4] = b"alis";
}

/// Data Reference Entry Flags
pub mod flags {
    /// Media data is in the same file as the Movie Atom
    pub const SELF_CONTAINED: u32 = 0x000001;
}

#[derive(Debug, Clone)]
pub enum DataReferenceEntryInner {
    Url(String),
    Urn(String),
    Alias(Vec<u8>),
    Unknown(FourCC, Vec<u8>),
}

impl DataReferenceEntryInner {
    fn new(entry_type: FourCC, data: Vec<u8>) -> Self {
        match &entry_type.0 {
            entry_types::URL => DataReferenceEntryInner::Url(String::from_utf8(data).unwrap()),
            entry_types::URN => DataReferenceEntryInner::Urn(String::from_utf8(data).unwrap()),
            entry_types::ALIS => DataReferenceEntryInner::Alias(data),
            _ => DataReferenceEntryInner::Unknown(entry_type, data),
        }
    }
}

/// A single data reference entry
#[derive(Debug, Clone, Builder)]
pub struct DataReferenceEntry {
    /// Entry data/type (URL string, URN, alias data, etc.)
    #[builder(setters(vis = ""))]
    pub inner: DataReferenceEntryInner,
    /// Version of the entry format
    #[builder(default)]
    pub version: u8,
    /// Entry flags
    #[builder(default)]
    pub flags: [u8; 3],
}

impl<S: data_reference_entry_builder::State> DataReferenceEntryBuilder<S> {
    pub fn url(
        self,
        url: impl Into<String>,
    ) -> DataReferenceEntryBuilder<data_reference_entry_builder::SetInner<S>>
    where
        S::Inner: data_reference_entry_builder::IsUnset,
    {
        self.inner(DataReferenceEntryInner::Url(url.into()))
    }

    pub fn urn(
        self,
        urn: impl Into<String>,
    ) -> DataReferenceEntryBuilder<data_reference_entry_builder::SetInner<S>>
    where
        S::Inner: data_reference_entry_builder::IsUnset,
    {
        self.inner(DataReferenceEntryInner::Urn(urn.into()))
    }
}

impl DataReferenceEntry {
    /// Check if this entry has the self-contained flag set
    pub fn is_self_contained(&self) -> bool {
        let flags_u32 = u32::from_be_bytes([0, self.flags[0], self.flags[1], self.flags[2]]);
        (flags_u32 & flags::SELF_CONTAINED) != 0
    }
}

/// Data Reference Atom (dref) - ISO/IEC 14496-12
/// Contains a table of data references that declare the location(s) of the media data
#[derive(Debug, Clone, Builder)]
pub struct DataReferenceAtom {
    /// Version of the dref atom format
    #[builder(default = 0)]
    pub version: u8,
    /// Atom flags
    #[builder(default = [0u8; 3])]
    pub flags: [u8; 3],
    /// Data reference entries
    #[builder(with = FromIterator::from_iter)]
    pub entries: Vec<DataReferenceEntry>,
}

impl<S: data_reference_atom_builder::State> DataReferenceAtomBuilder<S> {
    pub fn entry(
        self,
        entry: DataReferenceEntry,
    ) -> DataReferenceAtomBuilder<data_reference_atom_builder::SetEntries<S>>
    where
        S::Entries: data_reference_atom_builder::IsUnset,
    {
        self.entries(vec![entry])
    }
}

impl ParseAtom for DataReferenceAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != DREF {
            return Err(ParseError::new_unexpected_atom(atom_type, DREF));
        }
        let data = read_to_end(reader).await?;
        parser::parse_dref_data(&data)
    }
}

impl SerializeAtom for DataReferenceAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*DREF)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_dref_data(self)
    }
}

mod serializer {
    use crate::atom::{
        dref::{entry_types, DataReferenceEntry, DataReferenceEntryInner},
        DataReferenceAtom,
    };

    pub fn serialize_dref_data(data: DataReferenceAtom) -> Vec<u8> {
        let entries = data.entries;
        vec![
            version(data.version),
            flags(data.flags),
            entry_count(entries.len()),
            entries.into_iter().flat_map(entry).collect(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn version(version: u8) -> Vec<u8> {
        vec![version]
    }

    fn flags(flags: [u8; 3]) -> Vec<u8> {
        flags.to_vec()
    }

    fn entry_count(n: usize) -> Vec<u8> {
        u32::try_from(n)
            .expect("entries len must fit in a u32")
            .to_be_bytes()
            .to_vec()
    }

    fn entry(e: DataReferenceEntry) -> Vec<u8> {
        let e = raw_entry(e);
        let data: Vec<u8> = vec![version(e.version), flags(e.flags), e.data]
            .into_iter()
            .flatten()
            .collect();
        let header_size = 4 + 4; // size + type
        vec![
            entry_size(header_size + data.len()).to_vec(),
            e.typ.to_vec(),
            data,
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn entry_size(n: usize) -> [u8; 4] {
        u32::try_from(n)
            .expect("entry size len must fit in a u32")
            .to_be_bytes()
    }

    struct RawEntry {
        version: u8,
        flags: [u8; 3],
        typ: [u8; 4],
        data: Vec<u8>,
    }

    fn raw_entry(e: DataReferenceEntry) -> RawEntry {
        let (typ, data) = match e.inner {
            DataReferenceEntryInner::Url(url) => (*entry_types::URL, url.into_bytes()),
            DataReferenceEntryInner::Urn(urn) => (*entry_types::URN, urn.into_bytes()),
            DataReferenceEntryInner::Alias(alias_data) => (*entry_types::ALIS, alias_data),
            DataReferenceEntryInner::Unknown(typ, unknown_data) => (typ.0, unknown_data),
        };
        RawEntry {
            version: e.version,
            flags: e.flags,
            typ,
            data,
        }
    }
}

mod parser {
    use winnow::{
        binary::length_repeat,
        combinator::{seq, trace},
        error::StrContext,
        Parser,
    };

    use super::{DataReferenceAtom, DataReferenceEntry, DataReferenceEntryInner};
    use crate::{
        atom::util::parser::{
            combinators::with_len, flags3, fourcc, stream, take_vec, usize_be_u32, version, Stream,
        },
        FourCC,
    };

    pub fn parse_dref_data(input: &[u8]) -> Result<DataReferenceAtom, crate::ParseError> {
        parse_dref_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_dref_data_inner(input: &mut Stream<'_>) -> winnow::ModalResult<DataReferenceAtom> {
        trace(
            "dref",
            (version, flags3, entries)
                .map(|(version, flags, entries)| DataReferenceAtom {
                    version,
                    flags,
                    entries,
                })
                .context(StrContext::Label("dref")),
        )
        .parse_next(input)
    }

    fn entries(input: &mut Stream<'_>) -> winnow::ModalResult<Vec<DataReferenceEntry>> {
        trace(
            "entries",
            length_repeat(
                usize_be_u32.context(StrContext::Label("entry_count")),
                trace("entry", entry.context(StrContext::Label("entry"))),
            ),
        )
        .parse_next(input)
    }

    fn entry(input: &mut Stream<'_>) -> winnow::ModalResult<DataReferenceEntry> {
        struct EntryHeader {
            size: usize,
            typ: FourCC,
            version: u8,
            flags: [u8; 3],
        }

        fn entry_header(input: &mut Stream<'_>) -> winnow::ModalResult<EntryHeader> {
            trace(
                "entry_header",
                seq!(EntryHeader {
                    size: usize_be_u32,
                    typ: fourcc,
                    version: version,
                    flags: flags3
                })
                .context(StrContext::Label("entry_header")),
            )
            .parse_next(input)
        }

        let (
            EntryHeader {
                size,
                typ,
                version,
                flags,
            },
            header_size,
        ) = with_len(entry_header).parse_next(input)?;

        let data: Vec<u8> = trace("entry_data", take_vec(size - header_size))
            .context(StrContext::Label("entry_data"))
            .parse_next(input)?;

        Ok(DataReferenceEntry {
            inner: DataReferenceEntryInner::new(typ, data),
            version,
            flags,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available dref test data files
    #[test]
    fn test_dref_roundtrip() {
        test_atom_roundtrip_sync::<DataReferenceAtom>(DREF);
    }
}
