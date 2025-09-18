use bon::Builder;
use futures_io::AsyncRead;
use winnow::{
    binary::{be_u32, u8},
    combinator::{repeat, trace},
    error::{StrContext, StrContextValue},
    Bytes, Parser,
};

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
    Unknown(Vec<u8>),
}

impl DataReferenceEntryInner {
    fn new(entry_type: FourCC, data: Vec<u8>) -> Self {
        match &entry_type.0 {
            entry_types::URL => DataReferenceEntryInner::Url(String::from_utf8(data).unwrap()),
            entry_types::URN => DataReferenceEntryInner::Urn(String::from_utf8(data).unwrap()),
            entry_types::ALIS => DataReferenceEntryInner::Alias(data),
            _ => DataReferenceEntryInner::Unknown(data),
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
        parse_dref_data
            .parse(stream(&data))
            .map_err(ParseError::from_winnow)
    }
}

impl SerializeAtom for DataReferenceAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*DREF)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Entry count (4 bytes, big-endian)
        data.extend_from_slice(
            &u32::try_from(self.entries.len())
                .expect("entries len must fit in a u32")
                .to_be_bytes(),
        );

        // Entries
        for entry in self.entries {
            let mut entry_data = Vec::new();

            // Entry version and flags (4 bytes)
            entry_data.push(entry.version);
            entry_data.extend_from_slice(&entry.flags);

            // Entry data based on type
            let (entry_type, entry_payload) = match entry.inner {
                DataReferenceEntryInner::Url(url) => (entry_types::URL, url.into_bytes()),
                DataReferenceEntryInner::Urn(urn) => (entry_types::URN, urn.into_bytes()),
                DataReferenceEntryInner::Alias(alias_data) => (entry_types::ALIS, alias_data),
                DataReferenceEntryInner::Unknown(unknown_data) => {
                    // For unknown types, we can't determine the original type,
                    // so we'll use a generic approach
                    (b"unkn", unknown_data)
                }
            };

            entry_data.extend_from_slice(&entry_payload);

            // Calculate total entry size (4 + 4 + entry_data.len())
            let entry_size = 8 + entry_data.len();

            // Write entry size (4 bytes, big-endian)
            data.extend_from_slice(
                &(u32::try_from(entry_size).expect("entry_size should fit in u32")).to_be_bytes(),
            );

            // Write entry type (4 bytes)
            data.extend_from_slice(entry_type);

            // Write entry data (version + flags + payload)
            data.extend_from_slice(&entry_data);
        }

        data
    }
}

type Stream<'i> = &'i Bytes;

fn stream(b: &[u8]) -> Stream<'_> {
    Bytes::new(b)
}

fn parse_dref_data(input: &mut Stream<'_>) -> winnow::ModalResult<DataReferenceAtom> {
    (parse_version, parse_flags, parse_entries)
        .map(|(version, flags, entries)| DataReferenceAtom {
            version,
            flags,
            entries,
        })
        .context(StrContext::Label("dref"))
        .parse_next(input)
}

fn parse_version(input: &mut Stream<'_>) -> winnow::ModalResult<u8> {
    trace("version", u8)
        .context(StrContext::Label("version"))
        .parse_next(input)
}

fn parse_flags(input: &mut Stream<'_>) -> winnow::ModalResult<[u8; 3]> {
    trace(
        "flags",
        (
            u8.context(StrContext::Label("[0]")),
            u8.context(StrContext::Label("[1]")),
            u8.context(StrContext::Label("[2]")),
        ),
    )
    .map(|(a, b, c)| [a, b, c])
    .context(StrContext::Label("flags"))
    .parse_next(input)
}

fn parse_entries(input: &mut Stream<'_>) -> winnow::ModalResult<Vec<DataReferenceEntry>> {
    let entry_count = parse_entry_count(input)?;
    trace(
        "entries",
        repeat(
            entry_count,
            trace("entry", parse_entry.context(StrContext::Label("entry"))),
        ),
    )
    .parse_next(input)
}

fn parse_entry_count(input: &mut Stream<'_>) -> winnow::ModalResult<usize> {
    trace("entry_count", be_u32)
        .map(|s| s as usize)
        .parse_next(input)
}

fn parse_entry_size(input: &mut Stream<'_>) -> winnow::ModalResult<usize> {
    trace(
        "entry_size",
        be_u32
            .map(|s| s as usize)
            .context(StrContext::Label("entry_size"))
            .context(StrContext::Expected(StrContextValue::Description("be u32"))),
    )
    .parse_next(input)
}

fn parse_fourcc(input: &mut Stream<'_>) -> winnow::ModalResult<FourCC> {
    trace(
        "fourcc",
        (
            u8.context(StrContext::Label("[0]")),
            u8.context(StrContext::Label("[1]")),
            u8.context(StrContext::Label("[2]")),
            u8.context(StrContext::Label("[3]")),
        )
            .map(|(a, b, c, d)| FourCC([a, b, c, d]))
            .context(StrContext::Label("fourcc")),
    )
    .parse_next(input)
}

fn parse_entry(input: &mut Stream<'_>) -> winnow::ModalResult<DataReferenceEntry> {
    let start_len = input.len();
    let size = parse_entry_size(input)?;
    let typ = parse_fourcc(input)?;
    let version = parse_version(input)?;
    let flags = parse_flags(input)?;
    let header_size = start_len - input.len();
    let data: Vec<u8> = trace("entry_data", repeat(size - header_size, u8))
        .context(StrContext::Label("entry_data"))
        .parse_next(input)?;

    Ok(DataReferenceEntry {
        inner: DataReferenceEntryInner::new(typ, data),
        version,
        flags,
    })
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
