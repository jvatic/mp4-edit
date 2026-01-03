use derive_more::Display;

pub use crate::atom::stsd::extension::{
    BtrtExtension, DecoderSpecificInfo, EsdsExtension, StsdExtension,
};
use crate::{atom::FourCC, parser::ParseAtomData, writer::SerializeAtom, ParseError};

mod audio;
mod extension;
mod text;

pub use audio::*;
pub use text::*;

pub const STSD: FourCC = FourCC::new(b"stsd");

pub const SAMPLE_ENTRY_MP4A: FourCC = FourCC::new(b"mp4a"); // AAC audio
pub const SAMPLE_ENTRY_AAVD: FourCC = FourCC::new(b"aavd"); // Audible Audio
pub const SAMPLE_ENTRY_TEXT: FourCC = FourCC::new(b"text"); // Plain text

#[derive(Debug, Clone, Display, PartialEq)]
#[display("{}", self.as_str())]
pub enum SampleEntryType {
    /// AAC audio
    Mp4a,
    /// Audible Audio (can be treated as Mp4a)
    Aavd,
    /// QuickTime Text Media
    Text,
    /// Unknown/unsupported sample entry type
    Unknown(FourCC),
}

impl From<FourCC> for SampleEntryType {
    fn from(fourcc: FourCC) -> Self {
        match fourcc {
            SAMPLE_ENTRY_MP4A => SampleEntryType::Mp4a,
            SAMPLE_ENTRY_AAVD => SampleEntryType::Aavd,
            SAMPLE_ENTRY_TEXT => SampleEntryType::Text,
            _ => SampleEntryType::Unknown(fourcc),
        }
    }
}

impl From<SampleEntryType> for FourCC {
    fn from(value: SampleEntryType) -> Self {
        match value {
            SampleEntryType::Mp4a => SAMPLE_ENTRY_MP4A,
            SampleEntryType::Aavd => SAMPLE_ENTRY_AAVD,
            SampleEntryType::Text => SAMPLE_ENTRY_TEXT,
            SampleEntryType::Unknown(bytes) => bytes,
        }
    }
}

impl SampleEntryType {
    fn into_bytes(self) -> [u8; 4] {
        FourCC::from(self).into_bytes()
    }

    pub fn as_str(&self) -> &str {
        match self {
            SampleEntryType::Mp4a => SAMPLE_ENTRY_MP4A.as_str(),
            SampleEntryType::Aavd => SAMPLE_ENTRY_AAVD.as_str(),
            SampleEntryType::Text => SAMPLE_ENTRY_TEXT.as_str(),
            SampleEntryType::Unknown(bytes) => bytes.as_str(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SampleEntryData {
    Audio(AudioSampleEntry),
    Text(TextSampleEntry),
    Other(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct SampleEntry {
    /// Sample entry type (4CC code)
    pub entry_type: SampleEntryType,
    /// Data reference index
    pub data_reference_index: u16,
    /// Raw sample entry data (codec-specific)
    pub data: SampleEntryData,
}

#[derive(Default, Debug, Clone)]
pub struct SampleDescriptionTableAtom {
    /// Version of the stsd atom format (0)
    pub version: u8,
    /// Flags for the stsd atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of sample entries
    pub entries: Vec<SampleEntry>,
}

impl From<Vec<SampleEntry>> for SampleDescriptionTableAtom {
    fn from(entries: Vec<SampleEntry>) -> Self {
        SampleDescriptionTableAtom {
            version: 0,
            flags: [0u8; 3],
            entries,
        }
    }
}

impl SampleDescriptionTableAtom {
    pub fn find_or_create_entry<P, D>(&mut self, pred: P, default_fn: D) -> &mut SampleEntry
    where
        P: Fn(&SampleEntry) -> bool,
        D: FnOnce() -> SampleEntry,
    {
        if let Some(index) = self.entries.iter().position(pred) {
            return &mut self.entries[index];
        }
        self.entries.push(default_fn());
        self.entries.last_mut().unwrap()
    }
}

impl ParseAtomData for SampleDescriptionTableAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, STSD);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_stsd_data.parse(stream(input))?)
    }
}

impl SerializeAtom for SampleDescriptionTableAtom {
    fn atom_type(&self) -> FourCC {
        STSD
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_stsd_atom(self)
    }
}

mod serializer {
    use super::{
        audio::serializer::serialize_audio_sample_entry,
        text::serializer::serialize_text_sample_entry, SampleDescriptionTableAtom, SampleEntryData,
    };
    use crate::atom::util::serializer::{be_u32, prepend_size_inclusive, SizeU32};

    pub fn serialize_stsd_atom(stsd: SampleDescriptionTableAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(stsd.version);
        data.extend(stsd.flags);
        data.extend(be_u32(
            stsd.entries
                .len()
                .try_into()
                .expect("stsd entries len must fit in u32"),
        ));

        for entry in stsd.entries {
            data.extend(prepend_size_inclusive::<SizeU32, _>(move || {
                let mut entry_data = Vec::new();
                entry_data.extend(entry.entry_type.into_bytes());
                entry_data.extend([0u8; 6]); // reserved
                entry_data.extend(entry.data_reference_index.to_be_bytes());
                match entry.data {
                    SampleEntryData::Audio(audio) => {
                        entry_data.extend(serialize_audio_sample_entry(audio));
                    }
                    SampleEntryData::Text(text) => {
                        entry_data.extend(serialize_text_sample_entry(text));
                    }
                    SampleEntryData::Other(other_data) => {
                        entry_data.extend_from_slice(&other_data);
                    }
                }
                entry_data
            }));
        }

        data
    }
}

mod parser {
    use winnow::{
        binary::{be_u16, be_u32, length_repeat},
        combinator::seq,
        error::StrContext,
        ModalResult, Parser,
    };

    use super::{
        audio::parser::parse_audio_sample_entry, text::parser::parse_text_sample_entry,
        SampleDescriptionTableAtom, SampleEntry, SampleEntryData, SampleEntryType,
    };
    use crate::atom::util::parser::{
        byte_array, combinators::inclusive_length_and_then, flags3, fourcc, rest_vec, version,
        Stream,
    };

    pub fn parse_stsd_data(input: &mut Stream<'_>) -> ModalResult<SampleDescriptionTableAtom> {
        seq!(SampleDescriptionTableAtom {
            version: version.verify(|v| *v == 0),
            flags: flags3,
            entries: length_repeat(be_u32, parse_sample_entry)
                .context(StrContext::Label("entries")),
        })
        .parse_next(input)
    }

    fn parse_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntry> {
        inclusive_length_and_then(
            be_u32,
            seq!(SampleEntry {
                entry_type: fourcc
                    .map(SampleEntryType::from)
                    .context(StrContext::Label("entry_type")),
                _: byte_array::<6>.context(StrContext::Label("reserved")), // reserved
                data_reference_index: be_u16.context(StrContext::Label("data_reference_index")),
                data: match entry_type {
                    SampleEntryType::Mp4a | SampleEntryType::Aavd => {
                        parse_audio_sample_entry
                    }
                    SampleEntryType::Text => {
                        parse_text_sample_entry
                    }
                    _ => parse_unknown_sample_entry,
                }.context(StrContext::Label("data")),
            }),
        )
        .parse_next(input)
    }

    pub fn parse_unknown_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        rest_vec.map(SampleEntryData::Other).parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::test_utils::test_atom_roundtrip;

    use super::*;

    /// Test round-trip for all available stsd test data files
    #[test]
    fn test_stsd_roundtrip() {
        test_atom_roundtrip::<SampleDescriptionTableAtom>(STSD);
    }
}
