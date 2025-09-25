use futures_io::AsyncRead;

pub use crate::atom::stsd::{
    extension::{
        btrt::BtrtExtension,
        esds::{DecoderSpecificInfo, EsdsExtension},
        StsdExtension,
    },
    mp4a::Mp4aEntryData,
    tx3g::Tx3gEntryData,
};
use crate::{
    atom::{util::read_to_end, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub mod extension;
pub mod mp4a;
pub mod tx3g;

pub const STSD: &[u8; 4] = b"stsd";

#[derive(Debug, Clone)]
pub enum SampleEntryData {
    Mp4a(Mp4aEntryData),
    Tx3g(Tx3gEntryData),
    Unknown(FourCC, Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct SampleEntry {
    /// Data reference index
    pub data_reference_index: u16,
    /// Raw sample entry data (codec-specific)
    pub data: SampleEntryData,
}

impl SampleEntry {
    fn entry_type(&self) -> &FourCC {
        match &self.data {
            SampleEntryData::Mp4a(_) => &Mp4aEntryData::TYPE,
            SampleEntryData::Tx3g(_) => &Tx3gEntryData::TYPE,
            SampleEntryData::Unknown(typ, _) => typ,
        }
    }

    pub fn is_audio(&self) -> bool {
        match &self.data {
            SampleEntryData::Mp4a(_) => true,
            SampleEntryData::Unknown(typ, _)
                if match &typ.0 {
                    b"aavd" => true,
                    _ => false,
                } =>
            {
                true
            }
            _ => false,
        }
    }
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

    /// Finds an audio sample entry matching `pred` or inserts a new one with `default_fn`
    ///
    /// NOTE: audio sample entries that don't match `pred` will be removed.
    pub fn find_or_create_audio_entry<P, D>(&mut self, pred: P, default_fn: D) -> &mut SampleEntry
    where
        P: Fn(&SampleEntry) -> bool,
        D: FnOnce() -> SampleEntry,
    {
        let mut index = None;
        let mut remove_indices: Vec<usize> = Vec::new();
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if !entry.is_audio() {
                continue;
            }
            if pred(entry) {
                index = Some(i);
            } else {
                remove_indices.push(i);
            }
        }
        for i in remove_indices.into_iter() {
            if let Some(index) = index.as_mut() {
                if *index < i {
                    *index -= 1;
                }
            }
            self.entries.remove(i);
        }
        if let Some(index) = index {
            return &mut self.entries[index];
        }
        self.entries.push(default_fn());
        self.entries.last_mut().unwrap()
    }
}

impl ParseAtom for SampleDescriptionTableAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STSD {
            return Err(ParseError::new_unexpected_atom(atom_type, STSD));
        }
        let data = read_to_end(reader).await?;
        parser::parse_stsd_data(&data)
    }
}

impl SerializeAtom for SampleDescriptionTableAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STSD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_stsd_data(self)
    }
}

mod serializer {
    use crate::atom::{
        stsd::{
            mp4a::serializer::serialize_mp4a_entry_data,
            tx3g::serializer::serialize_tx3g_entry_data,
        },
        util::serializer::{be_u32, prepend_size, SizeU32OrU64},
    };

    use super::{SampleDescriptionTableAtom, SampleEntry, SampleEntryData};

    pub fn serialize_stsd_data(stsd: SampleDescriptionTableAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(stsd.version);
        data.extend(stsd.flags);

        data.extend(be_u32(
            stsd.entries
                .len()
                .try_into()
                .expect("stsd entries len should fit in u32"),
        ));

        for entry in stsd.entries {
            data.extend(serialize_stsd_entry(entry))
        }

        data
    }

    fn serialize_stsd_entry(entry: SampleEntry) -> Vec<u8> {
        prepend_size::<SizeU32OrU64, _>(move || {
            let mut data = Vec::new();

            data.extend(entry.entry_type().as_bytes());

            // Reserved (6 bytes)
            data.extend(&[0u8; 6]);

            data.extend(entry.data_reference_index.to_be_bytes());

            data.extend(serialize_entry_data(entry.data));

            data
        })
    }

    fn serialize_entry_data(entry_data: SampleEntryData) -> Vec<u8> {
        match entry_data {
            SampleEntryData::Mp4a(mp4a) => serialize_mp4a_entry_data(mp4a),
            SampleEntryData::Tx3g(tx3g) => serialize_tx3g_entry_data(tx3g),
            SampleEntryData::Unknown(_, raw) => {
                let mut data = Vec::new();
                data.extend(raw);
                data
            }
        }
    }
}

mod parser {
    use winnow::{
        binary::{be_u16, be_u32, length_repeat},
        combinator::{seq, trace},
        error::{ContextError, ErrMode, StrContext},
        token::rest,
        ModalResult, Parser,
    };

    use super::{Mp4aEntryData, SampleDescriptionTableAtom, SampleEntry, SampleEntryData};
    use crate::{
        atom::{
            stsd::{
                mp4a::parser::mp4a_sample_entry,
                tx3g::{parser::tx3g_sample_entry, Tx3gEntryData},
            },
            util::parser::{
                atom_size, byte_array, combinators::inclusive_length_and_then, flags3, fourcc,
                stream, version, Stream,
            },
        },
        FourCC,
    };

    pub fn parse_stsd_data(input: &[u8]) -> Result<SampleDescriptionTableAtom, crate::ParseError> {
        parse_stsd_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_stsd_data_inner(input: &mut Stream<'_>) -> ModalResult<SampleDescriptionTableAtom> {
        trace(
            "stsd",
            seq!(SampleDescriptionTableAtom {
                version: version,
                flags: flags3,
                entries: length_repeat(be_u32, inclusive_length_and_then(atom_size, entry))
                    .context(StrContext::Label("entries")),
            })
            .context(StrContext::Label("stsd")),
        )
        .parse_next(input)
    }

    fn entry(input: &mut Stream<'_>) -> ModalResult<SampleEntry> {
        trace("entry", move |input: &mut Stream| {
            let typ = fourcc.parse_next(input)?;
            let _ = reserved.parse_next(input)?;
            let data_reference_index = be_u16
                .context(StrContext::Label("data_reference_index"))
                .parse_next(input)?;
            let data = match typ {
                Mp4aEntryData::TYPE => mp4a_sample_entry.parse_next(input)?,
                Tx3gEntryData::TYPE => tx3g_sample_entry.parse_next(input)?,
                _ => unknown_sample_entry(typ).parse_next(input)?,
            };
            Ok(SampleEntry {
                data_reference_index,
                data,
            })
        })
        .parse_next(input)
    }

    fn reserved(input: &mut Stream<'_>) -> ModalResult<[u8; 6]> {
        trace("reserved", byte_array).parse_next(input)
    }

    fn unknown_sample_entry<'i>(
        typ: FourCC,
    ) -> impl Parser<Stream<'i>, SampleEntryData, ErrMode<ContextError>> {
        trace("other_sample_entry", move |input: &mut Stream<'_>| {
            rest.map(|buf: &[u8]| buf.to_vec())
                .map(|data| SampleEntryData::Unknown(typ, data))
                .parse_next(input)
        })
        .context(StrContext::Label("unknown"))
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    use super::*;

    /// Test round-trip for all available stsd test data files
    #[test]
    fn test_stsd_roundtrip() {
        test_atom_roundtrip_sync::<SampleDescriptionTableAtom>(STSD);
    }
}
