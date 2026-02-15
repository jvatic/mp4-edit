use std::time::Duration;

use bon::bon;

use crate::{
    atom::{util::scaled_duration, FourCC},
    parser::ParseAtomData,
    writer::SerializeAtom,
    ParseError,
};

pub const ELST: FourCC = FourCC::new(b"elst");

#[derive(Default, Debug, Clone)]
pub struct EditListAtom {
    /// Version of the elst atom format (0 or 1)
    pub version: u8,
    /// Flags for the elst atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of edit entries
    pub entries: Vec<EditEntry>,
}

impl EditListAtom {
    pub fn new(entries: impl Into<Vec<EditEntry>>) -> Self {
        Self {
            entries: entries.into(),
            ..Default::default()
        }
    }

    pub fn replace_entries(&mut self, entries: impl Into<Vec<EditEntry>>) -> &mut Self {
        self.entries = entries.into();
        self
    }
}

pub struct MediaTime {
    /// None implies -1, or a gap
    start_offset: Option<Duration>,
}

impl Default for MediaTime {
    fn default() -> Self {
        Self {
            start_offset: Some(Duration::from_secs(0)),
        }
    }
}

impl MediaTime {
    /// media time starting at a specified offset
    pub fn new(start_offset: Duration) -> Self {
        Self {
            start_offset: Some(start_offset),
        }
    }

    /// media time representing a gap in playback
    pub fn new_empty() -> Self {
        Self { start_offset: None }
    }

    pub fn scaled(&self, movie_timescale: u64) -> i64 {
        match self.start_offset {
            Some(start_offset) if start_offset.is_zero() => 0,
            Some(start_offset) => i64::try_from(scaled_duration(start_offset, movie_timescale))
                .expect("scaled duration should fit in i64"),
            None => -1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EditEntry {
    /// Duration of this edit segment (in movie timescale units)
    pub segment_duration: u64,
    /// Starting time within the media (in media timescale units)
    /// -1 indicates an empty edit (no media displayed)
    pub media_time: i64,
    /// Playback rate for this segment (1.0 = normal speed)
    pub media_rate: f32,
}

#[bon]
impl EditEntry {
    #[builder]
    pub fn new(
        movie_timescale: u64,
        segment_duration: Duration,
        #[builder(default = Default::default())] media_time: MediaTime,
        #[builder(default = 1.0)] media_rate: f32,
    ) -> Self {
        Self {
            segment_duration: scaled_duration(segment_duration, movie_timescale),
            media_time: media_time.scaled(movie_timescale),
            media_rate,
        }
    }
}

impl ParseAtomData for EditListAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, ELST);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_elst_data.parse(stream(input))?)
    }
}

impl SerializeAtom for EditListAtom {
    fn atom_type(&self) -> FourCC {
        ELST
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_elst_atom(self)
    }
}

mod serializer {
    use crate::atom::{elst::EditEntry, util::serializer::fixed_point_16x16};

    use super::EditListAtom;

    pub fn serialize_elst_atom(atom: EditListAtom) -> Vec<u8> {
        vec![
            version(atom.version),
            flags(atom.flags),
            entry_count(atom.entries.len()),
            entries(atom.version, atom.entries),
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

    fn entry_count(count: usize) -> Vec<u8> {
        u32::try_from(count)
            .expect("entries len should fit in u32")
            .to_be_bytes()
            .to_vec()
    }

    fn entries(version: u8, entries: Vec<EditEntry>) -> Vec<u8> {
        match version {
            1 => entries.into_iter().flat_map(entry_64).collect(),
            _ => entries.into_iter().flat_map(entry_32).collect(),
        }
    }

    fn entry_32(entry: EditEntry) -> Vec<u8> {
        vec![
            u32::try_from(entry.segment_duration)
                .expect("segument_duration should fit in u32")
                .to_be_bytes()
                .to_vec(),
            i32::try_from(entry.media_time)
                .expect("media_time should fit in i32")
                .to_be_bytes()
                .to_vec(),
            media_rate(entry.media_rate),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn entry_64(entry: EditEntry) -> Vec<u8> {
        vec![
            entry.segment_duration.to_be_bytes().to_vec(),
            entry.media_time.to_be_bytes().to_vec(),
            media_rate(entry.media_rate),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn media_rate(media_rate: f32) -> Vec<u8> {
        fixed_point_16x16(media_rate).to_vec()
    }
}

mod parser {
    use winnow::{
        binary::{be_i64, be_u32, be_u64, length_repeat},
        combinator::{seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::EditListAtom;
    use crate::atom::{
        elst::EditEntry,
        util::parser::{be_i32_as, be_u32_as, fixed_point_16x16, flags3, version, Stream},
    };

    pub fn parse_elst_data(input: &mut Stream<'_>) -> ModalResult<EditListAtom> {
        trace(
            "elst",
            seq!(EditListAtom {
                version: version,
                flags: flags3,
                entries: length_repeat(
                    be_u32.context(StrContext::Label("entry_count")),
                    match version {
                        1 => entry_64,
                        _ => entry_32,
                    }
                ),
            })
            .context(StrContext::Label("elst")),
        )
        .parse_next(input)
    }

    fn entry_32(input: &mut Stream<'_>) -> ModalResult<EditEntry> {
        trace(
            "entry_32",
            seq!(EditEntry {
                segment_duration: be_u32_as.context(StrContext::Label("segment_duration")),
                media_time: be_i32_as.context(StrContext::Label("media_time")),
                media_rate: media_rate,
            })
            .context(StrContext::Label("entry")),
        )
        .parse_next(input)
    }

    fn entry_64(input: &mut Stream<'_>) -> ModalResult<EditEntry> {
        trace(
            "entry_64",
            seq!(EditEntry {
                segment_duration: be_u64.context(StrContext::Label("segment_duration")),
                media_time: be_i64.context(StrContext::Label("media_time")),
                media_rate: media_rate,
            })
            .context(StrContext::Label("entry")),
        )
        .parse_next(input)
    }

    fn media_rate(input: &mut Stream<'_>) -> ModalResult<f32> {
        trace(
            "media_rate",
            fixed_point_16x16.context(StrContext::Label("media_rate")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available elst test data files
    #[test]
    fn test_elst_roundtrip() {
        test_atom_roundtrip::<EditListAtom>(ELST);
    }
}
