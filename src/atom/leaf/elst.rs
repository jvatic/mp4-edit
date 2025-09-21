use std::time::Duration;

use bon::bon;
use futures_io::AsyncRead;

use crate::{
    atom::{
        util::{read_to_end, scaled_duration},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const ELST: &[u8; 4] = b"elst";

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

pub struct SegmentDuration {
    duration: Duration,
    movie_timescale: u32,
}

#[bon]
impl SegmentDuration {
    #[builder]
    pub fn new(duration: Duration, movie_timescale: u32) -> Self {
        Self {
            duration,
            movie_timescale,
        }
    }

    pub fn scaled(&self) -> u64 {
        scaled_duration(self.duration, self.movie_timescale as u64)
    }
}

#[derive(Default)]
pub struct MediaDuration {
    duration: Duration,
    media_timescale: u32,
}

#[bon]
impl MediaDuration {
    #[builder]
    pub fn new(duration: Duration, media_timescale: u32) -> Self {
        Self {
            duration,
            media_timescale,
        }
    }

    pub fn scaled(&self) -> i64 {
        i64::try_from(scaled_duration(self.duration, self.media_timescale as u64))
            .expect("scaled duration should fit in i64")
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
        segment_duration: SegmentDuration,
        #[builder(default = Default::default())] media_time: MediaDuration,
        #[builder(default = 1.0)] media_rate: f32,
    ) -> Self {
        Self {
            segment_duration: segment_duration.scaled(),
            media_time: media_time.scaled(),
            media_rate,
        }
    }
}

impl ParseAtom for EditListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != ELST {
            return Err(ParseError::new_unexpected_atom(atom_type, ELST));
        }
        let data = read_to_end(reader).await?;
        parser::parse_elst_data(&data)
    }
}

impl SerializeAtom for EditListAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*ELST)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_elst_atom(self)
    }
}

const FIXED_POINT_SCALE: f32 = 65536.0;

mod serializer {
    use crate::atom::elst::{EditEntry, FIXED_POINT_SCALE};

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
            1 => entries.into_iter().flat_map(entry_u64).collect(),
            _ => entries.into_iter().flat_map(entry_u32).collect(),
        }
    }

    fn entry_u32(entry: EditEntry) -> Vec<u8> {
        vec![
            u32::try_from(entry.segment_duration)
                .expect("segument_duration should fit in u32")
                .to_be_bytes()
                .to_vec(),
            i32::try_from(entry.media_time)
                .expect("media_time should fit in u32")
                .to_be_bytes()
                .to_vec(),
            media_rate(entry.media_rate),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn entry_u64(entry: EditEntry) -> Vec<u8> {
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
        // Convert f32 to fixed-point 16.16
        let rate_fixed = (media_rate * FIXED_POINT_SCALE) as u32;
        rate_fixed.to_be_bytes().to_vec()
    }
}

mod parser {
    use winnow::{
        binary::{be_i32, be_i64, be_u32, be_u64, length_repeat},
        combinator::{seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::EditListAtom;
    use crate::atom::{
        elst::{EditEntry, FIXED_POINT_SCALE},
        util::parser::{flags3, stream, version, Stream},
    };

    pub fn parse_elst_data(input: &[u8]) -> Result<EditListAtom, crate::ParseError> {
        parse_elst_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_elst_data_inner(input: &mut Stream<'_>) -> ModalResult<EditListAtom> {
        trace(
            "elst",
            seq!(EditListAtom {
                version: version,
                flags: flags3,
                entries: length_repeat(
                    entry_count,
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

    fn entry_count(input: &mut Stream<'_>) -> ModalResult<u32> {
        trace(
            "entry_count",
            be_u32.context(StrContext::Label("entry_count")),
        )
        .parse_next(input)
    }

    fn entry_32(input: &mut Stream<'_>) -> ModalResult<EditEntry> {
        trace(
            "entry_u32",
            seq!(EditEntry {
                segment_duration: segment_duration_u32,
                media_time: media_time_i32,
                media_rate: media_rate,
            })
            .context(StrContext::Label("entry")),
        )
        .parse_next(input)
    }

    fn entry_64(input: &mut Stream<'_>) -> ModalResult<EditEntry> {
        trace(
            "entry_u64",
            seq!(EditEntry {
                segment_duration: segment_duration_u64,
                media_time: media_time_i64,
                media_rate: media_rate,
            })
            .context(StrContext::Label("entry")),
        )
        .parse_next(input)
    }

    fn segment_duration_u64(input: &mut Stream<'_>) -> ModalResult<u64> {
        trace(
            "segment_duration_u64",
            be_u64.context(StrContext::Label("segment_duration")),
        )
        .parse_next(input)
    }

    fn segment_duration_u32(input: &mut Stream<'_>) -> ModalResult<u64> {
        trace(
            "segment_duration_u32",
            be_u32
                .map(|v| v as u64)
                .context(StrContext::Label("segment_duration")),
        )
        .parse_next(input)
    }

    fn media_time_i64(input: &mut Stream<'_>) -> ModalResult<i64> {
        trace(
            "media_time_i64",
            be_i64.context(StrContext::Label("media_time")),
        )
        .parse_next(input)
    }

    fn media_time_i32(input: &mut Stream<'_>) -> ModalResult<i64> {
        trace(
            "media_time_i32",
            be_i32
                .map(|v| v as i64)
                .context(StrContext::Label("media_time")),
        )
        .parse_next(input)
    }

    fn media_rate(input: &mut Stream<'_>) -> ModalResult<f32> {
        trace(
            "media_rate",
            be_u32
                .map(|v| (v as f32) / FIXED_POINT_SCALE)
                .context(StrContext::Label("media_time")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available elst test data files
    #[test]
    fn test_elst_roundtrip() {
        test_atom_roundtrip_sync::<EditListAtom>(ELST);
    }
}
