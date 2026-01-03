use bon::Builder;

use std::time::Duration;

use crate::{
    atom::{
        util::{mp4_timestamp_now, scaled_duration, unscaled_duration},
        FourCC,
    },
    parser::ParseAtomData,
    writer::SerializeAtom,
    ParseError,
};

pub const MVHD: FourCC = FourCC::new(b"mvhd");

#[derive(Debug, Clone, Builder)]
pub struct MovieHeaderAtom {
    /// Version of the mvhd atom format (0 or 1)
    #[builder(default = 0)]
    pub version: u8,
    /// Flags for the mvhd atom (usually all zeros)
    #[builder(default = [0u8; 3])]
    pub flags: [u8; 3],
    /// When the movie was created (seconds since Jan 1, 1904 UTC)
    #[builder(default = mp4_timestamp_now())]
    pub creation_time: u64,
    /// When the movie was last modified (seconds since Jan 1, 1904 UTC)
    #[builder(default = mp4_timestamp_now())]
    pub modification_time: u64,
    /// Number of time units per second (e.g., 90000 for 90kHz)
    pub timescale: u32,
    /// Duration of the movie in timescale units
    pub duration: u64,
    /// Playback rate (1.0 = normal speed, 2.0 = double speed)
    #[builder(default = 1.0)]
    pub rate: f32,
    /// Audio volume level (1.0 = full volume, 0.0 = muted)
    #[builder(default = 1.0)]
    pub volume: f32,
    #[builder(default)]
    pub reserved: [u8; 10],
    /// 3x3 transformation matrix for video display positioning/rotation
    pub matrix: Option<[i32; 9]>,
    /// Time when preview starts (in timescale units)
    #[builder(default = 0)]
    pub preview_time: u32,
    /// Duration of the preview (in timescale units)
    #[builder(default = 0)]
    pub preview_duration: u32,
    /// Time of poster frame to display when movie is not playing
    #[builder(default = 0)]
    pub poster_time: u32,
    /// Start time of current selection (in timescale units)
    #[builder(default = 0)]
    pub selection_time: u32,
    /// Duration of current selection (in timescale units)
    #[builder(default = 0)]
    pub selection_duration: u32,
    /// Current playback time position (in timescale units)
    #[builder(default = 0)]
    pub current_time: u32,
    /// ID to use for the next track added to this movie
    pub next_track_id: u32,
}

impl MovieHeaderAtom {
    pub fn update_duration<F>(&mut self, mut closure: F) -> &mut Self
    where
        F: FnMut(Duration) -> Duration,
    {
        self.duration = scaled_duration(
            closure(unscaled_duration(self.duration, u64::from(self.timescale))),
            u64::from(self.timescale),
        );
        self
    }

    pub fn duration(&self) -> Duration {
        unscaled_duration(self.duration, u64::from(self.timescale))
    }
}

impl ParseAtomData for MovieHeaderAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, MVHD);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_mvhd_data.parse(stream(input))?)
    }
}

impl SerializeAtom for MovieHeaderAtom {
    fn atom_type(&self) -> FourCC {
        MVHD
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_mvhd_atom(self)
    }
}

mod serializer {
    use crate::atom::util::serializer::{fixed_point_16x16, fixed_point_8x8};

    use super::MovieHeaderAtom;

    pub fn serialize_mvhd_atom(mvhd: MovieHeaderAtom) -> Vec<u8> {
        let mut data = Vec::new();

        // Determine version based on whether values fit in 32-bit
        let needs_64_bit = mvhd.creation_time > u64::from(u32::MAX)
            || mvhd.modification_time > u64::from(u32::MAX)
            || mvhd.duration > u64::from(u32::MAX);

        let version: u8 = if needs_64_bit { 1 } else { 0 };

        let be_u32_or_u64 = |v: u64| match version {
            0 => u32::try_from(v).unwrap().to_be_bytes().to_vec(),
            1 => v.to_be_bytes().to_vec(),
            _ => unreachable!(),
        };

        data.extend(version.to_be_bytes());
        data.extend(mvhd.flags);
        data.extend(be_u32_or_u64(mvhd.creation_time));
        data.extend(be_u32_or_u64(mvhd.modification_time));
        data.extend(mvhd.timescale.to_be_bytes());
        data.extend(be_u32_or_u64(mvhd.duration));
        data.extend(fixed_point_16x16(mvhd.rate));
        data.extend(fixed_point_8x8(mvhd.volume));
        data.extend(mvhd.reserved);
        if let Some(matrix) = mvhd.matrix {
            for value in matrix {
                data.extend(value.to_be_bytes());
            }
        }
        data.extend_from_slice(&mvhd.preview_time.to_be_bytes());
        data.extend_from_slice(&mvhd.preview_duration.to_be_bytes());
        data.extend_from_slice(&mvhd.poster_time.to_be_bytes());
        data.extend_from_slice(&mvhd.selection_time.to_be_bytes());
        data.extend_from_slice(&mvhd.selection_duration.to_be_bytes());
        data.extend_from_slice(&mvhd.current_time.to_be_bytes());
        data.extend_from_slice(&mvhd.next_track_id.to_be_bytes());

        data
    }
}

mod parser {
    use winnow::{
        binary::{be_i32, be_u32, be_u64, u8},
        combinator::{opt, seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::MovieHeaderAtom;
    use crate::atom::util::parser::{
        be_u32_as_u64, fixed_array, fixed_point_16x16, fixed_point_8x8, flags3, version_0_or_1,
        Stream,
    };

    pub fn parse_mvhd_data(input: &mut Stream<'_>) -> ModalResult<MovieHeaderAtom> {
        let be_u32_or_u64 = |version: u8| {
            let be_u64_type_fix =
                |input: &mut Stream<'_>| -> ModalResult<u64> { be_u64.parse_next(input) };
            match version {
                0 => be_u32_as_u64,
                1 => be_u64_type_fix,
                _ => unreachable!(),
            }
        };

        trace(
            "mvhd",
            seq!(MovieHeaderAtom {
                version: version_0_or_1,
                flags: flags3,
                creation_time: be_u32_or_u64(version).context(StrContext::Label("creation_time")),
                modification_time: be_u32_or_u64(version)
                    .context(StrContext::Label("modification_time")),
                timescale: be_u32.context(StrContext::Label("timescale")),
                duration: be_u32_or_u64(version).context(StrContext::Label("duration")),
                rate: fixed_point_16x16.context(StrContext::Label("rate")),
                volume: fixed_point_8x8.context(StrContext::Label("volume")),
                reserved: fixed_array(u8).context(StrContext::Label("reserved")),
                matrix: opt(fixed_array(be_i32)).context(StrContext::Label("matrix")),
                preview_time: be_u32.context(StrContext::Label("preview_time")),
                preview_duration: be_u32.context(StrContext::Label("preview_duration")),
                poster_time: be_u32.context(StrContext::Label("poster_time")),
                selection_time: be_u32.context(StrContext::Label("selection_time")),
                selection_duration: be_u32.context(StrContext::Label("selection_duration")),
                current_time: be_u32.context(StrContext::Label("current_time")),
                next_track_id: be_u32.context(StrContext::Label("next_track_id")),
            }),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available mvhd test data files
    #[test]
    fn test_mvhd_roundtrip() {
        test_atom_roundtrip::<MovieHeaderAtom>(MVHD);
    }
}
