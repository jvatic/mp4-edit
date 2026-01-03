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

pub const TKHD: FourCC = FourCC::new(b"tkhd");

const IDENTITY_MATRIX: [i32; 9] = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];

#[derive(Default, Debug, Clone, Builder)]
pub struct TrackHeaderAtom {
    /// Version of the tkhd atom format (0 or 1)
    #[builder(default = 0)]
    pub version: u8,
    /// Flags for the tkhd atom (bit flags for track properties)
    #[builder(default = [0, 0, 7])]
    pub flags: [u8; 3],
    /// When the track was created (seconds since Jan 1, 1904 UTC)
    #[builder(default = mp4_timestamp_now())]
    pub creation_time: u64,
    /// When the track was last modified (seconds since Jan 1, 1904 UTC)
    #[builder(default = mp4_timestamp_now())]
    pub modification_time: u64,
    /// Unique identifier for this track within the movie
    pub track_id: u32,
    /// Duration of the track in movie timescale units
    pub duration: u64,
    /// Playback layer (lower numbers are closer to viewer)
    #[builder(default = 0)]
    pub layer: i16,
    /// Audio balance or stereo balance (-1.0 = left, 0.0 = center, 1.0 = right)
    #[builder(default = 0)]
    pub alternate_group: i16,
    /// Audio volume level (1.0 = full volume, 0.0 = muted)
    #[builder(default = 1.0)]
    pub volume: f32,
    /// 3x3 transformation matrix for video display positioning/rotation
    ///
    /// `None` if matrix is empty or is the identity matrix
    pub matrix: Option<[i32; 9]>,
    /// Track width in pixels (fixed-point 16.16)
    #[builder(default = 0.0)]
    pub width: f32,
    /// Track height in pixels (fixed-point 16.16)
    #[builder(default = 0.0)]
    pub height: f32,
}

impl TrackHeaderAtom {
    pub fn duration(&self, movie_timescale: u64) -> Duration {
        unscaled_duration(self.duration, movie_timescale)
    }

    pub fn update_duration<F>(&mut self, movie_timescale: u64, mut closure: F) -> &mut Self
    where
        F: FnMut(Duration) -> Duration,
    {
        self.duration = scaled_duration(closure(self.duration(movie_timescale)), movie_timescale);
        self
    }
}

impl ParseAtomData for TrackHeaderAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, TKHD);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_tkhd_data.parse(stream(input))?)
    }
}

impl SerializeAtom for TrackHeaderAtom {
    fn atom_type(&self) -> FourCC {
        TKHD
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_tkhd_data(self)
    }
}

mod serializer {
    use crate::atom::{
        tkhd::IDENTITY_MATRIX,
        util::serializer::{fixed_point_16x16, fixed_point_8x8},
    };

    use super::TrackHeaderAtom;

    pub fn serialize_tkhd_data(tkhd: TrackHeaderAtom) -> Vec<u8> {
        let mut data = Vec::new();

        let version: u8 = if tkhd.version == 1
            || tkhd.creation_time > u64::from(u32::MAX)
            || tkhd.modification_time > u64::from(u32::MAX)
            || tkhd.duration > u64::from(u32::MAX)
        {
            1
        } else {
            0
        };

        let be_u32_or_u64 = |v: u64| match version {
            0 => u32::try_from(v).unwrap().to_be_bytes().to_vec(),
            1 => v.to_be_bytes().to_vec(),
            _ => unreachable!(),
        };

        data.push(version);
        data.extend(tkhd.flags);
        data.extend(be_u32_or_u64(tkhd.creation_time));
        data.extend(be_u32_or_u64(tkhd.modification_time));
        data.extend(tkhd.track_id.to_be_bytes());
        data.extend([0u8; 4]); // reserved
        data.extend(be_u32_or_u64(tkhd.duration));
        data.extend([0u8; 8]); // reserved
        data.extend(tkhd.layer.to_be_bytes());
        data.extend(tkhd.alternate_group.to_be_bytes());
        data.extend(fixed_point_8x8(tkhd.volume));
        data.extend([0u8; 2]); // reserved
        data.extend(
            tkhd.matrix
                .unwrap_or(IDENTITY_MATRIX)
                .into_iter()
                .flat_map(|v| v.to_be_bytes()),
        );
        data.extend(fixed_point_16x16(tkhd.width));
        data.extend(fixed_point_16x16(tkhd.height));

        data
    }
}

mod parser {
    use winnow::{
        binary::{be_i16, be_i32, be_u32, be_u64},
        combinator::{seq, trace},
        error::{StrContext, StrContextValue},
        ModalResult, Parser,
    };

    use super::TrackHeaderAtom;
    use crate::atom::{
        tkhd::IDENTITY_MATRIX,
        util::parser::{
            be_u32_as_u64, byte_array, fixed_array, fixed_point_16x16, fixed_point_8x8, flags3,
            version, Stream,
        },
    };

    pub fn parse_tkhd_data(input: &mut Stream<'_>) -> ModalResult<TrackHeaderAtom> {
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
            "tkhd",
            seq!(TrackHeaderAtom {
                version: version
                    .verify(|version| *version <= 1)
                    .context(StrContext::Expected(StrContextValue::Description(
                        "expected version 0 or 1"
                    ))),
                flags: flags3,
                creation_time: be_u32_or_u64(version).context(StrContext::Label("creation_time")),
                modification_time: be_u32_or_u64(version).context(StrContext::Label("modification_time")),
                track_id: be_u32.context(StrContext::Label("track_id")),
                _: byte_array::<4>.context(StrContext::Label("reserved_1")),
                duration: be_u32_or_u64(version),
                _: byte_array::<8>.context(StrContext::Label("reserved_2")),
                layer: be_i16.context(StrContext::Label("layer")),
                alternate_group: be_i16.context(StrContext::Label("alternate_group")),
                volume: fixed_point_8x8.context(StrContext::Label("volume")),
                _: byte_array::<2>.context(StrContext::Label("reserved_3")),
                matrix: matrix.context(StrContext::Label("matrix")),
                width: fixed_point_16x16.context(StrContext::Label("width")),
                height: fixed_point_16x16.context(StrContext::Label("height")),
            })
            .context(StrContext::Label("tkhd")),
        )
        .parse_next(input)
    }

    fn matrix(input: &mut Stream<'_>) -> ModalResult<Option<[i32; 9]>> {
        trace(
            "matrix",
            fixed_array(be_i32).map(|matrix: [i32; 9]| {
                if is_empty_matrix(&matrix) {
                    None
                } else {
                    Some(matrix)
                }
            }),
        )
        .parse_next(input)
    }

    fn is_empty_matrix(matrix: &[i32; 9]) -> bool {
        let empty = [0; 9];
        let identity = IDENTITY_MATRIX;
        matrix == &empty || matrix == &identity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available tkhd test data files
    #[test]
    fn test_tkhd_roundtrip() {
        test_atom_roundtrip::<TrackHeaderAtom>(TKHD);
    }
}
