use futures_io::AsyncRead;

use crate::{
    atom::{util::read_to_end, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const SMHD: &[u8; 4] = b"smhd";

#[derive(Debug, Clone, Default)]
pub struct SoundMediaHeaderAtom {
    /// Version of the smhd atom format (0)
    pub version: u8,
    /// Flags for the smhd atom (usually all zeros)
    pub flags: [u8; 3],
    /// Audio balance (fixed-point 8.8 format, 0.0 = center)
    /// Negative values favor left channel, positive favor right
    pub balance: f32,
    /// Reserved field
    pub reserved: [u8; 2],
}

impl ParseAtom for SoundMediaHeaderAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != SMHD {
            return Err(ParseError::new_unexpected_atom(atom_type, SMHD));
        }
        let data = read_to_end(reader).await?;
        parser::parse_smhd_data(&data)
    }
}

impl SerializeAtom for SoundMediaHeaderAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*SMHD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_smhd_data(self)
    }
}

mod serializer {
    use crate::atom::util::serializer::fixed_point_8x8;

    use super::SoundMediaHeaderAtom;

    pub fn serialize_smhd_data(smhd: SoundMediaHeaderAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(smhd.version);
        data.extend(smhd.flags);
        data.extend(fixed_point_8x8(smhd.balance));
        data.extend(smhd.reserved);

        data
    }
}

mod parser {
    use winnow::{
        combinator::{seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::SoundMediaHeaderAtom;
    use crate::atom::util::parser::{byte_array, fixed_point_8x8, stream, version, Stream};

    pub fn parse_smhd_data(input: &[u8]) -> Result<SoundMediaHeaderAtom, crate::ParseError> {
        parse_smhd_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_smhd_data_inner(input: &mut Stream<'_>) -> ModalResult<SoundMediaHeaderAtom> {
        trace(
            "smhd",
            seq!(SoundMediaHeaderAtom {
                version: version,
                flags: byte_array.context(StrContext::Label("flags")),
                balance: fixed_point_8x8.context(StrContext::Label("balance")),
                reserved: byte_array.context(StrContext::Label("reserved")),
            })
            .context(StrContext::Label("chpl")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available smhd test data files
    #[test]
    fn test_smhd_roundtrip() {
        test_atom_roundtrip_sync::<SoundMediaHeaderAtom>(SMHD);
    }
}
