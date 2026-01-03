use crate::{atom::FourCC, parser::ParseAtomData, writer::SerializeAtom, ParseError};

pub const SMHD: FourCC = FourCC::new(b"smhd");

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

impl ParseAtomData for SoundMediaHeaderAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, SMHD);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_smhd_data.parse(stream(input))?)
    }
}

impl SerializeAtom for SoundMediaHeaderAtom {
    fn atom_type(&self) -> FourCC {
        SMHD
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
    use crate::atom::util::parser::{byte_array, fixed_point_8x8, version, Stream};

    pub fn parse_smhd_data(input: &mut Stream<'_>) -> ModalResult<SoundMediaHeaderAtom> {
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
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available smhd test data files
    #[test]
    fn test_smhd_roundtrip() {
        test_atom_roundtrip::<SoundMediaHeaderAtom>(SMHD);
    }
}
