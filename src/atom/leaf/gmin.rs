use futures_io::AsyncRead;

use crate::{
    atom::{
        util::{read_to_end, ColorRgb},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const GMIN: &[u8; 4] = b"gmin";

#[derive(Debug, Clone)]
pub struct BaseMediaInfoAtom {
    pub version: u8,
    pub flags: [u8; 3],
    pub graphics_mode: u16,
    pub op_color: ColorRgb,
    /// fixed point 8x8
    pub balance: f32,
    // reserved: 2 bytes
}

impl Default for BaseMediaInfoAtom {
    fn default() -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            graphics_mode: 64,
            op_color: ColorRgb {
                red: 32768,
                green: 32768,
                blue: 32768,
            },
            balance: 0.0,
        }
    }
}

impl ParseAtom for BaseMediaInfoAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != GMIN {
            return Err(ParseError::new_unexpected_atom(atom_type, GMIN));
        }

        let data = read_to_end(reader).await?;
        parser::parse_gmin_data(&data)
    }
}

impl SerializeAtom for BaseMediaInfoAtom {
    fn atom_type(&self) -> FourCC {
        FourCC::new(GMIN)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_gmin_data(self)
    }
}

mod serializer {
    use crate::atom::{
        gmin::BaseMediaInfoAtom,
        util::serializer::{color_rgb, fixed_point_8x8},
    };

    pub fn serialize_gmin_data(gmin: BaseMediaInfoAtom) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(gmin.version);
        data.extend(gmin.flags);
        data.extend(gmin.graphics_mode.to_be_bytes());
        data.extend(color_rgb(gmin.op_color));
        data.extend(fixed_point_8x8(gmin.balance));
        data.extend([0u8; 2]); // reserved
        data
    }
}

mod parser {
    use crate::atom::{
        gmin::BaseMediaInfoAtom,
        util::parser::{byte_array, color_rgb, fixed_point_8x8, flags3, stream, version, Stream},
    };
    use winnow::{binary::be_u16, combinator::seq, error::StrContext, ModalResult, Parser};

    pub fn parse_gmin_data(data: &[u8]) -> Result<BaseMediaInfoAtom, crate::ParseError> {
        parse_gmin_data_inner
            .parse(stream(data))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_gmin_data_inner(input: &mut Stream<'_>) -> ModalResult<BaseMediaInfoAtom> {
        seq!(BaseMediaInfoAtom {
            version: version,
            flags: flags3,
            graphics_mode: be_u16.context(StrContext::Label("graphics_mode")),
            op_color: color_rgb.context(StrContext::Label("op_color")),
            balance: fixed_point_8x8.context(StrContext::Label("balance")),
            _: byte_array::<2>.context(StrContext::Label("reserved")),
        })
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available gmin test data files
    #[test]
    fn test_gmin_roundtrip() {
        test_atom_roundtrip_sync::<BaseMediaInfoAtom>(GMIN);
    }
}
