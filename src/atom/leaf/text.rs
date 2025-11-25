use futures_io::AsyncRead;

use crate::{
    atom::{util::read_to_end, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const TEXT: &[u8; 4] = b"text";

#[derive(Debug, Clone)]
pub struct TextMediaInfoAtom {
    /// 3x3 transformation matrix for text media
    pub matrix: [i32; 9],
}

impl Default for TextMediaInfoAtom {
    fn default() -> Self {
        Self {
            matrix: [65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824],
        }
    }
}

impl ParseAtom for TextMediaInfoAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != TEXT {
            return Err(ParseError::new_unexpected_atom(atom_type, TEXT));
        }

        let data = read_to_end(reader).await?;
        parser::parse_text_data(&data)
    }
}

impl SerializeAtom for TextMediaInfoAtom {
    fn atom_type(&self) -> FourCC {
        FourCC::new(TEXT)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_text_data(self)
    }
}

mod serializer {
    use crate::atom::text::TextMediaInfoAtom;

    pub fn serialize_text_data(text: TextMediaInfoAtom) -> Vec<u8> {
        text.matrix
            .into_iter()
            .flat_map(|v| v.to_be_bytes())
            .collect()
    }
}

mod parser {
    use crate::atom::{
        text::TextMediaInfoAtom,
        util::parser::{fixed_array, stream, Stream},
    };
    use winnow::{binary::be_i32, combinator::seq, error::StrContext, ModalResult, Parser};

    pub fn parse_text_data(data: &[u8]) -> Result<TextMediaInfoAtom, crate::ParseError> {
        parse_text_data_inner
            .parse(stream(data))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_text_data_inner(input: &mut Stream<'_>) -> ModalResult<TextMediaInfoAtom> {
        seq!(TextMediaInfoAtom {
            matrix: fixed_array(be_i32).context(StrContext::Label("matrix")),
        })
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available text test data files
    #[test]
    fn test_text_roundtrip() {
        test_atom_roundtrip_sync::<TextMediaInfoAtom>(TEXT);
    }
}
