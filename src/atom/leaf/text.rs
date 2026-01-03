use crate::{atom::FourCC, parser::ParseAtomData, writer::SerializeAtom, ParseError};

pub const TEXT: FourCC = FourCC::new(b"text");

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

impl ParseAtomData for TextMediaInfoAtom {
    fn parse_atom_data(atom_type: FourCC, input: &[u8]) -> Result<Self, ParseError> {
        crate::atom::util::parser::assert_atom_type!(atom_type, TEXT);
        use crate::atom::util::parser::stream;
        use winnow::Parser;
        Ok(parser::parse_text_data.parse(stream(input))?)
    }
}

impl SerializeAtom for TextMediaInfoAtom {
    fn atom_type(&self) -> FourCC {
        TEXT
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
        util::parser::{fixed_array, Stream},
    };
    use winnow::{binary::be_i32, combinator::seq, error::StrContext, ModalResult, Parser};

    pub fn parse_text_data(input: &mut Stream<'_>) -> ModalResult<TextMediaInfoAtom> {
        seq!(TextMediaInfoAtom {
            matrix: fixed_array(be_i32).context(StrContext::Label("matrix")),
        })
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip;

    /// Test round-trip for all available text test data files
    #[test]
    fn test_text_roundtrip() {
        test_atom_roundtrip::<TextMediaInfoAtom>(TEXT);
    }
}
