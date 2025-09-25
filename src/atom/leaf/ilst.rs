use bon::bon;
use core::fmt;
use derive_more::Deref;
use futures_io::AsyncRead;

use crate::ParseError;
use crate::{
    atom::{
        util::{read_to_end, DebugEllipsis},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
};

pub const ILST: &[u8; 4] = b"ilst";

const DATA_TYPE_TEXT: u32 = 1;
const DATA_TYPE_JPEG: u32 = 13;

#[derive(Clone, Deref)]
pub struct RawData(Vec<u8>);

impl RawData {
    pub fn new(data: impl Into<Vec<u8>>) -> Self {
        RawData(data.into())
    }
}

impl fmt::Debug for RawData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() <= 10 {
            return f.debug_list().entries(self.0.iter()).finish();
        }
        f.debug_list()
            .entries(self.0.iter().take(10))
            .entry(&DebugEllipsis(Some(self.0.len() - 10)))
            .finish()
    }
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ListItemData {
    Text(String),
    Jpeg(RawData),
    Raw(RawData),
}

impl ListItemData {
    fn new(data_type: u32, data: Vec<u8>) -> Self {
        match data_type {
            DATA_TYPE_TEXT => String::from_utf8(data)
                .map_or_else(|e| Self::Raw(RawData(e.into_bytes())), Self::Text),
            DATA_TYPE_JPEG => Self::Jpeg(RawData(data)),
            _ => Self::Raw(RawData(data)),
        }
    }

    fn to_bytes(self: ListItemData) -> Vec<u8> {
        use ListItemData::{Jpeg, Raw, Text};
        match self {
            Text(s) => s.into_bytes(),
            Jpeg(data) | Raw(data) => data.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DataAtom {
    pub data_type: u32,
    pub reserved: u32,
    pub data: ListItemData,
}

impl DataAtom {
    pub fn new(data: ListItemData) -> Self {
        Self {
            data_type: 0,
            reserved: 0,
            data,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetadataItem {
    pub item_type: FourCC,
    pub mean: Option<Vec<u8>>,
    pub name: Option<Vec<u8>>,
    pub data_atoms: Vec<DataAtom>,
}

#[bon]
impl MetadataItem {
    #[builder]
    pub fn new(
        #[builder(into, start_fn)] item_type: FourCC,
        #[builder(into)] data_atoms: Vec<DataAtom>,
    ) -> Self {
        Self {
            item_type,
            mean: None,
            name: None,
            data_atoms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ItemListAtom {
    pub items: Vec<MetadataItem>,
}

impl ParseAtom for ItemListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != ILST {
            return Err(ParseError::new_unexpected_atom(atom_type, ILST));
        }
        let data = read_to_end(reader).await?;
        parser::parse_ilst_data(&data)
    }
}

impl SerializeAtom for ItemListAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*ILST)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_ilst_atom(self)
    }
}

mod serializer {
    use crate::atom::util::serializer::{prepend_size, SizeU32OrU64};

    use super::{
        DataAtom, ItemListAtom, ListItemData, MetadataItem, DATA_TYPE_JPEG, DATA_TYPE_TEXT,
    };

    pub fn serialize_ilst_atom(atom: ItemListAtom) -> Vec<u8> {
        atom.items.into_iter().flat_map(serialize_item).collect()
    }

    fn serialize_item(item: MetadataItem) -> Vec<u8> {
        prepend_size::<SizeU32OrU64, _>(move || {
            let mut item_data = Vec::new();

            item_data.extend(item.item_type.into_bytes());

            if let Some(mean) = item.mean {
                item_data.extend(prepend_size::<SizeU32OrU64, _>(move || {
                    let mut mean_data = Vec::new();
                    mean_data.extend(b"mean");
                    mean_data.extend(mean);
                    mean_data
                }));
            }

            if let Some(name) = item.name {
                item_data.extend(prepend_size::<SizeU32OrU64, _>(move || {
                    let mut name_data = Vec::new();
                    name_data.extend(b"name");
                    name_data.extend(name);
                    name_data
                }));
            }

            for data_atom in item.data_atoms {
                item_data.extend(prepend_size::<SizeU32OrU64, _>(move || {
                    let mut atom_data = Vec::new();
                    atom_data.extend(b"data");
                    atom_data.extend(serialize_data_type(&data_atom));
                    atom_data.extend(data_atom.reserved.to_be_bytes());
                    atom_data.extend(data_atom.data.to_bytes());
                    atom_data
                }));
            }

            item_data
        })
    }

    fn serialize_data_type(data_atom: &DataAtom) -> Vec<u8> {
        match &data_atom.data {
            ListItemData::Text(_) => DATA_TYPE_TEXT,
            ListItemData::Jpeg(_) => DATA_TYPE_JPEG,
            _ => data_atom.data_type,
        }
        .to_be_bytes()
        .to_vec()
    }
}

mod parser {
    use winnow::{
        binary::be_u32,
        combinator::{opt, preceded, repeat, seq, trace},
        error::StrContext,
        token::{literal, rest},
        ModalResult, Parser,
    };

    use super::{DataAtom, ItemListAtom, ListItemData, MetadataItem};
    use crate::atom::util::parser::{
        atom_size, combinators::inclusive_length_and_then, fourcc, rest_vec, stream, Stream,
    };

    pub fn parse_ilst_data(input: &[u8]) -> Result<ItemListAtom, crate::ParseError> {
        parse_ilst_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_ilst_data_inner(input: &mut Stream<'_>) -> ModalResult<ItemListAtom> {
        trace(
            "ilst",
            seq!(ItemListAtom {
                items: repeat(0.., item),
            })
            .context(StrContext::Label("ilst")),
        )
        .parse_next(input)
    }

    fn item(input: &mut Stream<'_>) -> ModalResult<MetadataItem> {
        trace(
            "item",
            inclusive_length_and_then(atom_size, item_inner).context(StrContext::Label("item")),
        )
        .parse_next(input)
    }

    fn item_inner(input: &mut Stream<'_>) -> ModalResult<MetadataItem> {
        seq!(MetadataItem {
            item_type: fourcc,
            mean: opt(inclusive_length_and_then(
                atom_size,
                preceded(literal(b"mean"), rest_vec)
            ))
            .context(StrContext::Label("mean")),
            name: opt(inclusive_length_and_then(
                atom_size,
                preceded(literal(b"name"), rest_vec)
            ))
            .context(StrContext::Label("name")),
            data_atoms: repeat(
                0..,
                inclusive_length_and_then(atom_size, preceded(literal(b"data"), data_atom))
            ),
        })
        .parse_next(input)
    }

    fn data_atom(input: &mut Stream<'_>) -> ModalResult<DataAtom> {
        trace(
            "data_atom",
            seq!(DataAtom {
                data_type: be_u32,
                reserved: be_u32,
                data: rest.map(|data: &[u8]| ListItemData::new(data_type, data.to_vec())),
            })
            .context(StrContext::Label("data_atom")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available ilst test data files
    #[test]
    fn test_ilst_roundtrip() {
        test_atom_roundtrip_sync::<ItemListAtom>(ILST);
    }
}
