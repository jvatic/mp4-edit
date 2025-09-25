use crate::{
    atom::{
        stsd::extension::{
            btrt::{serializer::serialize_btrt_extension, BtrtExtension},
            esds::{serializer::serialize_esds_extension, EsdsExtension},
        },
        util::serializer::SerializeSize,
    },
    FourCC,
};

mod audio_specific_config;

#[derive(Debug, Clone, PartialEq)]
pub enum StsdExtension {
    Esds(EsdsExtension),
    Btrt(BtrtExtension),
    Unknown(UnknownExtension),
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnknownExtension {
    pub typ: FourCC,
    pub data: Vec<u8>,
}

impl StsdExtension {
    pub fn to_bytes<Size>(self) -> Vec<u8>
    where
        Size: SerializeSize,
    {
        match self {
            Self::Esds(esds) => serialize_esds_extension(esds),
            Self::Btrt(btrt) => serialize_btrt_extension(btrt),
            Self::Unknown(ext) => serializer::serialize_unknown_extension::<Size>(ext),
        }
    }
}

pub mod btrt;
pub mod esds;

pub mod serializer {
    pub use crate::atom::{
        stsd::{extension::UnknownExtension, BtrtExtension},
        util::serializer::{prepend_size, SerializeSize, SizeU32},
    };

    pub fn serialize_unknown_extension<Size>(ext: UnknownExtension) -> Vec<u8>
    where
        Size: SerializeSize,
    {
        prepend_size::<Size, _>(move || {
            let mut data = Vec::with_capacity(ext.data.len() + 4);
            data.extend(ext.typ.0);
            data.extend(ext.data);
            data
        })
    }
}

pub mod parser {
    use winnow::{
        combinator::{empty, seq, trace},
        error::{ContextError, ErrMode, StrContext},
        Parser,
    };

    use crate::{
        atom::{
            stsd::{
                extension::{
                    btrt::{parser::parse_btrt_extension, BTRT},
                    esds::{parser::parse_esds_extension, ESDS},
                    UnknownExtension,
                },
                StsdExtension,
            },
            util::parser::{rest_vec, Stream},
        },
        FourCC,
    };

    pub fn parse_stsd_extension<'i>(
        typ: FourCC,
    ) -> impl Parser<Stream<'i>, StsdExtension, ErrMode<ContextError>> {
        trace(
            "parse_stsd_extension",
            move |input: &mut Stream| match &typ.0 {
                BTRT => parse_btrt_extension.parse_next(input),
                ESDS => parse_esds_extension.parse_next(input),
                _ => parse_unknown_extension(typ).parse_next(input),
            },
        )
    }

    fn parse_unknown_extension<'i>(
        typ: FourCC,
    ) -> impl Parser<Stream<'i>, StsdExtension, ErrMode<ContextError>> {
        trace("parse_stsd_extension", move |input: &mut Stream| {
            seq!(UnknownExtension {
                typ: empty.value(typ),
                data: rest_vec.context(StrContext::Label("data")),
            })
            .map(StsdExtension::Unknown)
            .parse_next(input)
        })
    }
}
