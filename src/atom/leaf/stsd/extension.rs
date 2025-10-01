use crate::{
    atom::stsd::extension::{
        btrt::{BtrtExtension, BTRT},
        esds::{EsdsExtension, ESDS},
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

impl StsdExtension {
    pub fn ext_type(&self) -> FourCC {
        match self {
            Self::Esds(_) => FourCC::new(ESDS),
            Self::Btrt(_) => FourCC::new(BTRT),
            Self::Unknown(ext) => ext.typ.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnknownExtension {
    pub typ: FourCC,
    pub data: Vec<u8>,
}

pub mod btrt;
pub mod esds;

pub mod serializer {
    use crate::atom::stsd::{
        extension::{
            btrt::serializer::serialize_btrt_extension, esds::serializer::serialize_esds_extension,
        },
        StsdExtension,
    };
    pub use crate::atom::{
        stsd::{extension::UnknownExtension, BtrtExtension},
        util::serializer::{prepend_size, SerializeSize, SizeU32},
    };

    pub fn serialize_stsd_extension(ext: StsdExtension) -> Vec<u8> {
        match ext {
            StsdExtension::Esds(esds) => serialize_esds_extension(esds),
            StsdExtension::Btrt(btrt) => serialize_btrt_extension(btrt),
            StsdExtension::Unknown(ext) => serialize_unknown_extension(ext),
        }
    }

    fn serialize_unknown_extension(ext: UnknownExtension) -> Vec<u8> {
        ext.data
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
