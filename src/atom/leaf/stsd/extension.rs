use std::fmt;

pub use audio_specific_config::AudioSpecificConfig;

use crate::{
    atom::util::{DebugList, DebugUpperHex},
    FourCC,
};

pub mod audio_specific_config;

#[derive(Clone, PartialEq)]
pub enum StsdExtension {
    Esds(EsdsExtension),
    Btrt(BtrtExtension),
    Unknown { fourcc: FourCC, data: Vec<u8> },
}

impl fmt::Debug for StsdExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StsdExtension::Btrt(btrt) => fmt::Debug::fmt(btrt, f),
            StsdExtension::Esds(esds) => fmt::Debug::fmt(esds, f),
            StsdExtension::Unknown { fourcc, data } => f
                .debug_struct("Unknown")
                .field("fourcc", &fourcc)
                .field("data", &DebugList::new(data.iter().map(DebugUpperHex), 10))
                .finish(),
        }
    }
}

trait Descriptor {
    const TAG: u8;
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct EsdsExtension {
    pub version: u8,
    pub flags: [u8; 3],
    pub es_descriptor: EsDescriptor,
}

impl EsdsExtension {
    const TYPE: FourCC = FourCC::new(b"esds");
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct EsDescriptor {
    pub es_id: u16,
    pub depends_on_es_id: Option<u16>,
    pub url: Option<String>,
    pub ocr_es_id: Option<u16>,
    pub stream_priority: u8,
    pub decoder_config_descriptor: Option<DecoderConfigDescriptor>,
    pub sl_config_descriptor: Option<SlConfigDescriptor>,
}

impl Descriptor for EsDescriptor {
    const TAG: u8 = 0x03;
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct DecoderConfigDescriptor {
    pub object_type_indication: u8,
    pub stream_type: u8,
    pub upstream: bool,
    pub buffer_size_db: u32,
    pub max_bitrate: u32,
    pub avg_bitrate: u32,
    pub decoder_specific_info: Option<DecoderSpecificInfo>,
}

impl Descriptor for DecoderConfigDescriptor {
    const TAG: u8 = 0x04;
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecoderSpecificInfo {
    Audio(AudioSpecificConfig),
    Unknown(Vec<u8>),
}

impl Descriptor for DecoderSpecificInfo {
    const TAG: u8 = 0x05;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlConfigDescriptor {
    pub predefined: u8,
}

impl Descriptor for SlConfigDescriptor {
    const TAG: u8 = 0x06;
}

#[derive(Debug, Clone, PartialEq)]
pub struct BtrtExtension {
    pub buffer_size_db: u32,
    pub max_bitrate: u32,
    pub avg_bitrate: u32,
}

impl BtrtExtension {
    const TYPE: FourCC = FourCC::new(b"btrt");
}

pub(super) mod serializer {
    use crate::{
        atom::{
            stsd::{
                extension::{
                    audio_specific_config::serializer::serialize_audio_specific_config,
                    DecoderConfigDescriptor, Descriptor, EsDescriptor, SlConfigDescriptor,
                },
                BtrtExtension, DecoderSpecificInfo, EsdsExtension, StsdExtension,
            },
            util::serializer::{
                be_u24, bits::Packer, pascal_string, prepend_size_exclusive,
                prepend_size_inclusive, SizeU32, SizeU32OrU64, SizeVLQ,
            },
        },
        FourCC,
    };

    pub fn serialize_stsd_extensions(extensions: Vec<StsdExtension>) -> Vec<u8> {
        extensions
            .into_iter()
            .flat_map(serialize_stsd_extension)
            .collect::<Vec<_>>()
    }

    fn serialize_stsd_extension(extension: StsdExtension) -> Vec<u8> {
        match extension {
            StsdExtension::Esds(esds) => {
                serialize_box(EsdsExtension::TYPE, serialize_esds_extension(esds))
            }
            StsdExtension::Btrt(btrt) => {
                serialize_box(BtrtExtension::TYPE, serialize_btrt_extension(btrt))
            }
            StsdExtension::Unknown { fourcc, data } => serialize_box(fourcc, data),
        }
    }

    fn serialize_esds_extension(esds: EsdsExtension) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(esds.version);
        data.extend(esds.flags);
        data.extend(serialize_es_descriptor(esds.es_descriptor));
        data
    }

    fn serialize_es_descriptor(es_desc: EsDescriptor) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend(es_desc.es_id.to_be_bytes());

        let mut flags = Packer::new();
        flags.push_bool(es_desc.depends_on_es_id.is_some());
        flags.push_bool(es_desc.url.is_some());
        flags.push_bool(es_desc.ocr_es_id.is_some());
        flags.push_n::<5>(es_desc.stream_priority);
        data.push(Vec::from(flags)[0]);

        if let Some(depends_on_es_id) = es_desc.depends_on_es_id {
            data.extend(depends_on_es_id.to_be_bytes());
        }

        if let Some(url) = es_desc.url {
            data.extend(pascal_string(url));
        }

        if let Some(ocr_es_id) = es_desc.ocr_es_id {
            data.extend(ocr_es_id.to_be_bytes());
        }

        if let Some(decoder_config) = es_desc.decoder_config_descriptor {
            data.extend(serialize_decoder_config(decoder_config));
        }

        if let Some(sl_config) = es_desc.sl_config_descriptor {
            data.extend(serialize_sl_config(sl_config));
        }

        serialize_descriptor(EsDescriptor::TAG, data)
    }

    fn serialize_decoder_config(decoder_config: DecoderConfigDescriptor) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(decoder_config.object_type_indication);

        let mut stream_info = Packer::new();
        stream_info.push_n::<6>(decoder_config.stream_type);
        stream_info.push_bool(decoder_config.upstream);
        stream_info.push_bool(true); // reserved
        data.push(Vec::from(stream_info)[0]);

        data.extend(be_u24(decoder_config.buffer_size_db));

        data.extend(decoder_config.max_bitrate.to_be_bytes());
        data.extend(decoder_config.avg_bitrate.to_be_bytes());

        if let Some(decoder_info) = decoder_config.decoder_specific_info {
            let decoder_info_bytes = match decoder_info {
                DecoderSpecificInfo::Audio(c) => serialize_audio_specific_config(c),
                DecoderSpecificInfo::Unknown(c) => c,
            };
            data.extend(serialize_descriptor(
                DecoderSpecificInfo::TAG,
                decoder_info_bytes,
            ));
        }

        serialize_descriptor(DecoderConfigDescriptor::TAG, data)
    }

    fn serialize_sl_config(sl_config: SlConfigDescriptor) -> Vec<u8> {
        serialize_descriptor(SlConfigDescriptor::TAG, vec![sl_config.predefined])
    }

    fn serialize_btrt_extension(btrt: BtrtExtension) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(btrt.buffer_size_db.to_be_bytes());
        data.extend(btrt.max_bitrate.to_be_bytes());
        data.extend(btrt.avg_bitrate.to_be_bytes());
        data
    }

    fn serialize_descriptor(tag: u8, descriptor_data: Vec<u8>) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(tag);
        data.extend(prepend_size_exclusive::<SizeVLQ<SizeU32>, _>(move || {
            descriptor_data
        }));
        data
    }

    fn serialize_box(fourcc: FourCC, box_data: Vec<u8>) -> Vec<u8> {
        prepend_size_inclusive::<SizeU32OrU64, _>(move || {
            let mut data = Vec::new();
            data.extend(fourcc.into_bytes());
            data.extend(box_data);
            data
        })
    }
}

pub(super) mod parser {
    use winnow::{
        binary::{be_u16, be_u24, be_u32, bits, length_and_then, u8},
        combinator::{opt, repeat, seq, trace},
        error::{ContextError, ErrMode, StrContext},
        token::literal,
        ModalResult, Parser,
    };

    use crate::atom::{
        stsd::extension::audio_specific_config::parser::parse_audio_specific_config,
        util::parser::{
            atom_size, combinators::inclusive_length_and_then, flags3, fourcc, pascal_string,
            rest_vec, variable_length_be_u32, version, Stream,
        },
    };

    use super::*;

    pub fn parse_stsd_extensions(input: &mut Stream<'_>) -> ModalResult<Vec<StsdExtension>> {
        repeat(0.., parse_stsd_extension).parse_next(input)
    }

    pub fn parse_stsd_extension(input: &mut Stream<'_>) -> ModalResult<StsdExtension> {
        inclusive_length_and_then(
            atom_size,
            move |input: &mut Stream<'_>| -> ModalResult<StsdExtension> {
                let fourcc = fourcc.parse_next(input)?;

                Ok(match fourcc {
                    EsdsExtension::TYPE => {
                        parse_esds_box.map(StsdExtension::Esds).parse_next(input)?
                    }
                    BtrtExtension::TYPE => {
                        parse_btrt_box.map(StsdExtension::Btrt).parse_next(input)?
                    }
                    _ => StsdExtension::Unknown {
                        fourcc,
                        data: rest_vec.parse_next(input)?,
                    },
                })
            },
        )
        .parse_next(input)
    }

    fn parse_esds_box(input: &mut Stream<'_>) -> ModalResult<EsdsExtension> {
        seq!(EsdsExtension {
            version: version,
            flags: flags3,
            es_descriptor: parse_es_descriptor,
        })
        .parse_next(input)
    }

    fn parse_es_descriptor(input: &mut Stream<'_>) -> ModalResult<EsDescriptor> {
        parse_descriptor(move |input: &mut Stream<'_>| {
            let es_id = be_u16.parse_next(input)?;

            struct Flags {
                stream_dependence_flag: bool,
                url_flag: bool,
                ocr_stream_flag: bool,
                stream_priority: u8,
            }
            let Flags {
                stream_dependence_flag,
                url_flag,
                ocr_stream_flag,
                stream_priority,
            } = bits::bits(
                move |input: &mut (Stream<'_>, usize)| -> ModalResult<Flags> {
                    seq!(Flags {
                        stream_dependence_flag: bits::bool
                            .context(StrContext::Label("stream_dependency_flag")),
                        url_flag: bits::bool.context(StrContext::Label("url_flag")),
                        ocr_stream_flag: bits::bool.context(StrContext::Label("ocr_stream_flag")),
                        stream_priority: bits::take(5usize)
                            .context(StrContext::Label("stream_priority")),
                    })
                    .parse_next(input)
                },
            )
            .parse_next(input)?;

            let depends_on_es_id = if stream_dependence_flag {
                Some(be_u16.parse_next(input)?)
            } else {
                None
            };

            let url = if url_flag {
                Some(pascal_string.parse_next(input)?)
            } else {
                None
            };

            let ocr_es_id = if ocr_stream_flag {
                Some(be_u16.parse_next(input)?)
            } else {
                None
            };

            let decoder_config_descriptor =
                opt(parse_decoder_config_descriptor).parse_next(input)?;

            let sl_config_descriptor = opt(parse_sl_config_descriptor).parse_next(input)?;

            Ok(EsDescriptor {
                es_id,
                depends_on_es_id,
                url,
                ocr_es_id,
                stream_priority,
                decoder_config_descriptor,
                sl_config_descriptor,
            })
        })
        .parse_next(input)
    }

    fn parse_decoder_config_descriptor(
        input: &mut Stream<'_>,
    ) -> ModalResult<DecoderConfigDescriptor> {
        parse_descriptor(move |input: &mut Stream<'_>| {
            let object_type_indication = u8.parse_next(input)?;

            struct StreamInfo {
                stream_type: u8,
                upstream: bool,
            }
            let StreamInfo {
                stream_type,
                upstream,
            } = bits::bits(
                move |input: &mut (Stream<'_>, usize)| -> ModalResult<StreamInfo> {
                    seq!(StreamInfo {
                        stream_type: bits::take(6usize).context(StrContext::Label("stream_type")),
                        upstream: bits::bool.context(StrContext::Label("upstream")),
                        _: bits::bool.context(StrContext::Label("reserved")),
                    })
                    .parse_next(input)
                },
            )
            .parse_next(input)?;

            let buffer_size_db = be_u24.parse_next(input)?;
            let max_bitrate = be_u32.parse_next(input)?;
            let avg_bitrate = be_u32.parse_next(input)?;

            // Parse DecoderSpecificInfo if present
            let decoder_specific_info = opt(move |input: &mut Stream<'_>| {
                parse_descriptor(match stream_type {
                    5 => |input: &mut Stream<'_>| {
                        parse_audio_specific_config
                            .map(DecoderSpecificInfo::Audio)
                            .context(StrContext::Label("audio_specific_config"))
                            .parse_next(input)
                    },
                    _ => |input: &mut Stream<'_>| {
                        rest_vec
                            .map(DecoderSpecificInfo::Unknown)
                            .context(StrContext::Label("unknown"))
                            .parse_next(input)
                    },
                })
                .parse_next(input)
            })
            .parse_next(input)?;

            Ok(DecoderConfigDescriptor {
                object_type_indication,
                stream_type,
                upstream,
                buffer_size_db,
                max_bitrate,
                avg_bitrate,
                decoder_specific_info,
            })
        })
        .parse_next(input)
    }

    fn parse_sl_config_descriptor(input: &mut Stream<'_>) -> ModalResult<SlConfigDescriptor> {
        trace(
            "parse_sl_config_descriptor",
            parse_descriptor(seq!(SlConfigDescriptor {
                predefined: u8.context(StrContext::Label("predefined")),
            })),
        )
        .parse_next(input)
    }

    fn parse_descriptor<'i, Output, ParseDescriptor>(
        mut parser: ParseDescriptor,
    ) -> impl Parser<Stream<'i>, Output, ErrMode<ContextError>>
    where
        ParseDescriptor: Parser<Stream<'i>, Output, ErrMode<ContextError>>,
        Output: Descriptor,
    {
        trace("parse_descriptor", move |input: &mut Stream<'i>| {
            literal(<Output as Descriptor>::TAG)
                .context(StrContext::Label("tag"))
                .parse_next(input)?;
            length_and_then(variable_length_be_u32, parser.by_ref()).parse_next(input)
        })
    }

    fn parse_btrt_box(input: &mut Stream<'_>) -> ModalResult<BtrtExtension> {
        trace(
            "parse_btrt_box",
            seq!(BtrtExtension {
                buffer_size_db: be_u32.context(StrContext::Label("buffer_size_db")),
                max_bitrate: be_u32.context(StrContext::Label("max_bitrate")),
                avg_bitrate: be_u32.context(StrContext::Label("avg_bitrate")),
            }),
        )
        .parse_next(input)
    }
}
