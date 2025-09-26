use crate::atom::stsd::extension::audio_specific_config::AudioSpecificConfig;

pub const ESDS: &'static [u8; 4] = b"esds";

#[derive(Default, Debug, Clone, PartialEq)]
pub struct EsdsExtension {
    pub version: u8,
    pub flags: [u8; 3],
    pub es_descriptor: EsDescriptor,
}

trait Descriptor {
    const TAG: u8;
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct EsDescriptor {
    pub es_id: u16,
    pub flags: u8,
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
    // TODO: extract tag
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

pub(crate) mod serializer {
    use crate::atom::stsd::extension::{
        audio_specific_config::serializer::serialize_audio_specific_config,
        esds::SlConfigDescriptor,
    };

    use super::{
        DecoderConfigDescriptor, DecoderSpecificInfo, Descriptor, EsDescriptor, EsdsExtension,
    };

    pub fn serialize_esds_extension(esds: EsdsExtension) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(esds.version);
        data.extend(esds.flags);
        data.extend(serialize_es_descriptor(esds.es_descriptor));
        data
    }

    fn serialize_descriptor<F>(tag: u8, f: F) -> Vec<u8>
    where
        F: FnOnce() -> Vec<u8>,
    {
        let descriptor_data = f();
        let mut data = Vec::with_capacity(descriptor_data.len());

        data.push(tag);

        let mut size = descriptor_data.len() as u32;
        while size > 0 {
            let mut b = (size & 0x7F) as u8;
            size >>= 7;
            if size > 0 {
                b |= 0x80;
            }
            data.push(b);
        }

        data.extend(descriptor_data);
        data
    }

    fn serialize_es_descriptor(es: EsDescriptor) -> Vec<u8> {
        serialize_descriptor(EsDescriptor::TAG, move || {
            let mut data = Vec::new();

            data.extend(es.es_id.to_be_bytes());

            data.push(es.flags);

            // Add DecoderConfigDescriptor if present
            if let Some(decoder_config) = es.decoder_config_descriptor {
                data.extend(serialize_decoder_config_descriptor(decoder_config));
            }

            // Add SLConfigDescriptor if present
            if let Some(sl_config) = es.sl_config_descriptor {
                data.extend(serialize_sl_config_descriptor(sl_config));
            }

            data
        })
    }

    fn serialize_decoder_config_descriptor(cfg: DecoderConfigDescriptor) -> Vec<u8> {
        serialize_descriptor(DecoderConfigDescriptor::TAG, move || {
            let mut data = Vec::new();

            data.push(cfg.object_type_indication);

            let stream_info = (cfg.stream_type << 2) + (cfg.upstream as u8 & 0x02) + 1;
            data.push(stream_info);

            // Buffer size (24-bit)
            let buffer_bytes = cfg.buffer_size_db.to_be_bytes();
            data.extend_from_slice(&buffer_bytes[1..4]);

            data.extend_from_slice(&cfg.max_bitrate.to_be_bytes());
            data.extend_from_slice(&cfg.avg_bitrate.to_be_bytes());

            // Add DecoderSpecificInfo if present
            if let Some(decoder_info) = cfg.decoder_specific_info {
                data.extend(serialize_decoder_specific_info(decoder_info));
            }

            data
        })
    }

    fn serialize_decoder_specific_info(dsi: DecoderSpecificInfo) -> Vec<u8> {
        serialize_descriptor(DecoderSpecificInfo::TAG, move || {
            let mut data = Vec::new();
            match dsi {
                DecoderSpecificInfo::Audio(asc) => {
                    data.extend(serialize_audio_specific_config(asc))
                }
                DecoderSpecificInfo::Unknown(d) => data.extend(d),
            }
            data
        })
    }

    fn serialize_sl_config_descriptor(cfg: SlConfigDescriptor) -> Vec<u8> {
        serialize_descriptor(SlConfigDescriptor::TAG, move || vec![cfg.predefined])
    }
}

pub(crate) mod parser {
    use winnow::{
        binary::{be_u16, be_u32, length_and_then, u8},
        combinator::{opt, seq, trace},
        error::{ContextError, ErrMode, StrContext},
        ModalResult, Parser,
    };

    use super::{
        DecoderConfigDescriptor, Descriptor, EsDescriptor, EsdsExtension, SlConfigDescriptor,
    };

    use crate::atom::{
        stsd::{
            extension::audio_specific_config::parser::parse_audio_specific_config,
            DecoderSpecificInfo, StsdExtension,
        },
        util::parser::{be_u24, flags3, rest_vec, version, Stream},
    };

    pub fn parse_esds_extension(input: &mut Stream<'_>) -> ModalResult<StsdExtension> {
        trace(
            "esds",
            seq!(EsdsExtension {
                version: version,
                flags: flags3,
                es_descriptor: es_descriptor.context(StrContext::Label("es_descriptor")),
            }),
        )
        .map(StsdExtension::Esds)
        .parse_next(input)
    }

    fn descriptor_tag<'i>(tag: u8) -> impl Parser<Stream<'i>, (), ErrMode<ContextError>> {
        trace("descriptor_tag", move |input: &mut Stream<'_>| {
            u8.context(StrContext::Label("tag"))
                .verify(|t| *t == tag)
                .void()
                .parse_next(input)
        })
    }

    fn descriptor_size(input: &mut Stream<'_>) -> ModalResult<usize> {
        trace("descriptor_size", move |input: &mut Stream<'_>| {
            let mut size: u32 = 0;
            for _ in 0..4 {
                let b = u8.parse_next(input)?;
                size = (size << 7) | (b & 0x7F) as u32;
                if b & 0x80 == 0 {
                    break;
                }
            }
            Ok(size as usize)
        })
        .parse_next(input)
    }

    fn es_descriptor(input: &mut Stream<'_>) -> ModalResult<EsDescriptor> {
        trace("es_descriptor", move |input: &mut Stream<'_>| {
            descriptor_tag(EsDescriptor::TAG).parse_next(input)?;
            length_and_then(descriptor_size, move |input: &mut Stream<'_>| {
                let es_id = be_u16
                    .context(StrContext::Label("es_id"))
                    .parse_next(input)?;

                let flags = u8.context(StrContext::Label("flags")).parse_next(input)?;

                let decoder_config_descriptor = opt(decoder_config_descriptor)
                    .context(StrContext::Label("decoder_config_descriptor"))
                    .parse_next(input)?;

                let sl_config_descriptor = opt(sl_config_descriptor)
                    .context(StrContext::Label("sl_config_descriptor"))
                    .parse_next(input)?;

                Ok(EsDescriptor {
                    es_id,
                    flags,
                    decoder_config_descriptor,
                    sl_config_descriptor,
                })
            })
            .parse_next(input)
        })
        .parse_next(input)
    }

    fn decoder_config_descriptor(input: &mut Stream<'_>) -> ModalResult<DecoderConfigDescriptor> {
        trace(
            "decoder_config_descriptor",
            move |input: &mut Stream<'_>| {
                descriptor_tag(DecoderConfigDescriptor::TAG).parse_next(input)?;
                length_and_then(descriptor_size, move |input: &mut Stream<'_>| {
                    let object_type_indication = u8
                        .context(StrContext::Label("object_type_indication"))
                        .parse_next(input)?;

                    let stream_info = u8
                        .context(StrContext::Label("stream_info"))
                        .parse_next(input)?;
                    let stream_type = (stream_info >> 2) & 0x3F;
                    let upstream = (stream_info & 0x02) != 0;

                    let buffer_size_db = be_u24
                        .context(StrContext::Label("buffer_size_db"))
                        .parse_next(input)?;

                    let max_bitrate = be_u32
                        .context(StrContext::Label("max_bitrate"))
                        .parse_next(input)?;

                    let avg_bitrate = be_u32
                        .context(StrContext::Label("avg_bitrate"))
                        .parse_next(input)?;

                    let decoder_specific_info = opt(decoder_specific_info(object_type_indication))
                        .context(StrContext::Label("decoder_specific_info"))
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
            },
        )
        .parse_next(input)
    }

    fn decoder_specific_info<'i>(
        object_type_indication: u8,
    ) -> impl Parser<Stream<'i>, DecoderSpecificInfo, ErrMode<ContextError>> {
        trace("decoder_specific_info", move |input: &mut Stream<'_>| {
            descriptor_tag(DecoderSpecificInfo::TAG).parse_next(input)?;
            length_and_then(descriptor_size, move |input: &mut Stream<'_>| {
                let decoder_specific_info = match object_type_indication {
                    0x40 | 0x66 | 0x67 | 0x68 => parse_audio_specific_config
                        .map(DecoderSpecificInfo::Audio)
                        .parse_next(input)?,
                    _ => {
                        let data = rest_vec.parse_next(input)?;
                        DecoderSpecificInfo::Unknown(data)
                    }
                };
                Ok(decoder_specific_info)
            })
            .parse_next(input)
        })
    }

    fn sl_config_descriptor(input: &mut Stream<'_>) -> ModalResult<SlConfigDescriptor> {
        trace("sl_config_descriptor", move |input: &mut Stream<'_>| {
            descriptor_tag(SlConfigDescriptor::TAG).parse_next(input)?;
            length_and_then(descriptor_size, move |input: &mut Stream<'_>| {
                seq!(SlConfigDescriptor {
                    predefined: u8.context(StrContext::Label("predefined")),
                })
                .parse_next(input)
            })
            .parse_next(input)
        })
        .parse_next(input)
    }

    #[cfg(test)]
    mod tests {
        use crate::atom::stsd::extension::esds::serializer::serialize_esds_extension;
        use crate::atom::stsd::extension::esds::ESDS;
        use crate::atom::test_utils::{assert_bytes_equal, test_stsd_extension_roundtrip};
        use crate::atom::util::parser::stream;

        use super::*;

        /// Test round-trip for all available stsd/esds test data files
        #[test]
        fn test_esds_roundtrip() {
            test_stsd_extension_roundtrip(ESDS);
        }

        #[test]
        fn test_esds_size_padding() {
            // needlessly uses all 4 bytes for each size header
            let input = vec![
                0x00, // version
                0x00, 0x00, 0x00, // flags
                0x03, // tag
                0x80, 0x80, 0x80, 0x23, // size
                0x00, 0x00, 0x00, // data
                0x04, // tag
                0x80, 0x80, 0x80, 0x15, // size
                0x40, 0x15, 0x00, 0x03, 0x00, 0x00, // data
                0x00, 0x3e, 0x80, 0x00, 0x00, 0x3e, 0x80, // data
                0x05, // tag
                0x80, 0x80, 0x80, 0x03, // size
                0x31, 0x88, 0xe0, // data
                0x06, // tag
                0x80, 0x80, 0x80, 0x01, // size
                0x02, // data
            ];

            // compact size headers (1 byte)
            let expected = vec![
                0x00, // version
                0x00, 0x00, 0x00, // flags
                0x03, // tag
                0x1a, //size
                0x00, 0x00, 0x00, // data
                0x04, // tag
                0x12, // size
                0x40, 0x15, 0x00, 0x03, 0x00, 0x00, // data
                0x00, 0x3e, 0x80, 0x00, 0x00, 0x3e, 0x80, // data
                0x05, // tag
                0x03, // size
                0x31, 0x88, 0xe0, // data
                0x06, // tag
                0x01, // size
                0x02, // data
            ];

            let parsed = parse_esds_extension
                .parse(stream(&input))
                .expect("error parsing input");

            let parsed = match parsed {
                StsdExtension::Esds(esds) => esds,
                _ => unreachable!(),
            };

            let re_encoded = serialize_esds_extension(parsed);

            assert_bytes_equal(&re_encoded, &expected);
        }
    }
}
