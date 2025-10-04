use std::fmt;

use audio_specific_config::AudioSpecificConfig;

use crate::{
    atom::{
        stsd::extension::audio_specific_config::serializer::serialize_audio_specific_config,
        util::{
            serializer::{pascal_string, prepend_size, SizeU32OrU64},
            DebugList, DebugUpperHex,
        },
    },
    FourCC,
};

mod audio_specific_config;

#[derive(Clone, PartialEq)]
pub enum StsdExtension {
    Esds(EsdsExtension),
    Btrt(BtrtExtension),
    Unknown { fourcc: [u8; 4], data: Vec<u8> },
}

impl fmt::Debug for StsdExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StsdExtension::Btrt(btrt) => fmt::Debug::fmt(btrt, f),
            StsdExtension::Esds(esds) => fmt::Debug::fmt(esds, f),
            StsdExtension::Unknown { fourcc, data } => f
                .debug_struct("Unknown")
                .field("fourcc", &FourCC::new(fourcc))
                .field("data", &DebugList::new(data.iter().map(DebugUpperHex), 10))
                .finish(),
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct EsdsExtension {
    pub version: u8,
    pub flags: [u8; 3],
    pub es_descriptor: EsDescriptor,
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

#[derive(Debug, Clone, PartialEq)]
pub enum DecoderSpecificInfo {
    Audio(AudioSpecificConfig),
    Unknown(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlConfigDescriptor {}

#[derive(Debug, Clone, PartialEq)]
pub struct BtrtExtension {
    pub buffer_size_db: u32,
    pub max_bitrate: u32,
    pub avg_bitrate: u32,
}

impl StsdExtension {
    pub fn to_bytes(self) -> Vec<u8> {
        match self {
            StsdExtension::Esds(esds) => serialize_box(b"esds", &esds.into_vec()),
            StsdExtension::Btrt(btrt) => serialize_box(b"btrt", &btrt.into_bytes()),
            StsdExtension::Unknown { fourcc, data } => serialize_box(&fourcc, &data),
        }
    }
}

impl EsdsExtension {
    fn into_vec(self) -> Vec<u8> {
        let mut result = Vec::new();
        result.push(self.version);
        result.extend_from_slice(&self.flags);
        result.extend(self.es_descriptor.into_bytes());
        result
    }
}

impl EsDescriptor {
    fn into_bytes(self) -> Vec<u8> {
        let mut payload = Vec::new();

        // ES ID
        payload.extend_from_slice(&self.es_id.to_be_bytes());

        // Flags byte
        let mut flags_byte = self.stream_priority & 0x1F;
        if self.depends_on_es_id.is_some() {
            flags_byte |= 0x80;
        }
        if self.url.is_some() {
            flags_byte |= 0x40;
        }
        if self.ocr_es_id.is_some() {
            flags_byte |= 0x20;
        }
        payload.push(flags_byte);

        if let Some(depends_on_es_id) = self.depends_on_es_id {
            payload.extend(depends_on_es_id.to_be_bytes());
        }

        if let Some(url) = self.url {
            payload.extend(pascal_string(url));
        }

        if let Some(ocr_es_id) = self.ocr_es_id {
            payload.extend(ocr_es_id.to_be_bytes());
        }

        // Add DecoderConfigDescriptor if present
        if let Some(decoder_config) = self.decoder_config_descriptor {
            payload.extend(decoder_config.into_bytes());
        }

        // Add SLConfigDescriptor if present
        if let Some(sl_config) = self.sl_config_descriptor {
            payload.extend(sl_config.into_bytes());
        }

        serialize_descriptor(0x03, &payload)
    }
}

impl DecoderConfigDescriptor {
    fn into_bytes(self) -> Vec<u8> {
        let mut payload = Vec::new();

        payload.push(self.object_type_indication);

        let stream_info = (self.stream_type << 2) | if self.upstream { 0x02 } else { 0x00 } | 0x01;
        payload.push(stream_info);

        // Buffer size (24-bit)
        let buffer_bytes = self.buffer_size_db.to_be_bytes();
        payload.extend_from_slice(&buffer_bytes[1..4]);

        payload.extend_from_slice(&self.max_bitrate.to_be_bytes());
        payload.extend_from_slice(&self.avg_bitrate.to_be_bytes());

        // Add DecoderSpecificInfo if present
        if let Some(decoder_info) = self.decoder_specific_info {
            let decoder_info_bytes = match decoder_info {
                DecoderSpecificInfo::Audio(c) => serialize_audio_specific_config(c),
                DecoderSpecificInfo::Unknown(c) => c.clone(),
            };
            payload.extend(serialize_descriptor(0x05, &decoder_info_bytes));
        }

        serialize_descriptor(0x04, &payload)
    }
}

impl SlConfigDescriptor {
    fn into_bytes(self) -> Vec<u8> {
        serialize_descriptor(0x06, &[0x02])
    }
}

impl BtrtExtension {
    fn into_bytes(self) -> Vec<u8> {
        let mut result = Vec::new();
        result.extend_from_slice(&self.buffer_size_db.to_be_bytes());
        result.extend_from_slice(&self.max_bitrate.to_be_bytes());
        result.extend_from_slice(&self.avg_bitrate.to_be_bytes());
        result
    }
}

fn serialize_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    prepend_size::<SizeU32OrU64, _>(move || {
        let mut data = Vec::new();
        data.extend_from_slice(fourcc);
        data.extend_from_slice(payload);
        data
    })
}

fn serialize_descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    result.push(tag);
    result.extend(serialize_descriptor_length(payload.len() as u32));
    result.extend_from_slice(payload);
    result
}

fn serialize_descriptor_length(length: u32) -> Vec<u8> {
    // Always use the 4-byte extended format to match the original data format
    // This matches the encoding pattern seen in the original: 128, 128, 128, X
    vec![
        0x80,         // First continuation byte
        0x80,         // Second continuation byte
        0x80,         // Third continuation byte
        length as u8, // Final length byte (no continuation bit)
    ]
}

pub(crate) mod parser {
    use std::ops::Deref;

    use winnow::{
        binary::{be_u16, be_u24, be_u32, length_and_then, u8},
        combinator::{opt, repeat, seq, trace},
        error::{StrContext, StrContextValue},
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

                Ok(match fourcc.deref() {
                    b"esds" => parse_esds_box.map(StsdExtension::Esds).parse_next(input)?,
                    b"btrt" => parse_btrt_box.map(StsdExtension::Btrt).parse_next(input)?,
                    _ => StsdExtension::Unknown {
                        fourcc: fourcc.into_bytes(),
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
        // ES Descriptor tag
        literal(0x03).parse_next(input)?;

        length_and_then(variable_length_be_u32, move |input: &mut Stream<'_>| {
            let es_id = be_u16.parse_next(input)?;

            let flags_byte = u8.parse_next(input)?;
            let stream_dependence_flag = (flags_byte & 0x80) != 0;
            let url_flag = (flags_byte & 0x40) != 0;
            let ocr_stream_flag = (flags_byte & 0x20) != 0;
            let stream_priority = flags_byte & 0x1F;

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

            // Parse DecoderConfigDescriptor
            let decoder_config_descriptor =
                opt(parse_decoder_config_descriptor).parse_next(input)?;

            // Parse SLConfigDescriptor
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
        literal(0x04).parse_next(input)?;

        length_and_then(variable_length_be_u32, move |input: &mut Stream<'_>| {
            let object_type_indication = u8.parse_next(input)?;

            let stream_info = u8.parse_next(input)?;
            let stream_type = (stream_info >> 2) & 0x3F;
            let upstream = (stream_info & 0x02) != 0;

            let buffer_size_db = be_u24.parse_next(input)?;
            let max_bitrate = be_u32.parse_next(input)?;
            let avg_bitrate = be_u32.parse_next(input)?;

            // Parse DecoderSpecificInfo if present
            let decoder_specific_info = opt(move |input: &mut Stream<'_>| {
                literal(0x05).parse_next(input)?;
                length_and_then(
                    variable_length_be_u32,
                    match stream_type {
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
                    },
                )
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
        trace("parse_sl_config_descriptor",
        seq!(SlConfigDescriptor {
            _: literal(0x06).context(StrContext::Expected(StrContextValue::Description("0x06"))),
            _: length_and_then(variable_length_be_u32, u8).context(StrContext::Label("predefined")),
        }))
        .parse_next(input)
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

#[cfg(test)]
mod tests {
    use winnow::Parser;

    use crate::atom::util::parser::stream;

    use super::*;

    #[test]
    fn test_parse_sample_data() {
        let data = [
            0, 0, 0, 51, 101, 115, 100, 115, 0, 0, 0, 0, 3, 128, 128, 128, 34, 0, 1, 0, 4, 128,
            128, 128, 20, 64, 21, 0, 0, 0, 0, 0, 245, 74, 0, 0, 245, 74, 5, 128, 128, 128, 2, 19,
            144, 6, 128, 128, 128, 1, 2, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 245, 74,
            0, 0, 245, 74,
        ];

        let result = parser::parse_stsd_extensions
            .parse(stream(&data))
            .expect("Failed to parse data");

        assert_eq!(result.len(), 2);

        // Check ESDS box
        if let StsdExtension::Esds(esds) = &result[0] {
            assert_eq!(esds.version, 0);
            assert_eq!(esds.flags, [0, 0, 0]);
            assert_eq!(esds.es_descriptor.es_id, 1);

            if let Some(ref decoder_config) = esds.es_descriptor.decoder_config_descriptor {
                assert_eq!(decoder_config.object_type_indication, 64);
                assert_eq!(decoder_config.max_bitrate, 62794);
                assert_eq!(decoder_config.avg_bitrate, 62794);
            }
        } else {
            panic!("Expected ESDS box");
        }

        // Check BTRT box
        if let StsdExtension::Btrt(btrt) = &result[1] {
            assert_eq!(btrt.buffer_size_db, 0);
            assert_eq!(btrt.max_bitrate, 62794);
            assert_eq!(btrt.avg_bitrate, 62794);
        } else {
            panic!("Expected BTRT box");
        }
    }

    #[test]
    fn test_round_trip_serialization() {
        let data = [
            0, 0, 0, 51, 101, 115, 100, 115, 0, 0, 0, 0, 3, 128, 128, 128, 34, 0, 1, 0, 4, 128,
            128, 128, 20, 64, 21, 0, 0, 0, 0, 0, 245, 74, 0, 0, 245, 74, 5, 128, 128, 128, 2, 19,
            144, 6, 128, 128, 128, 1, 2, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 245, 74,
            0, 0, 245, 74,
        ];

        // Parse the original data
        let parsed = parser::parse_stsd_extensions
            .parse(stream(&data))
            .expect("Failed to parse original data");

        // Serialize back to bytes
        let re_encoded = parsed
            .into_iter()
            .flat_map(|ext| ext.to_bytes())
            .collect::<Vec<_>>();

        // Compare with original
        assert_eq!(re_encoded, data, "Round-trip serialization failed");
    }

    #[test]
    fn test_individual_box_serialization() {
        let data = [
            0, 0, 0, 51, 101, 115, 100, 115, 0, 0, 0, 0, 3, 128, 128, 128, 34, 0, 1, 0, 4, 128,
            128, 128, 20, 64, 21, 0, 0, 0, 0, 0, 245, 74, 0, 0, 245, 74, 5, 128, 128, 128, 2, 19,
            144, 6, 128, 128, 128, 1, 2, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 245, 74,
            0, 0, 245, 74,
        ];

        let mut parsed = parser::parse_stsd_extensions
            .parse(stream(&data))
            .expect("Failed to parse data");

        // Test ESDS box serialization
        if let StsdExtension::Esds(esds) = parsed.swap_remove(0) {
            let esds_serialized = esds.into_vec();
            let expected_esds = &data[8..51]; // Skip box header (8 bytes), take payload
            assert_eq!(esds_serialized, expected_esds, "ESDS serialization failed");
        }

        // Test BTRT box serialization
        if let StsdExtension::Btrt(btrt) = parsed.swap_remove(0) {
            let btrt_serialized = btrt.into_bytes();
            let expected_btrt = &data[59..]; // Skip to BTRT payload
            assert_eq!(btrt_serialized, expected_btrt, "BTRT serialization failed");
        }
    }
}
