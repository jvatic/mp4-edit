use std::io::{Cursor, Read, Seek, SeekFrom};

use thiserror::Error;

use audio_specific_config::{
    AudioSpecificConfig, ParseAudioSpecificConfigError as AudioSpecificConfigParseError,
};

use crate::{atom::util::serializer::SerializeSize, FourCC};

mod audio_specific_config;

#[derive(Debug, Clone, PartialEq)]
pub struct StsdExtensionData {
    pub extensions: Vec<StsdExtension>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StsdExtension {
    Esds(EsdsExtension),
    Btrt(BtrtExtension),
    Unknown(UnknownExtension),
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnknownExtension {
    typ: FourCC,
    data: Vec<u8>,
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
    pub stream_dependence_flag: bool,
    pub url_flag: bool,
    pub ocr_stream_flag: bool,
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
    Audio(AudioSpecificConfig, Vec<u8>), // AudioSpecificConfig + extra bytes
    Unknown(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlConfigDescriptor {
    pub predefined: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BtrtExtension {
    pub buffer_size_db: u32,
    pub max_bitrate: u32,
    pub avg_bitrate: u32,
}

#[derive(Debug, Error)]
pub enum ParseStsdExtensionError {
    #[error("insufficient data")]
    InsufficientData,
    #[error("invalid format")]
    InvalidFormat,
    #[error("invalid audio specific config: {_0}")]
    InvalidAudioSpecificConfig(#[from] AudioSpecificConfigParseError),
    #[error("io error")]
    IoError(std::io::Error),
}

impl From<std::io::Error> for ParseStsdExtensionError {
    fn from(error: std::io::Error) -> Self {
        ParseStsdExtensionError::IoError(error)
    }
}

impl StsdExtension {
    pub fn to_bytes<Size>(self) -> Vec<u8>
    where
        Size: SerializeSize,
    {
        match self {
            StsdExtension::Esds(esds) => serialize_box(b"esds", &esds.into_vec()),
            StsdExtension::Btrt(btrt) => serialize_box(b"btrt", &btrt.into_bytes()),
            StsdExtension::Unknown(ext) => serializer::serialize_unknown_extension::<Size>(ext),
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
        if self.stream_dependence_flag {
            flags_byte |= 0x80;
        }
        if self.url_flag {
            flags_byte |= 0x40;
        }
        if self.ocr_stream_flag {
            flags_byte |= 0x20;
        }
        payload.push(flags_byte);

        // Add optional fields (not present in our sample data)

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
        if let Some(ref decoder_info) = self.decoder_specific_info {
            let decoder_info_bytes = match decoder_info {
                DecoderSpecificInfo::Audio(c, extra) => {
                    let mut bytes = c.serialize();
                    bytes.extend_from_slice(extra);
                    bytes
                }
                DecoderSpecificInfo::Unknown(c) => c.clone(),
            };
            payload.extend(serialize_descriptor(0x05, &decoder_info_bytes));
        }

        serialize_descriptor(0x04, &payload)
    }
}

impl SlConfigDescriptor {
    fn into_bytes(self) -> Vec<u8> {
        serialize_descriptor(0x06, &[self.predefined])
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
    let size = 8 + payload.len() as u32;
    let mut result = Vec::new();
    result.extend_from_slice(&size.to_be_bytes());
    result.extend_from_slice(fourcc);
    result.extend_from_slice(payload);
    result
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

pub const ESDS: &'static [u8; 4] = b"esds";
pub const BTRT: &'static [u8; 4] = b"btrt";

pub mod serializer {
    use crate::atom::{
        stsd::{
            extension::{EsDescriptor, UnknownExtension},
            BtrtExtension,
        },
        util::serializer::{prepend_size, SerializeSize, SizeU32, SizeU8},
    };

    pub fn serialize_esds_extension(esds: EsDescriptor) -> Vec<u8> {
        todo!()
    }

    pub fn serialize_btrt_extension(btrt: BtrtExtension) -> Vec<u8> {
        todo!()
    }

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
        combinator::{empty, seq, todo, trace},
        error::{ContextError, ErrMode, StrContext},
        token::rest,
        ModalResult, Parser,
    };

    use super::{BTRT, ESDS};

    use crate::{
        atom::{
            stsd::{extension::UnknownExtension, EsdsExtension, StsdExtension},
            util::parser::{rest_vec, Stream},
        },
        FourCC,
    };

    pub fn parse_esds_extension(input: &mut Stream<'_>) -> ModalResult<StsdExtension> {
        parse_unknown_extension(FourCC(*ESDS)).parse_next(input)
    }

    pub fn parse_btrt_extension(input: &mut Stream<'_>) -> ModalResult<StsdExtension> {
        parse_unknown_extension(FourCC(*BTRT)).parse_next(input)
    }

    pub fn parse_unknown_extension<'i>(
        typ: FourCC,
    ) -> impl Parser<Stream<'i>, StsdExtension, ErrMode<ContextError>> {
        trace("parse_unknown_extension", move |input: &mut Stream| {
            seq!(UnknownExtension {
                typ: empty.value(typ),
                data: rest_vec.context(StrContext::Label("data")),
            })
            .map(StsdExtension::Unknown)
            .parse_next(input)
        })
    }
}

fn parse_esds_box(data: &[u8]) -> Result<EsdsExtension, ParseStsdExtensionError> {
    if data.len() < 4 {
        return Err(ParseStsdExtensionError::InsufficientData);
    }

    let version = data[0];
    let flags = [data[1], data[2], data[3]];

    let mut cursor = Cursor::new(&data[4..]);
    let es_descriptor = parse_es_descriptor(&mut cursor)?;

    Ok(EsdsExtension {
        version,
        flags,
        es_descriptor,
    })
}

fn parse_es_descriptor(
    cursor: &mut Cursor<&[u8]>,
) -> Result<EsDescriptor, ParseStsdExtensionError> {
    // ES Descriptor tag
    let tag = read_u8(cursor)?;
    if tag != 0x03 {
        return Err(ParseStsdExtensionError::InvalidFormat);
    }

    let _length = read_descriptor_length(cursor)?;
    let es_id = read_u16_be(cursor)?;

    let flags_byte = read_u8(cursor)?;
    let stream_dependence_flag = (flags_byte & 0x80) != 0;
    let url_flag = (flags_byte & 0x40) != 0;
    let ocr_stream_flag = (flags_byte & 0x20) != 0;
    let stream_priority = flags_byte & 0x1F;

    // Skip dependent stream ID if present
    if stream_dependence_flag {
        read_u16_be(cursor)?;
    }

    // Skip URL if present
    if url_flag {
        let url_length = read_u8(cursor)? as usize;
        cursor.seek(SeekFrom::Current(url_length as i64))?;
    }

    // Skip OCR ES ID if present
    if ocr_stream_flag {
        read_u16_be(cursor)?;
    }

    // Parse DecoderConfigDescriptor
    let decoder_config_descriptor = if cursor.position() < cursor.get_ref().len() as u64 {
        Some(parse_decoder_config_descriptor(cursor)?)
    } else {
        None
    };

    // Parse SLConfigDescriptor
    let sl_config_descriptor = if cursor.position() < cursor.get_ref().len() as u64 {
        Some(parse_sl_config_descriptor(cursor)?)
    } else {
        None
    };

    Ok(EsDescriptor {
        es_id,
        stream_dependence_flag,
        url_flag,
        ocr_stream_flag,
        stream_priority,
        decoder_config_descriptor,
        sl_config_descriptor,
    })
}

fn parse_decoder_config_descriptor(
    cursor: &mut Cursor<&[u8]>,
) -> Result<DecoderConfigDescriptor, ParseStsdExtensionError> {
    let tag = read_u8(cursor)?;
    if tag != 0x04 {
        return Err(ParseStsdExtensionError::InvalidFormat);
    }

    let _length = read_descriptor_length(cursor)?;
    let object_type_indication = read_u8(cursor)?;

    let stream_info = read_u8(cursor)?;
    let stream_type = (stream_info >> 2) & 0x3F;
    let upstream = (stream_info & 0x02) != 0;

    let buffer_size_db = read_u24_be(cursor)?;
    let max_bitrate = read_u32_be(cursor)?;
    let avg_bitrate = read_u32_be(cursor)?;

    // Parse DecoderSpecificInfo if present
    let decoder_specific_info = if cursor.position() < cursor.get_ref().len() as u64 {
        let tag = read_u8(cursor)?;
        if tag == 0x05 {
            let length = read_descriptor_length(cursor)? as usize;
            let mut info_bytes = vec![0u8; length];
            cursor.read_exact(&mut info_bytes)?;
            let info = match stream_type {
                5 => {
                    let audio_config = AudioSpecificConfig::parse(&info_bytes)?;
                    let extra_bytes = info_bytes[audio_config.bytes_read..].to_vec();
                    DecoderSpecificInfo::Audio(audio_config, extra_bytes)
                }
                _ => DecoderSpecificInfo::Unknown(info_bytes),
            };
            Some(info)
        } else {
            cursor.seek(SeekFrom::Current(-1))?; // Rewind
            None
        }
    } else {
        None
    };

    Ok(DecoderConfigDescriptor {
        object_type_indication,
        stream_type,
        upstream,
        buffer_size_db,
        max_bitrate,
        avg_bitrate,
        decoder_specific_info,
    })
}

fn parse_sl_config_descriptor(
    cursor: &mut Cursor<&[u8]>,
) -> Result<SlConfigDescriptor, ParseStsdExtensionError> {
    let tag = read_u8(cursor)?;
    if tag != 0x06 {
        return Err(ParseStsdExtensionError::InvalidFormat);
    }

    let _length = read_descriptor_length(cursor)?;
    let predefined = read_u8(cursor)?;

    Ok(SlConfigDescriptor { predefined })
}

fn parse_btrt_box(data: &[u8]) -> Result<BtrtExtension, ParseStsdExtensionError> {
    if data.len() < 12 {
        return Err(ParseStsdExtensionError::InsufficientData);
    }

    let mut cursor = Cursor::new(data);
    let buffer_size_db = read_u32_be(&mut cursor)?;
    let max_bitrate = read_u32_be(&mut cursor)?;
    let avg_bitrate = read_u32_be(&mut cursor)?;

    Ok(BtrtExtension {
        buffer_size_db,
        max_bitrate,
        avg_bitrate,
    })
}

// Helper functions for reading data
fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8, ParseStsdExtensionError> {
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16_be(cursor: &mut Cursor<&[u8]>) -> Result<u16, ParseStsdExtensionError> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u24_be(cursor: &mut Cursor<&[u8]>) -> Result<u32, ParseStsdExtensionError> {
    let mut buf = [0u8; 3];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes([0, buf[0], buf[1], buf[2]]))
}

fn read_u32_be(cursor: &mut Cursor<&[u8]>) -> Result<u32, ParseStsdExtensionError> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_descriptor_length(cursor: &mut Cursor<&[u8]>) -> Result<u32, ParseStsdExtensionError> {
    let mut length = 0u32;
    for _ in 0..4 {
        let byte = read_u8(cursor)?;
        length = (length << 7) | u32::from(byte & 0x7F);
        if (byte & 0x80) == 0 {
            break;
        }
    }
    Ok(length)
}
