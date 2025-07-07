use std::io::{Cursor, Read, Seek, SeekFrom};

use thiserror::Error;

use audio_specific_config::{AudioSpecificConfig, ParseError as AudioSpecificConfigParseError};

mod audio_specific_config;

#[derive(Debug, Clone, PartialEq)]
pub struct StsdExtensionData {
    pub extensions: Vec<StsdExtension>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StsdExtension {
    Esds(EsdsExtension),
    Btrt(BtrtExtension),
    Unknown {
        fourcc: [u8; 4],
        size: u32,
        data: Vec<u8>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EsdsExtension {
    pub version: u8,
    pub flags: [u8; 3],
    pub es_descriptor: EsDescriptor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EsDescriptor {
    pub es_id: u16,
    pub stream_dependence_flag: bool,
    pub url_flag: bool,
    pub ocr_stream_flag: bool,
    pub stream_priority: u8,
    pub decoder_config_descriptor: Option<DecoderConfigDescriptor>,
    pub sl_config_descriptor: Option<SlConfigDescriptor>,
}

#[derive(Debug, Clone, PartialEq)]
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
pub enum ParseError {
    #[error("insufficient data")]
    InsufficientData,
    #[error("invalid format")]
    InvalidFormat,
    #[error("invalid audio specific config: {_0}")]
    InvalidAudioSpecificConfig(#[from] AudioSpecificConfigParseError),
    #[error("io error")]
    IoError(std::io::Error),
}

impl From<std::io::Error> for ParseError {
    fn from(error: std::io::Error) -> Self {
        ParseError::IoError(error)
    }
}

impl StsdExtension {
    pub fn to_bytes(self) -> Vec<u8> {
        match self {
            StsdExtension::Esds(esds) => serialize_box(b"esds", &esds.into_vec()),
            StsdExtension::Btrt(btrt) => serialize_box(b"btrt", &btrt.into_bytes()),
            StsdExtension::Unknown {
                fourcc,
                size: _,
                data,
            } => serialize_box(&fourcc, &data),
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
                DecoderSpecificInfo::Audio(c) => &c.serialize(),
                DecoderSpecificInfo::Unknown(c) => c,
            };
            payload.extend(serialize_descriptor(0x05, decoder_info_bytes));
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

pub fn parse_stsd_extensions(data: &[u8]) -> Result<StsdExtensionData, ParseError> {
    let mut cursor = Cursor::new(data);
    let mut boxes = Vec::new();

    while cursor.position() < data.len() as u64 {
        let box_data = parse_stsd_extension(&mut cursor)?;
        boxes.push(box_data);
    }

    Ok(StsdExtensionData { extensions: boxes })
}

fn parse_stsd_extension(cursor: &mut Cursor<&[u8]>) -> Result<StsdExtension, ParseError> {
    // Read box header (size + fourcc)
    let size = read_u32_be(cursor)?;
    let mut fourcc = [0u8; 4];
    cursor.read_exact(&mut fourcc)?;

    if size < 8 {
        return Err(ParseError::InvalidFormat);
    }

    let payload_size = (size - 8) as usize;
    let mut payload = vec![0u8; payload_size];
    cursor.read_exact(&mut payload)?;

    match &fourcc {
        b"esds" => parse_esds_box(&payload).map(StsdExtension::Esds),
        b"btrt" => parse_btrt_box(&payload).map(StsdExtension::Btrt),
        _ => Ok(StsdExtension::Unknown {
            fourcc,
            size,
            data: payload,
        }),
    }
}

fn parse_esds_box(data: &[u8]) -> Result<EsdsExtension, ParseError> {
    if data.len() < 4 {
        return Err(ParseError::InsufficientData);
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

fn parse_es_descriptor(cursor: &mut Cursor<&[u8]>) -> Result<EsDescriptor, ParseError> {
    // ES Descriptor tag
    let tag = read_u8(cursor)?;
    if tag != 0x03 {
        return Err(ParseError::InvalidFormat);
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
) -> Result<DecoderConfigDescriptor, ParseError> {
    let tag = read_u8(cursor)?;
    if tag != 0x04 {
        return Err(ParseError::InvalidFormat);
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
                5 => DecoderSpecificInfo::Audio(AudioSpecificConfig::parse(&info_bytes)?),
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
) -> Result<SlConfigDescriptor, ParseError> {
    let tag = read_u8(cursor)?;
    if tag != 0x06 {
        return Err(ParseError::InvalidFormat);
    }

    let _length = read_descriptor_length(cursor)?;
    let predefined = read_u8(cursor)?;

    Ok(SlConfigDescriptor { predefined })
}

fn parse_btrt_box(data: &[u8]) -> Result<BtrtExtension, ParseError> {
    if data.len() < 12 {
        return Err(ParseError::InsufficientData);
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
fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8, ParseError> {
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16_be(cursor: &mut Cursor<&[u8]>) -> Result<u16, ParseError> {
    let mut buf = [0u8; 2];
    cursor.read_exact(&mut buf)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u24_be(cursor: &mut Cursor<&[u8]>) -> Result<u32, ParseError> {
    let mut buf = [0u8; 3];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes([0, buf[0], buf[1], buf[2]]))
}

fn read_u32_be(cursor: &mut Cursor<&[u8]>) -> Result<u32, ParseError> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_descriptor_length(cursor: &mut Cursor<&[u8]>) -> Result<u32, ParseError> {
    let mut length = 0u32;
    for _ in 0..4 {
        let byte = read_u8(cursor)?;
        length = (length << 7) | (byte & 0x7F) as u32;
        if (byte & 0x80) == 0 {
            break;
        }
    }
    Ok(length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sample_data() {
        let data = [
            0, 0, 0, 51, 101, 115, 100, 115, 0, 0, 0, 0, 3, 128, 128, 128, 34, 0, 1, 0, 4, 128,
            128, 128, 20, 64, 21, 0, 0, 0, 0, 0, 245, 74, 0, 0, 245, 74, 5, 128, 128, 128, 2, 19,
            144, 6, 128, 128, 128, 1, 2, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 245, 74,
            0, 0, 245, 74,
        ];

        let result = parse_stsd_extensions(&data).expect("Failed to parse data");

        assert_eq!(result.extensions.len(), 2);

        // Check ESDS box
        if let StsdExtension::Esds(esds) = &result.extensions[0] {
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
        if let StsdExtension::Btrt(btrt) = &result.extensions[1] {
            assert_eq!(btrt.buffer_size_db, 0);
            assert_eq!(btrt.max_bitrate, 62794);
            assert_eq!(btrt.avg_bitrate, 62794);
        } else {
            panic!("Expected BTRT box");
        }
    }

    #[test]
    fn test_round_trip_serialization() {
        let original_data = [
            0, 0, 0, 51, 101, 115, 100, 115, 0, 0, 0, 0, 3, 128, 128, 128, 34, 0, 1, 0, 4, 128,
            128, 128, 20, 64, 21, 0, 0, 0, 0, 0, 245, 74, 0, 0, 245, 74, 5, 128, 128, 128, 2, 19,
            144, 6, 128, 128, 128, 1, 2, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 245, 74,
            0, 0, 245, 74,
        ];

        // Parse the original data
        let parsed = parse_stsd_extensions(&original_data).expect("Failed to parse original data");

        // Serialize back to bytes
        let serialized = parsed
            .extensions
            .into_iter()
            .flat_map(|ext| ext.to_bytes())
            .collect::<Vec<_>>();

        // Compare with original
        assert_eq!(serialized, original_data, "Round-trip serialization failed");
    }

    #[test]
    fn test_individual_box_serialization() {
        let original_data = [
            0, 0, 0, 51, 101, 115, 100, 115, 0, 0, 0, 0, 3, 128, 128, 128, 34, 0, 1, 0, 4, 128,
            128, 128, 20, 64, 21, 0, 0, 0, 0, 0, 245, 74, 0, 0, 245, 74, 5, 128, 128, 128, 2, 19,
            144, 6, 128, 128, 128, 1, 2, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 245, 74,
            0, 0, 245, 74,
        ];

        let mut parsed = parse_stsd_extensions(&original_data).expect("Failed to parse data");

        // Test ESDS box serialization
        if let StsdExtension::Esds(esds) = parsed.extensions.swap_remove(0) {
            let esds_serialized = esds.into_vec();
            let expected_esds = &original_data[8..51]; // Skip box header (8 bytes), take payload
            assert_eq!(esds_serialized, expected_esds, "ESDS serialization failed");
        }

        // Test BTRT box serialization
        if let StsdExtension::Btrt(btrt) = parsed.extensions.swap_remove(0) {
            let btrt_serialized = btrt.into_bytes();
            let expected_btrt = &original_data[59..]; // Skip to BTRT payload
            assert_eq!(btrt_serialized, expected_btrt, "BTRT serialization failed");
        }
    }
}
