use thiserror::Error;

#[derive(Debug, Clone)]
pub struct EsdsExtension {
    pub es_id: u16,
    pub version: u8,
    pub flags: u32,
    pub stream_priority: Option<u8>,
    pub decoder_config: Option<EsdsDecoderConfig>,
    pub sl_config: Option<EsdsSlConfig>,
}

#[derive(Debug, Clone)]
pub struct EsdsDecoderConfig {
    pub object_type_indication: u8,
    pub stream_type: u8,
    pub upstream: bool,
    pub buffer_size_db: u32,
    pub max_bitrate: u32,
    pub avg_bitrate: u32,
    pub decoder: Option<EsdsDecoder>,
}

#[derive(Debug, Clone)]
pub struct EsdsSlConfig {
    pub predefined: u8,
}

#[derive(Debug, Clone)]
pub enum EsdsDecoder {
    Audio(EsdsAudioConfig),
}

#[derive(Debug, Clone)]
pub struct EsdsAudioConfig {
    pub object_type: EsdsAudioType,
    pub sample_frequency: EsdsAudioSampleFrequency,
    pub channel_config: EsdsAudioChannelConfig,
    pub frame_length: EsdsAudioFrameLength,
    pub core_coder: Option<EsdsAudioCoder>,
    pub extensions: Vec<EsdsAudioExtension>,
}

#[derive(Debug, Clone)]
pub enum EsdsAudioType {
    AacMain = 1,
    AacLowComplexity = 2,
    AacSsr = 3,
    AacLtp = 4,
    Sbr = 5,
}

#[derive(Debug, Clone)]
pub enum EsdsAudioSampleFrequency {
    Hz96000 = 0,
    Hz88200 = 1,
    Hz64000 = 2,
    Hz48000 = 3,
    Hz44100 = 4,
    Hz32000 = 5,
    Hz24000 = 6,
    Hz22050 = 7,
    Hz16000 = 8,
    Hz12000 = 9,
    Hz11025 = 10,
    Hz8000 = 11,
    Hz7350 = 12,
    Escape = 15, // Use with explicit frequency
}

#[derive(Debug, Clone)]
pub enum EsdsAudioChannelConfig {
    Mono = 1,
    Stereo = 2,
    Three = 3,
    Surround = 4,
    FivePointOne = 6,
    SevenPointOne = 8,
}

#[derive(Debug, Clone)]
pub enum EsdsAudioFrameLength {
    S1024 = 0,
    S960 = 1,
}

#[derive(Debug, Clone)]
pub struct EsdsAudioCoder {
    pub delay: u16,
}

#[derive(Debug, Clone)]
pub enum EsdsAudioExtension {
    Sbr {
        sample_frequency: EsdsAudioSampleFrequency,
        explicit_frequency: Option<u32>,
    },
    Ps,
    Unknown(Vec<u8>),
}

pub fn parse_esds(data: &[u8]) -> Result<EsdsExtension, ParseError> {
    if data.len() < 4 {
        return Err(ParseError::InsufficientData);
    }

    let version = data[0];
    let flags = u32::from_be_bytes([0, data[1], data[2], data[3]]);

    let offset = 4;

    // Parse ES_Descriptor
    let (es_descriptor, _) = parse_descriptor(&data[offset..])?;

    if let Descriptor::Es(mut es_desc) = es_descriptor {
        es_desc.version = version;
        es_desc.flags = flags;
        Ok(es_desc)
    } else {
        Err(ParseError::InvalidDescriptor)
    }
}

#[derive(Debug)]
enum Descriptor {
    Es(EsdsExtension),
    DecoderConfig(EsdsDecoderConfig),
    DecoderSpecificInfo(Vec<u8>),
    SlConfig(EsdsSlConfig),
    Unknown { tag: u8, data: Vec<u8> },
}

fn parse_descriptor(data: &[u8]) -> Result<(Descriptor, usize), ParseError> {
    if data.is_empty() {
        return Err(ParseError::InsufficientData);
    }

    let tag = data[0];
    let mut offset = 1;

    // Parse variable-length size
    let (size, size_bytes) = parse_variable_length(&data[offset..])?;
    offset += size_bytes;

    if offset + size > data.len() {
        return Err(ParseError::InsufficientData);
    }

    let payload = &data[offset..offset + size];
    let total_consumed = offset + size;

    let descriptor = match tag {
        0x03 => parse_es_descriptor(payload)?,
        0x04 => parse_decoder_config_descriptor(payload)?,
        0x05 => Descriptor::DecoderSpecificInfo(payload.to_vec()),
        0x06 => parse_sl_config_descriptor(payload)?,
        _ => Descriptor::Unknown {
            tag,
            data: payload.to_vec(),
        },
    };

    Ok((descriptor, total_consumed))
}

fn parse_variable_length(data: &[u8]) -> Result<(usize, usize), ParseError> {
    let mut size = 0usize;
    let mut bytes_consumed = 0;

    for &byte in data.iter().take(4) {
        // Max 4 bytes for size
        bytes_consumed += 1;
        size = (size << 7) | ((byte & 0x7F) as usize);

        if (byte & 0x80) == 0 {
            break; // MSB not set, this is the last byte
        }

        if bytes_consumed >= 4 {
            return Err(ParseError::InvalidVariableLength);
        }
    }

    if bytes_consumed == 0 {
        return Err(ParseError::InsufficientData);
    }

    Ok((size, bytes_consumed))
}

fn parse_es_descriptor(data: &[u8]) -> Result<Descriptor, ParseError> {
    if data.len() < 2 {
        return Err(ParseError::InsufficientData);
    }

    let es_id = u16::from_be_bytes([data[0], data[1]]);
    let mut offset = 2;

    let flags = data[offset];
    offset += 1;

    let mut stream_priority = None;

    // Check if streamDependenceFlag is set
    if (flags & 0x80) != 0 {
        if offset + 2 > data.len() {
            return Err(ParseError::InsufficientData);
        }
        // Skip dependsOn_ES_ID
        offset += 2;
    }

    // Check if URL_Flag is set
    if (flags & 0x40) != 0 {
        if offset >= data.len() {
            return Err(ParseError::InsufficientData);
        }
        let url_length = data[offset] as usize;
        offset += 1 + url_length; // Skip URL string
    }

    // Check if OCRstreamFlag is set
    if (flags & 0x20) != 0 {
        if offset + 2 > data.len() {
            return Err(ParseError::InsufficientData);
        }
        // Skip OCR_ES_Id
        offset += 2;
    }

    // Check if streamPriority is present
    if (flags & 0x10) != 0 {
        if offset >= data.len() {
            return Err(ParseError::InsufficientData);
        }
        stream_priority = Some(data[offset]);
        offset += 1;
    }

    // Parse nested descriptors
    let mut decoder_config = None;
    let mut sl_config = None;

    while offset < data.len() {
        match parse_descriptor(&data[offset..]) {
            Ok((desc, consumed)) => {
                match desc {
                    Descriptor::DecoderConfig(dc) => decoder_config = Some(dc),
                    Descriptor::SlConfig(sl) => sl_config = Some(sl),
                    _ => {} // Ignore other descriptors for now
                }
                offset += consumed;
            }
            Err(_) => break,
        }
    }

    Ok(Descriptor::Es(EsdsExtension {
        es_id,
        version: 0,
        flags: 0,
        stream_priority,
        decoder_config,
        sl_config,
    }))
}

fn parse_decoder_config_descriptor(data: &[u8]) -> Result<Descriptor, ParseError> {
    if data.len() < 13 {
        return Err(ParseError::InsufficientData);
    }

    let object_type_indication = data[0];
    let stream_type_flags = data[1];
    let stream_type = (stream_type_flags >> 2) & 0x3F;
    let upstream = (stream_type_flags & 0x02) != 0;

    let buffer_size_db = u32::from_be_bytes([0, data[2], data[3], data[4]]);
    let max_bitrate = u32::from_be_bytes([data[5], data[6], data[7], data[8]]);
    let avg_bitrate = u32::from_be_bytes([data[9], data[10], data[11], data[12]]);

    let mut offset = 13;
    let mut decoder = None;

    // Parse nested DecoderSpecificInfo if present
    while offset < data.len() {
        match parse_descriptor(&data[offset..]) {
            Ok((Descriptor::DecoderSpecificInfo(info), _consumed)) => {
                // Parse decoder-specific info based on object type indication
                decoder = parse_decoder_specific_info(object_type_indication, &info)?;
                break;
            }
            Ok((_, consumed)) => {
                offset += consumed;
            }
            Err(_) => break,
        }
    }

    Ok(Descriptor::DecoderConfig(EsdsDecoderConfig {
        object_type_indication,
        stream_type,
        upstream,
        buffer_size_db,
        max_bitrate,
        avg_bitrate,
        decoder,
    }))
}

fn parse_decoder_specific_info(
    object_type_indication: u8,
    data: &[u8],
) -> Result<Option<EsdsDecoder>, ParseError> {
    match object_type_indication {
        0x40 => {
            // AAC Audio
            let audio_config = parse_aac_audio_specific_config(data)?;
            Ok(Some(EsdsDecoder::Audio(audio_config)))
        }
        _ => {
            // Unknown decoder type, return None
            Ok(None)
        }
    }
}

fn parse_aac_audio_specific_config(data: &[u8]) -> Result<EsdsAudioConfig, ParseError> {
    if data.is_empty() {
        return Err(ParseError::InsufficientData);
    }

    let mut bit_reader = BitReader::new(data);

    // Parse audioObjectType (5 bits)
    let audio_object_type = bit_reader.read_bits(5)? as u8;
    let object_type = match audio_object_type {
        1 => EsdsAudioType::AacMain,
        2 => EsdsAudioType::AacLowComplexity,
        3 => EsdsAudioType::AacSsr,
        4 => EsdsAudioType::AacLtp,
        5 => EsdsAudioType::Sbr,
        _ => return Err(ParseError::UnsupportedAudioObjectType),
    };

    // Parse samplingFrequencyIndex (4 bits)
    let sample_freq_index = bit_reader.read_bits(4)? as u8;
    let sample_frequency = match sample_freq_index {
        0 => EsdsAudioSampleFrequency::Hz96000,
        1 => EsdsAudioSampleFrequency::Hz88200,
        2 => EsdsAudioSampleFrequency::Hz64000,
        3 => EsdsAudioSampleFrequency::Hz48000,
        4 => EsdsAudioSampleFrequency::Hz44100,
        5 => EsdsAudioSampleFrequency::Hz32000,
        6 => EsdsAudioSampleFrequency::Hz24000,
        7 => EsdsAudioSampleFrequency::Hz22050,
        8 => EsdsAudioSampleFrequency::Hz16000,
        9 => EsdsAudioSampleFrequency::Hz12000,
        10 => EsdsAudioSampleFrequency::Hz11025,
        11 => EsdsAudioSampleFrequency::Hz8000,
        12 => EsdsAudioSampleFrequency::Hz7350,
        15 => EsdsAudioSampleFrequency::Escape,
        _ => return Err(ParseError::InvalidSampleFrequency),
    };

    // If escape frequency, read 24-bit explicit frequency
    if matches!(sample_frequency, EsdsAudioSampleFrequency::Escape) {
        let _explicit_freq = bit_reader.read_bits(24)?; // We'd store this if needed
    }

    // Parse channelConfiguration (4 bits)
    let channel_config_value = bit_reader.read_bits(4)? as u8;
    let channel_config = match channel_config_value {
        1 => EsdsAudioChannelConfig::Mono,
        2 => EsdsAudioChannelConfig::Stereo,
        3 => EsdsAudioChannelConfig::Three,
        4 => EsdsAudioChannelConfig::Surround,
        6 => EsdsAudioChannelConfig::FivePointOne,
        8 => EsdsAudioChannelConfig::SevenPointOne,
        _ => return Err(ParseError::InvalidChannelConfig),
    };

    let mut frame_length = EsdsAudioFrameLength::S1024;
    let mut core_coder = None;
    let mut extensions = Vec::new();

    // Parse additional fields based on audio object type
    match object_type {
        EsdsAudioType::AacMain
        | EsdsAudioType::AacLowComplexity
        | EsdsAudioType::AacSsr
        | EsdsAudioType::AacLtp => {
            // Parse frameLengthFlag (1 bit)
            let frame_length_flag = bit_reader.read_bits(1)?;
            frame_length = if frame_length_flag == 0 {
                EsdsAudioFrameLength::S1024
            } else {
                EsdsAudioFrameLength::S960
            };

            // Parse dependsOnCoreCoder (1 bit)
            let depends_on_core_coder = bit_reader.read_bits(1)?;
            if depends_on_core_coder == 1 {
                // Parse coreCoderDelay (14 bits)
                let delay = bit_reader.read_bits(14)? as u16;
                core_coder = Some(EsdsAudioCoder { delay });
            }

            // Parse extensionFlag (1 bit)
            let extension_flag = bit_reader.read_bits(1)?;
            if extension_flag == 1 {
                // Handle extensions - this is simplified
                // In a full implementation, you'd parse various extension types
            }
        }
        EsdsAudioType::Sbr => {
            // SBR has different parsing logic
            // This would need more complex handling in a full implementation
        }
    }

    Ok(EsdsAudioConfig {
        object_type,
        sample_frequency,
        channel_config,
        frame_length,
        core_coder,
        extensions,
    })
}

fn parse_sl_config_descriptor(data: &[u8]) -> Result<Descriptor, ParseError> {
    if data.is_empty() {
        return Err(ParseError::InsufficientData);
    }

    let predefined = data[0];

    Ok(Descriptor::SlConfig(EsdsSlConfig { predefined }))
}

// Helper struct for bit-level reading
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    fn read_bits(&mut self, num_bits: u8) -> Result<u32, ParseError> {
        if num_bits > 32 {
            return Err(ParseError::InvalidBitRead);
        }

        let mut result = 0u32;
        let mut bits_read = 0u8;

        while bits_read < num_bits {
            if self.byte_pos >= self.data.len() {
                return Err(ParseError::InsufficientData);
            }

            let current_byte = self.data[self.byte_pos];
            let bits_available_in_byte = 8 - self.bit_pos;
            let bits_needed = num_bits - bits_read;
            let bits_to_read = bits_available_in_byte.min(bits_needed);

            let mask = (1u8 << bits_to_read) - 1;
            let shift = bits_available_in_byte - bits_to_read;
            let bits_value = (current_byte >> shift) & mask;

            result = (result << bits_to_read) | (bits_value as u32);
            bits_read += bits_to_read;
            self.bit_pos += bits_to_read;

            if self.bit_pos >= 8 {
                self.byte_pos += 1;
                self.bit_pos = 0;
            }
        }

        Ok(result)
    }
}

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("insufficient data")]
    InsufficientData,
    #[error("invalid descriptor")]
    InvalidDescriptor,
    #[error("invalid variable length")]
    InvalidVariableLength,
    #[error("unsupported audio object type")]
    UnsupportedAudioObjectType,
    #[error("invalid sample frequency")]
    InvalidSampleFrequency,
    #[error("invalid channel config")]
    InvalidChannelConfig,
    #[error("invalid bit read")]
    InvalidBitRead,
}

impl From<EsdsExtension> for Vec<u8> {
    fn from(esds: EsdsExtension) -> Vec<u8> {
        let mut result = Vec::new();

        // Add version and flags (typically version=0, flags=0)
        result.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

        // Create ES_Descriptor (tag 0x03)
        let es_descriptor_data = serialize_es_descriptor(&esds);
        result.extend(serialize_descriptor(0x03, &es_descriptor_data));

        result
    }
}

fn serialize_es_descriptor(esds: &EsdsExtension) -> Vec<u8> {
    let mut data = Vec::new();

    // ES_ID (2 bytes)
    data.extend_from_slice(&esds.es_id.to_be_bytes());

    // Flags byte
    let mut flags = 0u8;
    if esds.stream_priority.is_some() {
        flags |= 0x10; // streamPriority flag
    }
    data.push(flags);

    // Add streamPriority if present
    if let Some(priority) = esds.stream_priority {
        data.push(priority);
    }

    // Add DecoderConfigDescriptor if present
    if let Some(decoder_config) = &esds.decoder_config {
        let decoder_data = serialize_decoder_config_descriptor(decoder_config);
        data.extend(serialize_descriptor(0x04, &decoder_data));
    }

    // Add SLConfigDescriptor if present
    if let Some(sl_config) = &esds.sl_config {
        let sl_data = serialize_sl_config_descriptor(sl_config);
        data.extend(serialize_descriptor(0x06, &sl_data));
    }

    data
}

fn serialize_decoder_config_descriptor(config: &EsdsDecoderConfig) -> Vec<u8> {
    let mut data = Vec::new();

    // objectTypeIndication (1 byte)
    data.push(config.object_type_indication);

    // streamType and flags (1 byte)
    let mut stream_type_flags = (config.stream_type & 0x3F) << 2;
    if config.upstream {
        stream_type_flags |= 0x02;
    }
    data.push(stream_type_flags);

    // bufferSizeDB (3 bytes)
    let buffer_bytes = config.buffer_size_db.to_be_bytes();
    data.extend_from_slice(&buffer_bytes[1..4]); // Skip first byte (24-bit value)

    // maxBitrate (4 bytes)
    data.extend_from_slice(&config.max_bitrate.to_be_bytes());

    // avgBitrate (4 bytes)
    data.extend_from_slice(&config.avg_bitrate.to_be_bytes());

    // Add DecoderSpecificInfo if present
    if let Some(decoder) = &config.decoder {
        let specific_info = serialize_decoder_specific_info(decoder);
        data.extend(serialize_descriptor(0x05, &specific_info));
    }

    data
}

fn serialize_decoder_specific_info(decoder: &EsdsDecoder) -> Vec<u8> {
    match decoder {
        EsdsDecoder::Audio(audio_config) => serialize_aac_audio_specific_config(audio_config),
    }
}

fn serialize_aac_audio_specific_config(config: &EsdsAudioConfig) -> Vec<u8> {
    let mut bit_writer = BitWriter::new();

    // Write audioObjectType (5 bits)
    bit_writer.write_bits(config.object_type.clone() as u32, 5);

    // Write samplingFrequencyIndex (4 bits)
    bit_writer.write_bits(config.sample_frequency.clone() as u32, 4);

    // Write channelConfiguration (4 bits)
    bit_writer.write_bits(config.channel_config.clone() as u32, 4);

    // Write additional fields based on audio object type
    match config.object_type {
        EsdsAudioType::AacMain
        | EsdsAudioType::AacLowComplexity
        | EsdsAudioType::AacSsr
        | EsdsAudioType::AacLtp => {
            // Write frameLengthFlag (1 bit)
            let frame_length_flag = match config.frame_length {
                EsdsAudioFrameLength::S1024 => 0,
                EsdsAudioFrameLength::S960 => 1,
            };
            bit_writer.write_bits(frame_length_flag, 1);

            // Write dependsOnCoreCoder (1 bit)
            if let Some(core_coder) = &config.core_coder {
                bit_writer.write_bits(1, 1);
                bit_writer.write_bits(core_coder.delay as u32, 14);
            } else {
                bit_writer.write_bits(0, 1);
            }

            // Write extensionFlag (1 bit)
            if !config.extensions.is_empty() {
                bit_writer.write_bits(1, 1);
                // Extension serialization would go here
            } else {
                bit_writer.write_bits(0, 1);
            }
        }
        EsdsAudioType::Sbr => {
            // SBR serialization would be more complex
        }
    }

    bit_writer.finish()
}

// Helper struct for bit-level writing
struct BitWriter {
    data: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    fn write_bits(&mut self, value: u32, num_bits: u8) {
        let mut remaining_bits = num_bits;
        let mut shifted_value = value;

        while remaining_bits > 0 {
            let bits_available_in_byte = 8 - self.bit_pos;
            let bits_to_write = remaining_bits.min(bits_available_in_byte);

            let mask = (1u32 << bits_to_write) - 1;
            let bits_to_add = (shifted_value >> (remaining_bits - bits_to_write)) & mask;

            self.current_byte |= (bits_to_add as u8) << (bits_available_in_byte - bits_to_write);
            self.bit_pos += bits_to_write;

            if self.bit_pos >= 8 {
                self.data.push(self.current_byte);
                self.current_byte = 0;
                self.bit_pos = 0;
            }

            remaining_bits -= bits_to_write;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.data.push(self.current_byte);
        }
        self.data
    }
}

fn serialize_sl_config_descriptor(config: &EsdsSlConfig) -> Vec<u8> {
    vec![config.predefined]
}

fn serialize_descriptor(tag: u8, payload: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    // Add tag
    result.push(tag);

    // Add variable-length size
    result.extend(encode_variable_length(payload.len()));

    // Add payload
    result.extend_from_slice(payload);

    result
}

fn encode_variable_length(size: usize) -> Vec<u8> {
    if size < 0x80 {
        // Single byte encoding
        vec![size as u8]
    } else if size < 0x4000 {
        // Two byte encoding
        vec![0x80 | ((size >> 7) as u8), (size & 0x7F) as u8]
    } else if size < 0x200000 {
        // Three byte encoding
        vec![
            0x80 | ((size >> 14) as u8),
            0x80 | (((size >> 7) & 0x7F) as u8),
            (size & 0x7F) as u8,
        ]
    } else {
        // Four byte encoding (max supported)
        vec![
            0x80 | ((size >> 21) as u8),
            0x80 | (((size >> 14) & 0x7F) as u8),
            0x80 | (((size >> 7) & 0x7F) as u8),
            (size & 0x7F) as u8,
        ]
    }
}
