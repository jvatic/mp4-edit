use std::convert::TryFrom;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseAudioSpecificConfigError {
    #[error("too short")]
    TooShort,
    #[error("invalid audio object type")]
    InvalidAot(u8),
    #[error("invalid sampling frequency index")]
    InvalidSfIndex(u8),
    #[error("invalid channel configuration")]
    InvalidChannel(u8),
}

/// AAC profile (“Audio Object Type”)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioObjectType {
    AacMain, // 1
    AacLc,   // 2
    AacSsr,  // 3
    AacLtp,  // 4
    Sbr,     // 5
    // … up to 31
    Unknown(u8),
}

impl TryFrom<u8> for AudioObjectType {
    type Error = ParseAudioSpecificConfigError;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        let aot = match v {
            1 => AudioObjectType::AacMain,
            2 => AudioObjectType::AacLc,
            3 => AudioObjectType::AacSsr,
            4 => AudioObjectType::AacLtp,
            5 => AudioObjectType::Sbr,
            6..=31 => AudioObjectType::Unknown(v),
            other => return Err(ParseAudioSpecificConfigError::InvalidAot(other)),
        };
        Ok(aot)
    }
}

impl From<AudioObjectType> for u8 {
    fn from(aot: AudioObjectType) -> u8 {
        match aot {
            AudioObjectType::AacMain => 1,
            AudioObjectType::AacLc => 2,
            AudioObjectType::AacSsr => 3,
            AudioObjectType::AacLtp => 4,
            AudioObjectType::Sbr => 5,
            AudioObjectType::Unknown(v) => v.min(31),
        }
    }
}

/// Sampling frequency, either indexed or explicit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingFrequency {
    Hz96000,
    Hz88200,
    Hz64000,
    Hz48000,
    Hz44100,
    Hz32000,
    Hz24000,
    Hz22050,
    Hz16000,
    Hz12000,
    Hz11025,
    Hz8000,
    Hz7350,
    Explicit(u32),
}

// extract index 0–12 or 15
impl SamplingFrequency {
    pub fn index(&self) -> u8 {
        match *self {
            SamplingFrequency::Hz96000 => 0,
            SamplingFrequency::Hz88200 => 1,
            SamplingFrequency::Hz64000 => 2,
            SamplingFrequency::Hz48000 => 3,
            SamplingFrequency::Hz44100 => 4,
            SamplingFrequency::Hz32000 => 5,
            SamplingFrequency::Hz24000 => 6,
            SamplingFrequency::Hz22050 => 7,
            SamplingFrequency::Hz16000 => 8,
            SamplingFrequency::Hz12000 => 9,
            SamplingFrequency::Hz11025 => 10,
            SamplingFrequency::Hz8000 => 11,
            SamplingFrequency::Hz7350 => 12,
            SamplingFrequency::Explicit(_) => 15,
        }
    }

    pub fn as_hz(&self) -> u32 {
        match *self {
            SamplingFrequency::Hz96000 => 96_000,
            SamplingFrequency::Hz88200 => 88_200,
            SamplingFrequency::Hz64000 => 64_000,
            SamplingFrequency::Hz48000 => 48_000,
            SamplingFrequency::Hz44100 => 44_100,
            SamplingFrequency::Hz32000 => 32_000,
            SamplingFrequency::Hz24000 => 24_000,
            SamplingFrequency::Hz22050 => 22_050,
            SamplingFrequency::Hz16000 => 16_000,
            SamplingFrequency::Hz12000 => 12_000,
            SamplingFrequency::Hz11025 => 11_025,
            SamplingFrequency::Hz8000 => 8_000,
            SamplingFrequency::Hz7350 => 7_350,
            SamplingFrequency::Explicit(v) => v,
        }
    }
}

impl TryFrom<u8> for SamplingFrequency {
    type Error = ParseAudioSpecificConfigError;
    fn try_from(idx: u8) -> Result<Self, Self::Error> {
        let sf = match idx {
            0 => SamplingFrequency::Hz96000,
            1 => SamplingFrequency::Hz88200,
            2 => SamplingFrequency::Hz64000,
            3 => SamplingFrequency::Hz48000,
            4 => SamplingFrequency::Hz44100,
            5 => SamplingFrequency::Hz32000,
            6 => SamplingFrequency::Hz24000,
            7 => SamplingFrequency::Hz22050,
            8 => SamplingFrequency::Hz16000,
            9 => SamplingFrequency::Hz12000,
            10 => SamplingFrequency::Hz11025,
            11 => SamplingFrequency::Hz8000,
            12 => SamplingFrequency::Hz7350,
            15 => SamplingFrequency::Explicit(0), // placeholder
            other => return Err(ParseAudioSpecificConfigError::InvalidSfIndex(other)),
        };
        Ok(sf)
    }
}

/// Number of channels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelConfiguration {
    Mono,
    Stereo,
    Three,
    Four,
    Five,
    FiveOne,
    SevenOne,
}

impl TryFrom<u8> for ChannelConfiguration {
    type Error = ParseAudioSpecificConfigError;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        let cfg = match v {
            1 => ChannelConfiguration::Mono,
            2 => ChannelConfiguration::Stereo,
            3 => ChannelConfiguration::Three,
            4 => ChannelConfiguration::Four,
            5 => ChannelConfiguration::Five,
            6 => ChannelConfiguration::FiveOne,
            7 => ChannelConfiguration::SevenOne,
            other => return Err(ParseAudioSpecificConfigError::InvalidChannel(other)),
        };
        Ok(cfg)
    }
}

impl From<ChannelConfiguration> for u8 {
    fn from(ch: ChannelConfiguration) -> u8 {
        match ch {
            ChannelConfiguration::Mono => 1,
            ChannelConfiguration::Stereo => 2,
            ChannelConfiguration::Three => 3,
            ChannelConfiguration::Four => 4,
            ChannelConfiguration::Five => 5,
            ChannelConfiguration::FiveOne => 6,
            ChannelConfiguration::SevenOne => 7,
        }
    }
}

/// The parsed AAC AudioSpecificConfig
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    pub audio_object_type: AudioObjectType,
    pub sampling_frequency: SamplingFrequency,
    pub channel_configuration: ChannelConfiguration,
    pub reserved_bits: u8,
    pub bytes_read: usize, // 2 or 5
}

impl AudioSpecificConfig {
    /// Parse the 2- or 5-byte config
    pub fn parse(data: &[u8]) -> Result<Self, ParseAudioSpecificConfigError> {
        if data.len() < 2 {
            return Err(ParseAudioSpecificConfigError::TooShort);
        }
        let b0 = data[0];
        let b1 = data[1];
        let aot = AudioObjectType::try_from(b0 >> 3)?;
        let idx = ((b0 & 0b0000_0111) << 1) | ((b1 & 0b1000_0000) >> 7);
        let mut sf = SamplingFrequency::try_from(idx)?;
        let ch = ChannelConfiguration::try_from((b1 & 0b0111_1000) >> 3)?;
        let reserved_bits = b1 & 0b0000_0111;

        let (bytes_read, sampling_frequency) = if idx == 15 {
            if data.len() < 5 {
                return Err(ParseAudioSpecificConfigError::TooShort);
            }
            let explicit =
                (u32::from(data[2]) << 16) | (u32::from(data[3]) << 8) | u32::from(data[4]);
            sf = SamplingFrequency::Explicit(explicit);
            (5, sf)
        } else {
            (2, sf)
        };

        Ok(AudioSpecificConfig {
            audio_object_type: aot,
            sampling_frequency,
            channel_configuration: ch,
            reserved_bits,
            bytes_read,
        })
    }

    /// Serialize back to bytes
    pub fn serialize(&self) -> Vec<u8> {
        // first byte: [aot(5) | sf_idx(3 high bits)]
        let aot_u8: u8 = self.audio_object_type.into();
        let sf_idx = self.sampling_frequency.index();
        let b0 = (aot_u8 << 3) | ((sf_idx >> 1) & 0b0000_0111);

        // second byte: [sf_idx(low bit) | ch(4 bits) | reserved_bits(3 bits)]
        let ch_u8: u8 = self.channel_configuration.into();
        let b1 = ((sf_idx & 0b1) << 7) | (ch_u8 << 3) | (self.reserved_bits & 0b0000_0111);

        let mut out = vec![b0, b1];
        if let SamplingFrequency::Explicit(freq) = self.sampling_frequency {
            // append 24-bit big-endian explicit freq
            out.push(((freq >> 16) & 0xFF) as u8);
            out.push(((freq >> 8) & 0xFF) as u8);
            out.push((freq & 0xFF) as u8);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_indexed() {
        let data = [0x13, 0x10];
        let cfg = AudioSpecificConfig::parse(&data).unwrap();
        assert_eq!(cfg.bytes_read, 2);
        assert_eq!(cfg.reserved_bits, 0); // 0x10 & 0b111 = 0
        let ser = cfg.serialize();
        assert_eq!(ser, data);
    }

    #[test]
    fn round_trip_explicit() {
        let mut data = vec![0x2F, 0x88];
        data.extend(&[0x01, 0xE2, 0x40]);
        let cfg = AudioSpecificConfig::parse(&data).unwrap();
        assert_eq!(cfg.bytes_read, 5);
        assert_eq!(cfg.reserved_bits, 0); // 0x88 & 0b111 = 0
        let ser = cfg.serialize();
        assert_eq!(ser, data);
    }

    #[test]
    fn debug_problematic_bytes() {
        // Test the specific bytes that are failing in stsd09.bin
        let data = [0x2B, 0x8A];
        let cfg = AudioSpecificConfig::parse(&data).unwrap();
        println!("Parsed config: {:?}", cfg);
        assert_eq!(cfg.reserved_bits, 2); // 0x8A & 0b111 = 2
        let ser = cfg.serialize();
        println!("Original: {:02X?}", data);
        println!("Serialized: {:02X?}", ser);
        assert_eq!(ser, data, "Round-trip failed for bytes 2B 8A");
    }
}
