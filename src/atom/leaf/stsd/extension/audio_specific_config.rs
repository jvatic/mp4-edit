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

impl SamplingFrequency {
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
}

pub(crate) mod serializer {
    use crate::atom::util::serializer::{be_u24, bits::Packer};

    use super::{AudioObjectType, AudioSpecificConfig, ChannelConfiguration, SamplingFrequency};

    pub fn serialize_audio_specific_config(cfg: AudioSpecificConfig) -> Vec<u8> {
        let mut packer = Packer::new();

        let explicit = match cfg.sampling_frequency {
            SamplingFrequency::Explicit(hz) => Some(hz),
            _ => None,
        };

        packer.push_n::<5>(audio_object_type(cfg.audio_object_type));
        packer.push_n::<4>(sampling_frequency_index(cfg.sampling_frequency));
        packer.push_n::<4>(channel_configuration(cfg.channel_configuration));
        packer.push_n::<3>(cfg.reserved_bits);

        if let Some(hz) = explicit {
            packer.push_bytes(be_u24(hz));
        }

        Vec::from(packer)
    }

    fn audio_object_type(aot: AudioObjectType) -> u8 {
        match aot {
            AudioObjectType::AacMain => 1,
            AudioObjectType::AacLc => 2,
            AudioObjectType::AacSsr => 3,
            AudioObjectType::AacLtp => 4,
            AudioObjectType::Sbr => 5,
            AudioObjectType::Unknown(v) => v.min(31),
        }
    }

    fn sampling_frequency_index(sf: SamplingFrequency) -> u8 {
        match sf {
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

    fn channel_configuration(ch: ChannelConfiguration) -> u8 {
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

pub(crate) mod parser {
    use winnow::{
        binary::{be_u24, bits},
        combinator::{alt, backtrack_err, dispatch, empty, fail, seq},
        error::{StrContext, StrContextValue},
        ModalResult, Parser,
    };

    use crate::atom::util::parser::Stream;

    use super::{AudioObjectType, AudioSpecificConfig, ChannelConfiguration, SamplingFrequency};

    pub fn parse_audio_specific_config(input: &mut Stream<'_>) -> ModalResult<AudioSpecificConfig> {
        bits::bits(
            move |input: &mut (Stream<'_>, usize)| -> ModalResult<AudioSpecificConfig> {
                let mut asc = seq!(AudioSpecificConfig {
                    audio_object_type: audio_object_type // 5 bits
                        .context(StrContext::Label("audio_object_type")),
                    sampling_frequency: sampling_frequency // 4 bits
                        .context(StrContext::Label("sampling_frequency")),
                    channel_configuration: channel_configuration // 4 bits
                        .context(StrContext::Label("channel_configuration")),
                    reserved_bits: bits::take(3usize),
                })
                .parse_next(input)?;

                if matches!(asc.sampling_frequency, SamplingFrequency::Explicit(_)) {
                    asc.sampling_frequency = SamplingFrequency::Explicit(
                        bits::bytes(move |input: &mut Stream<'_>| -> ModalResult<u32> {
                            be_u24.parse_next(input) // 3 bytes
                        })
                        .context(StrContext::Label("sampling_frequency"))
                        .context(StrContext::Expected(StrContextValue::Description(
                            "explicit frequency (be_u24)",
                        )))
                        .parse_next(input)?,
                    );
                }

                Ok(asc)
            },
        )
        .parse_next(input)
    }

    fn audio_object_type(input: &mut (Stream<'_>, usize)) -> ModalResult<AudioObjectType> {
        alt((
            dispatch! {bits::take(5usize);
                1 => empty.value(AudioObjectType::AacMain),
                2 => empty.value(AudioObjectType::AacLc),
                3 => empty.value(AudioObjectType::AacSsr),
                4 => empty.value(AudioObjectType::Sbr),
                5 => empty.value(AudioObjectType::Sbr),
                _ => backtrack_err(fail),
            },
            bits::take(5usize)
                .verify(|v: &u8| (6..=31).contains(v))
                .map(AudioObjectType::Unknown),
            fail.context(StrContext::Expected(StrContextValue::Description(
                "0x01..0x1F",
            ))),
        ))
        .parse_next(input)
    }

    fn sampling_frequency(input: &mut (Stream<'_>, usize)) -> ModalResult<SamplingFrequency> {
        dispatch! {bits::take(4usize);
            0 => empty.value(SamplingFrequency::Hz96000),
            1 => empty.value(SamplingFrequency::Hz88200),
            2 => empty.value(SamplingFrequency::Hz64000),
            3 => empty.value(SamplingFrequency::Hz48000),
            4 => empty.value(SamplingFrequency::Hz44100),
            5 => empty.value(SamplingFrequency::Hz32000),
            6 => empty.value(SamplingFrequency::Hz24000),
            7 => empty.value(SamplingFrequency::Hz22050),
            8 => empty.value(SamplingFrequency::Hz16000),
            9 => empty.value(SamplingFrequency::Hz12000),
            10 => empty.value(SamplingFrequency::Hz11025),
            11 => empty.value(SamplingFrequency::Hz8000),
            12 => empty.value(SamplingFrequency::Hz7350),
            15 => empty.value(SamplingFrequency::Explicit(0)), // placeholder
            _ => fail.context(StrContext::Expected(StrContextValue::Description(
                "0x00..0x0C, 0x0F",
            ))),
        }
        .parse_next(input)
    }

    fn channel_configuration(input: &mut (Stream<'_>, usize)) -> ModalResult<ChannelConfiguration> {
        dispatch! {bits::take(4usize);
            1 => empty.value(ChannelConfiguration::Mono),
            2 => empty.value(ChannelConfiguration::Stereo),
            3 => empty.value(ChannelConfiguration::Three),
            4 => empty.value(ChannelConfiguration::Four),
            5 => empty.value(ChannelConfiguration::Five),
            6 => empty.value(ChannelConfiguration::FiveOne),
            7 => empty.value(ChannelConfiguration::SevenOne),
            _ =>
            fail.context(StrContext::Expected(StrContextValue::Description(
                "0x01..0x07",
            ))),
        }
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use winnow::Parser;

    use crate::atom::util::parser::stream;

    use super::{parser::parse_audio_specific_config, serializer::serialize_audio_specific_config};

    #[test]
    fn round_trip_indexed() {
        let data = [0x13, 0x10];
        let cfg = parse_audio_specific_config.parse(stream(&data)).unwrap();
        assert_eq!(cfg.reserved_bits, 0); // 0x10 & 0b111 = 0
        let ser = serialize_audio_specific_config(cfg);
        assert_eq!(ser, data);
    }

    #[test]
    fn round_trip_explicit() {
        let mut data = vec![0x2F, 0x88];
        data.extend(&[0x01, 0xE2, 0x40]);
        let cfg = parse_audio_specific_config.parse(stream(&data)).unwrap();
        assert_eq!(cfg.reserved_bits, 0); // 0x88 & 0b111 = 0
        let ser = serialize_audio_specific_config(cfg);
        assert_eq!(ser, data);
    }

    #[test]
    fn debug_problematic_bytes() {
        let data = [0x2B, 0x8A];
        let cfg = parse_audio_specific_config.parse(stream(&data)).unwrap();
        println!("Parsed config: {:?}", cfg);
        assert_eq!(cfg.reserved_bits, 2); // 0x8A & 0b111 = 2
        let ser = serialize_audio_specific_config(cfg);
        println!("Original: {:02X?}", data);
        println!("Serialized: {:02X?}", ser);
        assert_eq!(ser, data, "Round-trip failed for bytes 2B 8A");
    }
}
