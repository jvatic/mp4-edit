use either::Either;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AudioObjectType {
    Null,
    AACMain,
    AACLowComplexity,
    AACScalableSampleRate,
    AACLongTermPrediction,
    SpectralBandReplication,
    AACScalable,
    ErrorResilientAACLowComplexity,
    ErrorResilientAACLongTermPrediction,
    ErrorResilientAACScalable,
    ErrorResilientBSAC,
    ErrorResilientAACLowDelay,
    ParametricStereo,
    Unknown(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    Reserved(u8),
    Explicit(u32),
}

impl SamplingFrequency {
    pub fn as_hz(&self) -> Option<u32> {
        Some(match self {
            Self::Hz96000 => 96_000,
            Self::Hz88200 => 88_200,
            Self::Hz64000 => 64_000,
            Self::Hz48000 => 48_000,
            Self::Hz44100 => 44_100,
            Self::Hz32000 => 32_000,
            Self::Hz24000 => 24_000,
            Self::Hz22050 => 22_050,
            Self::Hz16000 => 16_000,
            Self::Hz12000 => 12_000,
            Self::Hz11025 => 11_025,
            Self::Hz8000 => 8_000,
            Self::Hz7350 => 7_350,
            Self::Reserved(_) => return None,
            Self::Explicit(v) => *v,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelConfiguration {
    Mono,
    Stereo,
    Three,
    Four,
    Five,
    FiveOne,
    SevenOne,
    Reserved(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSpecificConfig {
    pub audio_object_type: AudioObjectType,
    pub sampling_frequency: SamplingFrequency,
    pub channel_configuration: ChannelConfiguration,
    pub extension: Either<SbrPsSpecificConfig, GeneralAudioSpecificConfig>,
    pub extra: Option<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SbrPsSpecificConfig {
    ext_audio_object_type: u8,
    sbr_present: bool,
    ext_sampling_frequency_index: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralAudioSpecificConfig {
    frame_length_flag: bool,
    depends_on_core_coder: bool,
    /// first extension bit, remaining extension bytes
    extensions: Option<(u8, Vec<u8>)>,
}

pub(crate) mod serializer {
    use either::Either;

    use crate::atom::{
        stsd::extension::audio_specific_config::{
            AudioObjectType, AudioSpecificConfig, ChannelConfiguration, GeneralAudioSpecificConfig,
            SamplingFrequency, SbrPsSpecificConfig,
        },
        util::serializer::bits::Packer,
    };

    pub fn serialize_audio_specific_config(asc: AudioSpecificConfig) -> Vec<u8> {
        let mut packer = Packer::new();

        object_type(&mut packer, asc.audio_object_type); // 5 bits
        sampling_frequency(&mut packer, asc.sampling_frequency); // 4 or 28 bits
        channel_configuration(&mut packer, asc.channel_configuration); // 4 bits
        extension(&mut packer, asc.extension);

        let mut data: Vec<u8> = packer.into();
        if let Some(extra) = asc.extra {
            data.extend(extra);
        }
        data
    }

    fn object_type(packer: &mut Packer, audio_object_type: AudioObjectType) {
        let id = match audio_object_type {
            AudioObjectType::Null => 0,
            AudioObjectType::AACMain => 1,
            AudioObjectType::AACLowComplexity => 2,
            AudioObjectType::AACScalableSampleRate => 3,
            AudioObjectType::AACLongTermPrediction => 4,
            AudioObjectType::SpectralBandReplication => 5,
            AudioObjectType::AACScalable => 6,
            AudioObjectType::ErrorResilientAACLowComplexity => 17,
            AudioObjectType::ErrorResilientAACLongTermPrediction => 19,
            AudioObjectType::ErrorResilientAACScalable => 20,
            AudioObjectType::ErrorResilientBSAC => 22,
            AudioObjectType::ErrorResilientAACLowDelay => 23,
            AudioObjectType::ParametricStereo => 29,
            AudioObjectType::Unknown(idx) => idx,
        };
        packer.push_n::<5>(id);
    }

    fn sampling_frequency(packer: &mut Packer, sampling_frequency: SamplingFrequency) {
        let index = match sampling_frequency {
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
            SamplingFrequency::Reserved(index) => index,
            SamplingFrequency::Explicit(hz) => {
                packer.push_n::<4>(0b0000_1111);
                packer.push_n_u32::<24>(hz);
                return;
            }
        };
        packer.push_n::<4>(index);
    }

    fn channel_configuration(packer: &mut Packer, channel_configuration: ChannelConfiguration) {
        let index = match channel_configuration {
            ChannelConfiguration::Mono => 0,
            ChannelConfiguration::Stereo => 1,
            ChannelConfiguration::Three => 2,
            ChannelConfiguration::Four => 3,
            ChannelConfiguration::Five => 4,
            ChannelConfiguration::FiveOne => 5,
            ChannelConfiguration::SevenOne => 6,
            ChannelConfiguration::Reserved(index) => index,
        };
        packer.push_n::<4>(index);
    }

    fn extension(
        packer: &mut Packer,
        extension: Either<SbrPsSpecificConfig, GeneralAudioSpecificConfig>,
    ) {
        match extension {
            Either::Left(cfg) => sbr_ps_specific_config(packer, cfg),
            Either::Right(cfg) => general_audio_specific_config(packer, cfg),
        };
    }

    fn sbr_ps_specific_config(packer: &mut Packer, cfg: SbrPsSpecificConfig) {
        packer.push_n::<5>(cfg.ext_audio_object_type);
        packer.push_n::<1>(cfg.sbr_present as u8);
        packer.push_n::<4>(cfg.ext_sampling_frequency_index);
    }

    fn general_audio_specific_config(packer: &mut Packer, cfg: GeneralAudioSpecificConfig) {
        packer.push_n::<1>(cfg.frame_length_flag as u8);
        packer.push_n::<1>(cfg.depends_on_core_coder as u8);
        if let Some((first_ext_bit, ext_bytes)) = cfg.extensions {
            packer.push_n::<1>(first_ext_bit);
            packer.push_bytes(ext_bytes);
        }
    }
}

pub(crate) mod parser {
    use either::Either;
    use winnow::{
        binary::bits::{self, bits},
        combinator::{opt, seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::{AudioObjectType, AudioSpecificConfig, ChannelConfiguration, SamplingFrequency};

    use crate::atom::{
        stsd::extension::audio_specific_config::{GeneralAudioSpecificConfig, SbrPsSpecificConfig},
        util::parser::{rest_vec, Stream},
    };

    pub fn parse_audio_specific_config(input: &mut Stream<'_>) -> ModalResult<AudioSpecificConfig> {
        trace(
            "parse_audio_specific_config",
            bits(
                move |input: &mut (Stream<'_>, usize)| -> ModalResult<AudioSpecificConfig> {
                    seq!(AudioSpecificConfig {
                        audio_object_type: audio_object_type
                            .context(StrContext::Label("audio_object_type")), // 5 bits
                        sampling_frequency: sampling_frequency
                            .context(StrContext::Label("sampling_frequency")), // 4 or 28 bits
                        channel_configuration: channel_configuration
                            .context(StrContext::Label("channel_configuration")), // 4 bits
                        extension: match audio_object_type {
                            AudioObjectType::SpectralBandReplication
                            | AudioObjectType::ParametricStereo => sbr_ps_specific_config, // 11 bits
                            _ => general_audio_specific_config, // 3 bits + extensions
                        }
                        .context(StrContext::Label("extension")),
                        extra: bits::bytes(opt(rest_vec)).context(StrContext::Label("extra")),
                    })
                    .parse_next(input)
                },
            ),
        )
        .parse_next(input)
    }

    fn audio_object_type<'i>(input: &mut (Stream<'i>, usize)) -> ModalResult<AudioObjectType> {
        trace(
            "audio_object_type",
            move |input: &mut (Stream<'i>, usize)| {
                // 0b0000_0000 -> 0b0001_1111 (or 0u8 -> 31u8)
                let id: u8 = bits::take(5usize).parse_next(input)?;
                Ok(match id {
                    0 => AudioObjectType::Null,
                    1 => AudioObjectType::AACMain,
                    2 => AudioObjectType::AACLowComplexity,
                    3 => AudioObjectType::AACScalableSampleRate,
                    4 => AudioObjectType::AACLongTermPrediction,
                    5 => AudioObjectType::SpectralBandReplication,
                    6 => AudioObjectType::AACScalable,
                    17 => AudioObjectType::ErrorResilientAACLowComplexity,
                    19 => AudioObjectType::ErrorResilientAACLongTermPrediction,
                    20 => AudioObjectType::ErrorResilientAACScalable,
                    22 => AudioObjectType::ErrorResilientBSAC,
                    23 => AudioObjectType::ErrorResilientAACLowDelay,
                    29 => AudioObjectType::ParametricStereo,
                    idx if idx <= 31 => AudioObjectType::Unknown(idx),
                    _ => unreachable!("5 bits max out at 31u8 (11111)"),
                })
            },
        )
        .parse_next(input)
    }

    fn sampling_frequency<'i>(input: &mut (Stream<'i>, usize)) -> ModalResult<SamplingFrequency> {
        trace(
            "sampling_frequency",
            move |input: &mut (Stream<'i>, usize)| {
                // 0b0000_0000 -> 0b0000_1111 (or 0u8 -> 15u8)
                let index: u8 = bits::take(4usize).parse_next(input)?;
                Ok(match index {
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
                    13 => SamplingFrequency::Reserved(13),
                    14 => SamplingFrequency::Reserved(14),
                    15 => SamplingFrequency::Explicit(
                        bits::take(24usize)
                            .context(StrContext::Label("explicit frequency"))
                            .parse_next(input)?,
                    ),
                    _ => unreachable!("4 bits max out at 15u8 (1111)"),
                })
            },
        )
        .parse_next(input)
    }

    fn channel_configuration<'i>(
        input: &mut (Stream<'i>, usize),
    ) -> ModalResult<ChannelConfiguration> {
        trace(
            "channel_configuration",
            move |input: &mut (Stream<'i>, usize)| {
                // 0b0000_0000 -> 0b0000_1111 (or 0u8 -> 15u8)
                let index: u8 = bits::take(4usize).parse_next(input)?;
                Ok(match index {
                    0 => ChannelConfiguration::Mono,
                    1 => ChannelConfiguration::Stereo,
                    2 => ChannelConfiguration::Three,
                    3 => ChannelConfiguration::Four,
                    4 => ChannelConfiguration::Five,
                    5 => ChannelConfiguration::FiveOne,
                    6 => ChannelConfiguration::SevenOne,
                    7..=15 => ChannelConfiguration::Reserved(index),
                    _ => unreachable!("4 bits max out at 15u8 (1111)"),
                })
            },
        )
        .parse_next(input)
    }

    fn sbr_ps_specific_config<'i>(
        input: &mut (Stream<'i>, usize),
    ) -> ModalResult<Either<SbrPsSpecificConfig, GeneralAudioSpecificConfig>> {
        trace(
            "sbr_ps_specific_config",
            move |input: &mut (Stream<'i>, usize)| {
                Ok(Either::Left(
                    seq!(SbrPsSpecificConfig {
                        ext_audio_object_type: bits::take(5usize)
                            .context(StrContext::Label("ext_audio_object_type")),
                        sbr_present: bits::bool.context(StrContext::Label("sbr_present")),
                        ext_sampling_frequency_index: bits::take(4usize)
                            .context(StrContext::Label("ext_sampling_frequency_index")),
                    })
                    .parse_next(input)?,
                ))
            },
        )
        .parse_next(input)
    }

    fn general_audio_specific_config<'i>(
        input: &mut (Stream<'i>, usize),
    ) -> ModalResult<Either<SbrPsSpecificConfig, GeneralAudioSpecificConfig>> {
        trace(
            "general_audio_specific_config",
            move |input: &mut (Stream<'i>, usize)| {
                Ok(Either::Right(
                    seq!(GeneralAudioSpecificConfig {
                        frame_length_flag: bits::bool
                            .context(StrContext::Label("frame_length_flag")),
                        depends_on_core_coder: bits::bool
                            .context(StrContext::Label("depends_on_core_coder")),
                        extensions: extensions.context(StrContext::Label("extensions")),
                    })
                    .parse_next(input)?,
                ))
            },
        )
        .parse_next(input)
    }

    fn extensions<'i>(input: &mut (Stream<'i>, usize)) -> ModalResult<Option<(u8, Vec<u8>)>> {
        trace("extensions", move |input: &mut (Stream<'i>, usize)| {
            Ok(
                match bits::bool
                    .context(StrContext::Label("extension flag"))
                    .parse_next(input)?
                {
                    false => None,
                    _ => Some((
                        bits::take(1usize)
                            .context(StrContext::Label("first ext bit"))
                            .parse_next(input)?,
                        bits::bytes(rest_vec)
                            .context(StrContext::Label("extension bytes"))
                            .parse_next(input)?,
                    )),
                },
            )
        })
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use winnow::Parser;

    use crate::atom::{
        stsd::extension::audio_specific_config::{
            parser::parse_audio_specific_config, serializer::serialize_audio_specific_config,
        },
        test_utils::assert_bytes_equal,
        util::parser::stream,
    };

    #[test]
    fn test_parse_audio_specific_config_roundtrip() {
        let input: Vec<u8> = vec![0b00110001, 0b10001000, 0b11100000];
        let asc = parse_audio_specific_config
            .parse(stream(&input))
            .expect("error parsing audio specific config");
        eprintln!("{asc:#?}");
        let re_encoded = serialize_audio_specific_config(asc.clone());
        let asc2 = parse_audio_specific_config
            .parse(stream(&re_encoded))
            .expect("error parsing audio specific config");
        eprintln!("{asc:#?}");
        // assert_eq!(asc, asc2);
        assert_bytes_equal(&re_encoded, &input);
    }
}
