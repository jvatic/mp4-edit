use super::StsdExtension;

#[derive(Default, Debug, Clone)]
pub struct AudioSampleEntry {
    pub version: u16,
    pub channel_count: u16,
    pub sample_size: u16,
    pub predefined: u16,
    /// 16.16 fixed point
    pub sample_rate: f32,
    pub extensions: Vec<StsdExtension>,
}

impl AudioSampleEntry {
    pub fn find_or_create_extension<P, D>(&mut self, pred: P, default_fn: D) -> &mut StsdExtension
    where
        P: Fn(&StsdExtension) -> bool,
        D: FnOnce() -> StsdExtension,
    {
        if let Some(index) = self.extensions.iter().position(pred) {
            return &mut self.extensions[index];
        }
        self.extensions.push(default_fn());
        self.extensions.last_mut().unwrap()
    }
}

pub(super) mod serializer {
    use super::AudioSampleEntry;
    use crate::atom::{
        stsd::extension::serializer::serialize_stsd_extensions, util::serializer::fixed_point_16x16,
    };

    pub fn serialize_audio_sample_entry(audio: AudioSampleEntry) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(audio.version.to_be_bytes());
        data.extend([0u8; 6]); // reserved
        data.extend(audio.channel_count.to_be_bytes());
        data.extend(audio.sample_size.to_be_bytes());
        data.extend(audio.predefined.to_be_bytes());
        data.extend([0u8; 2]); // reserved
        data.extend(fixed_point_16x16(audio.sample_rate));
        data.extend(serialize_stsd_extensions(audio.extensions));
        data
    }
}

pub(super) mod parser {
    use winnow::{binary::be_u16, combinator::seq, error::StrContext, ModalResult, Parser};

    use super::AudioSampleEntry;
    use crate::atom::{
        stsd::{extension::parser::parse_stsd_extensions, SampleEntryData},
        util::parser::{byte_array, fixed_point_16x16, Stream},
    };

    pub fn parse_audio_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        seq!(AudioSampleEntry {
            version: be_u16.verify(|v| *v == 0).context(StrContext::Label("version")),
            _: byte_array::<6>.context(StrContext::Label("reserved")),
            channel_count: be_u16.context(StrContext::Label("channel_count")),
            sample_size: be_u16.context(StrContext::Label("sample_size")),
            predefined: be_u16.context(StrContext::Label("predefined")),
            _: byte_array::<2>.context(StrContext::Label("reserved")),
            sample_rate: fixed_point_16x16.context(StrContext::Label("sample_rate")),
            extensions: parse_stsd_extensions.context(StrContext::Label("extensions")),
        })
        .map(SampleEntryData::Audio)
        .parse_next(input)
    }
}
