use crate::{atom::stsd::StsdExtension, FourCC};

#[derive(Default, Debug, Clone)]
pub struct Mp4aEntryData {
    /// Version (usually 0)
    pub version: u16,
    /// Revision level (usually 0)
    pub revision_level: u16,
    /// Vendor identifier
    pub vendor: [u8; 4],
    /// Number of audio channels
    pub channel_count: u16,
    /// Sample size in bits
    pub sample_size: u16,
    /// Compression ID (usually 0)
    pub compression_id: u16,
    /// Packet size (usually 0)
    pub packet_size: u16,
    /// Sample rate (16.16 fixed point)
    pub sample_rate: f32,
    /// Extension data (codec-specific atoms)
    pub extensions: Vec<StsdExtension>,
}

impl Mp4aEntryData {
    pub const TYPE: FourCC = FourCC::new(b"mp4a");

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

pub mod serializer {
    use crate::atom::{
        stsd::extension::serializer::serialize_stsd_extension,
        util::serializer::{fixed_point_16x16, prepend_size, SizeU32},
    };

    use super::Mp4aEntryData;

    pub fn serialize_mp4a_entry_data(mp4a: Mp4aEntryData) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend(mp4a.version.to_be_bytes());
        data.extend(mp4a.revision_level.to_be_bytes());
        data.extend(mp4a.vendor);
        data.extend(mp4a.channel_count.to_be_bytes());
        data.extend(mp4a.sample_size.to_be_bytes());
        data.extend(mp4a.compression_id.to_be_bytes());
        data.extend(mp4a.packet_size.to_be_bytes());
        data.extend(fixed_point_16x16(mp4a.sample_rate));
        data.extend(mp4a.extensions.into_iter().flat_map(|ext| {
            prepend_size::<SizeU32, _>(move || {
                let mut data = Vec::new();
                data.extend(ext.ext_type().into_bytes());
                data.extend(serialize_stsd_extension(ext));
                data
            })
        }));

        data
    }
}

pub mod parser {
    use winnow::{
        binary::{be_u16, be_u32},
        combinator::{repeat, seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::Mp4aEntryData;
    use crate::atom::{
        stsd::{extension::parser::parse_stsd_extension, SampleEntryData, StsdExtension},
        util::parser::{
            byte_array, combinators::inclusive_length_and_then, fixed_point_16x16, fourcc, Stream,
        },
    };
    pub fn mp4a_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        trace(
            "mp4a_sample_entry",
            seq!(Mp4aEntryData {
                version: be_u16.context(StrContext::Label("version")),
                revision_level: be_u16.context(StrContext::Label("revision_level")),
                // TODO: is this a fourcc?
                vendor: byte_array.context(StrContext::Label("vendor")),
                channel_count: be_u16.context(StrContext::Label("channel_count")),
                sample_size: be_u16.context(StrContext::Label("sample_size")),
                compression_id: be_u16.context(StrContext::Label("compression_id")),
                packet_size: be_u16.context(StrContext::Label("packet_size")),
                sample_rate: fixed_point_16x16.context(StrContext::Label("sample_rate")),
                extensions: extensions.context(StrContext::Label("extensions")),
            })
            .map(SampleEntryData::Mp4a),
        )
        .context(StrContext::Label("mp4a"))
        .parse_next(input)
    }

    pub fn extensions(input: &mut Stream<'_>) -> ModalResult<Vec<StsdExtension>> {
        trace(
            "extensions",
            repeat(
                0..,
                inclusive_length_and_then(be_u32, |input: &mut Stream<'_>| {
                    let typ = fourcc.parse_next(input)?;
                    parse_stsd_extension(typ).parse_next(input)
                }),
            ),
        )
        .parse_next(input)
    }
}
