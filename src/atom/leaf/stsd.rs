use anyhow::{anyhow, Context};
use derive_more::Display;
use futures_io::AsyncRead;
use std::{
    io::{Cursor, Read},
    marker::PhantomData,
};

pub use crate::atom::stsd::extension::{
    BtrtExtension, DecoderSpecificInfo, EsdsExtension, StsdExtension,
};
use crate::{
    atom::{
        util::{
            read_to_end,
            serializer::{SizeU32, SizeU8},
        },
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

mod extension;

pub const STSD: &[u8; 4] = b"stsd";

// Common sample entry types
pub const SAMPLE_ENTRY_AVC1: &[u8; 4] = b"avc1"; // H.264 video
pub const SAMPLE_ENTRY_HVC1: &[u8; 4] = b"hvc1"; // H.265/HEVC video
pub const SAMPLE_ENTRY_MP4A: &[u8; 4] = b"mp4a"; // AAC audio
pub const SAMPLE_ENTRY_MP4V: &[u8; 4] = b"mp4v"; // MPEG-4 video
pub const SAMPLE_ENTRY_TX3G: &[u8; 4] = b"tx3g"; // 3GPP text
pub const SAMPLE_ENTRY_WVTT: &[u8; 4] = b"wvtt"; // WebVTT text
pub const SAMPLE_ENTRY_STPP: &[u8; 4] = b"stpp"; // Subtitle
pub const SAMPLE_ENTRY_AAVD: &[u8; 4] = b"aavd"; // Apple Audio Video Data
pub const SAMPLE_ENTRY_TEXT: &[u8; 4] = b"text"; // Plain text

#[derive(Debug, Clone, Display, PartialEq)]
#[display("{}", self.as_str())]
pub enum SampleEntryType {
    /// H.264/AVC video
    Avc1,
    /// H.265/HEVC video
    Hvc1,
    /// AAC audio
    Mp4a,
    /// MPEG-4 video
    Mp4v,
    /// 3GPP text
    Tx3g,
    /// WebVTT text
    Wvtt,
    /// Subtitle
    Stpp,
    /// Apple Audio Video Data
    Aavd,
    /// Plain text
    Text,
    /// Unknown sample entry type
    Unknown(FourCC),
}

impl SampleEntryType {
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        match bytes {
            SAMPLE_ENTRY_AVC1 => SampleEntryType::Avc1,
            SAMPLE_ENTRY_HVC1 => SampleEntryType::Hvc1,
            SAMPLE_ENTRY_MP4A => SampleEntryType::Mp4a,
            SAMPLE_ENTRY_MP4V => SampleEntryType::Mp4v,
            SAMPLE_ENTRY_TX3G => SampleEntryType::Tx3g,
            SAMPLE_ENTRY_WVTT => SampleEntryType::Wvtt,
            SAMPLE_ENTRY_STPP => SampleEntryType::Stpp,
            SAMPLE_ENTRY_AAVD => SampleEntryType::Aavd,
            SAMPLE_ENTRY_TEXT => SampleEntryType::Text,
            _ => SampleEntryType::Unknown(FourCC(*bytes)),
        }
    }

    pub fn as_bytes(&self) -> &[u8; 4] {
        match self {
            SampleEntryType::Avc1 => SAMPLE_ENTRY_AVC1,
            SampleEntryType::Hvc1 => SAMPLE_ENTRY_HVC1,
            SampleEntryType::Mp4a => SAMPLE_ENTRY_MP4A,
            SampleEntryType::Mp4v => SAMPLE_ENTRY_MP4V,
            SampleEntryType::Tx3g => SAMPLE_ENTRY_TX3G,
            SampleEntryType::Wvtt => SAMPLE_ENTRY_WVTT,
            SampleEntryType::Stpp => SAMPLE_ENTRY_STPP,
            SampleEntryType::Aavd => SAMPLE_ENTRY_AAVD,
            SampleEntryType::Text => SAMPLE_ENTRY_TEXT,
            SampleEntryType::Unknown(bytes) => &bytes.0,
        }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(self.as_bytes()).unwrap_or("????")
    }
}

#[derive(Debug, Clone)]
pub enum SampleEntryData {
    Video(VideoSampleEntry),
    Audio(AudioSampleEntry),
    Text(TextSampleEntry),
    Other(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct SampleEntry {
    /// Sample entry type (4CC code)
    pub entry_type: SampleEntryType,
    /// Data reference index
    pub data_reference_index: u16,
    /// Raw sample entry data (codec-specific)
    pub data: SampleEntryData,
}

#[derive(Debug, Clone)]
pub struct VideoSampleEntry {
    /// Version (usually 0)
    pub version: u16,
    /// Revision level (usually 0)
    pub revision_level: u16,
    /// Vendor identifier
    pub vendor: [u8; 4],
    /// Temporal quality (0-1023)
    pub temporal_quality: u32,
    /// Spatial quality (0-1024)
    pub spatial_quality: u32,
    /// Width in pixels
    pub width: u16,
    /// Height in pixels
    pub height: u16,
    /// Horizontal resolution (pixels per inch, 16.16 fixed point)
    pub horizresolution: f32,
    /// Vertical resolution (pixels per inch, 16.16 fixed point)
    pub vertresolution: f32,
    /// Reserved (should be 0)
    pub entry_data_size: u32,
    /// Frame count (usually 1)
    pub frame_count: u16,
    /// Compressor name (32 bytes, Pascal string)
    pub compressor_name: String,
    /// Bit depth (usually 24)
    pub depth: u16,
    /// Color table ID (usually -1)
    pub color_table_id: i16,
    /// Extension data (codec-specific atoms)
    pub extensions: Vec<StsdExtension>,
}

#[derive(Default, Debug, Clone)]
pub struct AudioSampleEntry {
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

#[derive(Debug, Clone)]
pub struct TextSampleEntry {
    /// Version (usually 0)
    pub version: u16,
    /// Revision level (usually 0)
    pub revision_level: u16,
    /// Vendor identifier
    pub vendor: [u8; 4],
    /// Display flags
    pub display_flags: u32,
    /// Text justification (0 = left, 1 = center, -1 = right)
    pub text_justification: i8,
    /// Background color (RGB)
    pub background_color: [u16; 3],
    /// Default text box (top, left, bottom, right)
    pub default_text_box: [u16; 4],
    pub unknown: Option<[u8; 3]>,
    /// Extension data (codec-specific atoms)
    pub extensions: Vec<StsdExtension>,
    pub extensions_size: ExtensionSizeType,
}

#[derive(Debug, Clone)]
pub enum ExtensionSizeType {
    U8,
    U32,
}

#[derive(Default, Debug, Clone)]
pub struct SampleDescriptionTableAtom {
    /// Version of the stsd atom format (0)
    pub version: u8,
    /// Flags for the stsd atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of sample entries
    pub entries: Vec<SampleEntry>,
}

impl From<Vec<SampleEntry>> for SampleDescriptionTableAtom {
    fn from(entries: Vec<SampleEntry>) -> Self {
        SampleDescriptionTableAtom {
            version: 0,
            flags: [0u8; 3],
            entries,
        }
    }
}

impl SampleDescriptionTableAtom {
    pub fn find_or_create_entry<P, D>(&mut self, pred: P, default_fn: D) -> &mut SampleEntry
    where
        P: Fn(&SampleEntry) -> bool,
        D: FnOnce() -> SampleEntry,
    {
        if let Some(index) = self.entries.iter().position(pred) {
            return &mut self.entries[index];
        }
        self.entries.push(default_fn());
        self.entries.last_mut().unwrap()
    }
}

impl ParseAtom for SampleDescriptionTableAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STSD {
            return Err(ParseError::new_unexpected_atom(atom_type, STSD));
        }
        let data = read_to_end(reader).await?;
        parser::parse_stsd_data(&data)
    }
}

impl SerializeAtom for SampleDescriptionTableAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STSD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Entry count (4 bytes, big-endian)
        data.extend_from_slice(&(self.entries.len() as u32).to_be_bytes());

        // Sample entries
        for entry in self.entries {
            let mut entry_data = Vec::new();

            // Reserved (6 bytes) - must be zero
            entry_data.extend_from_slice(&[0u8; 6]);

            // Data reference index (2 bytes, big-endian)
            entry_data.extend_from_slice(&entry.data_reference_index.to_be_bytes());

            // Entry-specific data based on type
            match entry.data {
                SampleEntryData::Video(video) => {
                    // Video sample entry structure
                    entry_data.extend_from_slice(&video.version.to_be_bytes());
                    entry_data.extend_from_slice(&video.revision_level.to_be_bytes());
                    entry_data.extend_from_slice(&video.vendor);
                    entry_data.extend_from_slice(&video.temporal_quality.to_be_bytes());
                    entry_data.extend_from_slice(&video.spatial_quality.to_be_bytes());
                    entry_data.extend_from_slice(&video.width.to_be_bytes());
                    entry_data.extend_from_slice(&video.height.to_be_bytes());
                    entry_data.extend_from_slice(
                        &((video.horizresolution * 65536.0) as u32).to_be_bytes(),
                    );
                    entry_data.extend_from_slice(
                        &((video.vertresolution * 65536.0) as u32).to_be_bytes(),
                    );
                    entry_data.extend_from_slice(&video.entry_data_size.to_be_bytes());
                    entry_data.extend_from_slice(&video.frame_count.to_be_bytes());

                    // Compressor name (32 bytes, Pascal string)
                    let mut compressor_bytes = [0u8; 32];
                    let name_bytes = video.compressor_name.as_bytes();
                    let len = name_bytes.len().min(31);
                    compressor_bytes[0] = len as u8;
                    compressor_bytes[1..=len].copy_from_slice(&name_bytes[..len]);
                    entry_data.extend_from_slice(&compressor_bytes);

                    entry_data.extend_from_slice(&video.depth.to_be_bytes());
                    entry_data.extend_from_slice(&video.color_table_id.to_be_bytes());
                    video.extensions.into_iter().for_each(|ext| {
                        let ext_data: Vec<u8> = ext.to_bytes::<SizeU32>();
                        entry_data.extend_from_slice(&ext_data);
                    });
                }
                SampleEntryData::Audio(audio) => {
                    // Audio sample entry structure
                    entry_data.extend_from_slice(&audio.version.to_be_bytes());
                    entry_data.extend_from_slice(&audio.revision_level.to_be_bytes());
                    entry_data.extend_from_slice(&audio.vendor);
                    entry_data.extend_from_slice(&audio.channel_count.to_be_bytes());
                    entry_data.extend_from_slice(&audio.sample_size.to_be_bytes());
                    entry_data.extend_from_slice(&audio.compression_id.to_be_bytes());
                    entry_data.extend_from_slice(&audio.packet_size.to_be_bytes());
                    entry_data
                        .extend_from_slice(&((audio.sample_rate * 65536.0) as u32).to_be_bytes());
                    audio.extensions.into_iter().for_each(|ext| {
                        let ext_data = ext.to_bytes::<SizeU32>();
                        entry_data.extend_from_slice(&ext_data);
                    });
                }
                SampleEntryData::Text(text) => {
                    // Text sample entry structure
                    entry_data.extend_from_slice(&text.version.to_be_bytes());
                    entry_data.extend_from_slice(&text.revision_level.to_be_bytes());
                    entry_data.extend_from_slice(&text.vendor);
                    entry_data.extend_from_slice(&text.display_flags.to_be_bytes());
                    entry_data.push(text.text_justification as u8);

                    // Background color (3 * 2 bytes)
                    for &color in &text.background_color {
                        entry_data.extend_from_slice(&color.to_be_bytes());
                    }

                    // Default text box (4 * 2 bytes)
                    for &coord in &text.default_text_box {
                        entry_data.extend_from_slice(&coord.to_be_bytes());
                    }

                    if let Some(unknown) = text.unknown {
                        entry_data.extend(unknown);
                    }

                    // Add extensions
                    let ext_size = text.extensions_size;
                    text.extensions.into_iter().for_each(|ext| {
                        let ext_data = match ext_size {
                            ExtensionSizeType::U8 => ext.to_bytes::<SizeU8>(),
                            ExtensionSizeType::U32 => ext.to_bytes::<SizeU32>(),
                        };
                        entry_data.extend_from_slice(&ext_data);
                    });
                }
                SampleEntryData::Other(other_data) => {
                    // Other sample entry data
                    entry_data.extend_from_slice(&other_data);
                }
            }

            // Calculate total entry size (4 + 4 + entry_data.len())
            let entry_size = 8 + entry_data.len();

            // Write entry size (4 bytes, big-endian)
            data.extend_from_slice(&(entry_size as u32).to_be_bytes());

            // Write entry type (4 bytes)
            data.extend_from_slice(entry.entry_type.as_bytes());

            // Write entry data
            data.extend_from_slice(&entry_data);
        }

        data
    }
}

mod parser {
    use std::marker::PhantomData;

    use winnow::{
        binary::{be_i16, be_u16, be_u32, i8, length_and_then, u8},
        combinator::{alt, dispatch, empty, fail, opt, repeat, seq, todo, trace},
        error::{ContextError, ErrMode, Needed, ParserError, StrContext, StrContextValue},
        stream::ToUsize,
        token::rest,
        ModalResult, Parser,
    };

    use super::{
        AudioSampleEntry, SampleDescriptionTableAtom, SampleEntry, SampleEntryData,
        SampleEntryType, TextSampleEntry, VideoSampleEntry,
    };
    use crate::atom::{
        stsd::{
            extension::{
                parser::{parse_btrt_extension, parse_esds_extension, parse_unknown_extension},
                BTRT, ESDS,
            },
            ExtensionSizeType, StsdExtension,
        },
        util::{
            parser::{
                byte_array,
                combinators::{count_then_repeat, inclusive_length_and_then},
                fixed_array, fixed_point_16x16, flags3, fourcc, stream, version, Stream,
            },
            serializer::{SerializeSize, SizeU32},
        },
    };

    pub fn parse_stsd_data(input: &[u8]) -> Result<SampleDescriptionTableAtom, crate::ParseError> {
        parse_stsd_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_stsd_data_inner(input: &mut Stream<'_>) -> ModalResult<SampleDescriptionTableAtom> {
        trace(
            "stsd",
            seq!(SampleDescriptionTableAtom {
                version: version,
                flags: flags3,
                entries: count_then_repeat(be_u32, inclusive_length_and_then(be_u32, entry))
                    .context(StrContext::Label("entries")),
            })
            .context(StrContext::Label("stsd")),
        )
        .parse_next(input)
    }

    fn entry(input: &mut Stream<'_>) -> ModalResult<SampleEntry> {
        trace(
            "entry",
            seq!(SampleEntry {
                entry_type: byte_array
                    .map(|buf: [u8; 4]| SampleEntryType::from_bytes(&buf))
                    .context(StrContext::Label("entry_type")),
                _: reserved,
                data_reference_index: be_u16.context(StrContext::Label("data_reference_index")),
                data: match entry_type {
                    SampleEntryType::Avc1 | SampleEntryType::Hvc1 | SampleEntryType::Mp4v => {
                        video_sample_entry
                    }
                    SampleEntryType::Mp4a | SampleEntryType::Aavd => {
                        audio_sample_entry
                    }
                    SampleEntryType::Tx3g
                    | SampleEntryType::Wvtt
                    | SampleEntryType::Stpp
                    | SampleEntryType::Text => text_sample_entry,
                    _ => other_sample_entry,
                },
            }),
        )
        .parse_next(input)
    }

    fn reserved(input: &mut Stream<'_>) -> ModalResult<[u8; 6]> {
        trace("reserved", byte_array).parse_next(input)
    }

    fn video_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        trace(
            "video_sample_entry",
            seq!(VideoSampleEntry {
                version: be_u16.context(StrContext::Label("version")),
                revision_level: be_u16.context(StrContext::Label("revision_level")),
                // TODO: is this a fourcc?
                vendor: byte_array.context(StrContext::Label("vendor")),
                temporal_quality: be_u32.context(StrContext::Label("temporal_quality")),
                spatial_quality: be_u32.context(StrContext::Label("spatial_quality")),
                width: be_u16.context(StrContext::Label("width")),
                height: be_u16.context(StrContext::Label("height")),
                horizresolution: fixed_point_16x16.context(StrContext::Label("horizresolution")),
                vertresolution: fixed_point_16x16.context(StrContext::Label("vertresolution")),
                entry_data_size: be_u32.context(StrContext::Label("entry_data_size")),
                frame_count: be_u16.context(StrContext::Label("frame_count")),
                compressor_name: pascal_string.context(StrContext::Label("compressor_name")),
                depth: be_u16.context(StrContext::Label("depth")),
                color_table_id: be_i16.context(StrContext::Label("color_table_id")),
                extensions: extensions(be_u32).context(StrContext::Label("extensions")),
            })
            .map(SampleEntryData::Video),
        )
        .context(StrContext::Label("video"))
        .parse_next(input)
    }

    fn audio_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        trace(
            "audio_sample_entry",
            seq!(AudioSampleEntry {
                version: be_u16.context(StrContext::Label("version")),
                revision_level: be_u16.context(StrContext::Label("revision_level")),
                // TODO: is this a fourcc?
                vendor: byte_array.context(StrContext::Label("vendor")),
                channel_count: be_u16.context(StrContext::Label("channel_count")),
                sample_size: be_u16.context(StrContext::Label("sample_size")),
                compression_id: be_u16.context(StrContext::Label("compression_id")),
                packet_size: be_u16.context(StrContext::Label("packet_size")),
                sample_rate: fixed_point_16x16.context(StrContext::Label("sample_rate")),
                extensions: extensions(be_u32).context(StrContext::Label("extensions")),
            })
            .map(SampleEntryData::Audio),
        )
        .context(StrContext::Label("audio"))
        .parse_next(input)
    }

    fn text_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        #[derive(Clone, Copy)]
        struct TextSampleEntryBase {
            pub version: u16,
            pub revision_level: u16,
            pub vendor: [u8; 4],
            pub display_flags: u32,
            pub text_justification: i8,
            pub background_color: [u16; 3],
            pub default_text_box: [u16; 4],
        }

        trace("text_sample_entry", move |input: &mut Stream<'_>| {
            let base = seq!(TextSampleEntryBase {
                version: be_u16.context(StrContext::Label("version")),
                revision_level: be_u16.context(StrContext::Label("revision_level")),
                // TODO: is this a fourcc?
                vendor: byte_array.context(StrContext::Label("vendor")),
                display_flags: be_u32.context(StrContext::Label("display_flags")),
                text_justification: i8.context(StrContext::Label("text_justification")),
                background_color: fixed_array(be_u16)
                    .context(StrContext::Label("background_color")),
                default_text_box: fixed_array(be_u16)
                    .context(StrContext::Label("default_text_box")),
            })
            .parse_next(input)?;

            fn finally(
                base: TextSampleEntryBase,
                unknown: Option<[u8; 3]>,
                extensions: Vec<StsdExtension>,
                extensions_size: ExtensionSizeType,
            ) -> TextSampleEntry {
                let TextSampleEntryBase {
                    version,
                    revision_level,
                    vendor,
                    display_flags,
                    text_justification,
                    background_color,
                    default_text_box,
                } = base;

                TextSampleEntry {
                    version,
                    revision_level,
                    vendor,
                    display_flags,
                    text_justification,
                    background_color,
                    default_text_box,
                    unknown,
                    extensions,
                    extensions_size,
                }
            }

            alt((
                (empty.value(None), extensions(u8))
                    .map(move |(unknown, extensions)| {
                        SampleEntryData::Text(finally(
                            base,
                            unknown,
                            extensions,
                            ExtensionSizeType::U8,
                        ))
                    })
                    .context(StrContext::Label("u8 sized extensions")),
                (empty.value(None), extensions(be_u32))
                    .map(move |(unknown, extensions)| {
                        SampleEntryData::Text(finally(
                            base,
                            unknown,
                            extensions,
                            ExtensionSizeType::U32,
                        ))
                    })
                    .context(StrContext::Label("be_u32 sized extensions")),
                (byte_array::<3>.map(|v| Some(v)), extensions(be_u32))
                    .map(move |(unknown, extensions)| {
                        SampleEntryData::Text(finally(
                            base,
                            unknown,
                            extensions,
                            ExtensionSizeType::U32,
                        ))
                    })
                    .context(StrContext::Label(
                        "3 bytes and then be_u32 sized extensions",
                    )),
                fail.context(StrContext::Expected(StrContextValue::Description(
                    "u8 sized extensions",
                )))
                .context(StrContext::Expected(StrContextValue::Description(
                    "be_u32 sized extensions",
                )))
                .context(StrContext::Expected(StrContextValue::Description(
                    "3 bytes and then be_u32 sized extensions",
                ))),
            ))
            .parse_next(input)
        })
        .context(StrContext::Label("text"))
        .parse_next(input)
    }

    fn other_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        trace(
            "other_sample_entry",
            rest.map(|buf: &[u8]| buf.to_vec())
                .map(SampleEntryData::Other),
        )
        .context(StrContext::Label("unknown"))
        .parse_next(input)
    }

    fn extensions<'i, ParseSize, UsizeLike>(
        mut size_parser: ParseSize,
    ) -> impl Parser<Stream<'i>, Vec<StsdExtension>, ErrMode<ContextError>>
    where
        UsizeLike: ToUsize,
        ParseSize: Parser<Stream<'i>, UsizeLike, ErrMode<ContextError>>,
    {
        trace("extensions", move |input: &mut Stream<'i>| {
            repeat(
                1..,
                inclusive_length_and_then(
                    size_parser.by_ref(),
                    dispatch! {fourcc;
                        typ if typ == ESDS => parse_esds_extension,
                        typ if typ == BTRT => parse_btrt_extension,
                        typ => parse_unknown_extension(typ),
                    },
                ),
            )
            .parse_next(input)
        })
    }

    fn pascal_string(input: &mut Stream<'_>) -> ModalResult<String> {
        trace(
            "pascal_string",
            length_and_then(
                u8,
                rest.try_map(|buf: &[u8]| String::from_utf8(buf.to_vec()))
                    .context(StrContext::Expected(StrContextValue::Description(
                        "UTF8 string",
                    ))),
            ),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    use super::*;

    #[test]
    fn test_text_sample_entry_type_recognition() {
        // Test that "text" sample entry type is recognized correctly
        let text_bytes = b"text";
        let entry_type = SampleEntryType::from_bytes(text_bytes);
        assert_eq!(entry_type, SampleEntryType::Text);

        // Test round-trip
        let bytes_back = entry_type.as_bytes();
        assert_eq!(bytes_back, text_bytes);

        // Test string representation
        assert_eq!(entry_type.as_str(), "text");
    }

    /// Test round-trip for all available stsd test data files
    #[test]
    fn test_stsd_roundtrip() {
        test_atom_roundtrip_sync::<SampleDescriptionTableAtom>(STSD);
    }
}
