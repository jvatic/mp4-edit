use bon::Builder;
use derive_more::Display;
use futures_io::AsyncRead;

pub use crate::atom::stsd::extension::{
    BtrtExtension, DecoderSpecificInfo, EsdsExtension, StsdExtension,
};
use crate::{
    atom::{
        stsd::serializer::text::serialize_text_entry,
        util::{read_to_end, serializer::fixed_point_16x16},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

mod extension;

pub const STSD: &[u8; 4] = b"stsd";

pub const SAMPLE_ENTRY_MP4A: &[u8; 4] = b"mp4a"; // AAC audio
pub const SAMPLE_ENTRY_AAVD: &[u8; 4] = b"aavd"; // Audible Audio
pub const SAMPLE_ENTRY_TEXT: &[u8; 4] = b"text"; // Plain text

#[derive(Debug, Clone, Display, PartialEq)]
#[display("{}", self.as_str())]
pub enum SampleEntryType {
    /// AAC audio
    Mp4a,
    /// Audible Audio (can be treated as Mp4a)
    Aavd,
    /// QuickTime Text Media
    Text,
    /// Unknown/unsupported sample entry type
    Unknown(FourCC),
}

impl SampleEntryType {
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        match bytes {
            SAMPLE_ENTRY_MP4A => SampleEntryType::Mp4a,
            SAMPLE_ENTRY_AAVD => SampleEntryType::Aavd,
            SAMPLE_ENTRY_TEXT => SampleEntryType::Text,
            _ => SampleEntryType::Unknown(FourCC(*bytes)),
        }
    }

    pub fn as_bytes(&self) -> &[u8; 4] {
        match self {
            SampleEntryType::Mp4a => SAMPLE_ENTRY_MP4A,
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

#[derive(Default, Debug, Clone)]
pub struct AudioSampleEntry {
    pub version: u16,
    pub channel_count: u16,
    pub sample_size: u16,
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

#[derive(Debug, Clone, Default, Builder)]
pub struct TextSampleEntry {
    #[builder(default)]
    pub display_flags: DisplayFlags,
    #[builder(default)]
    pub text_justification: TextJustification,
    #[builder(default)]
    pub background_color: ColorRgb,
    #[builder(default)]
    pub default_text_box: TextBox,
    #[builder(default)]
    pub font_number: u16,
    /// 0 = normal text
    #[builder(default)]
    pub font_face: FontFace,
    #[builder(default)]
    pub foreground_color: ColorRgb,
    #[builder(default)]
    pub font_name: String,
    #[builder(default)]
    pub extensions: Vec<StsdExtension>,
    #[builder(default)]
    pub trailing_data: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct DisplayFlags {
    /// Reflow the text instead of scaling when the track is scaled.
    pub disable_auto_scale: bool, // 0x0002
    /// Ignore the background color field in the text sample description and use the movie’s background color instead.
    pub use_movie_background_color: bool, // 0x0008
    /// Scroll the text until the last of the text is in view.
    pub scroll_in: bool, // 0x0020
    /// Scroll the text until the last of the text is gone.
    pub scroll_out: bool, // 0x0040
    /// Scroll the text horizontally when set; otherwise, scroll the text vertically.
    pub horizontal_scroll: bool, // 0x0080
    /// Scroll down (if scrolling vertically) or backward (if scrolling horizontally)
    ///
    /// **Note:** Horizontal scrolling also depends upon text justification.
    pub reverse_scroll: bool, // 0x0100
    /// Display new samples by scrolling out the old ones.
    pub continuous_scroll: bool, // 0x0200
    /// Display the text with a drop shadow.
    pub drop_shadow: bool, // 0x1000
    /// Use anti-aliasing when drawing text.
    pub anti_alias: bool, // 0x2000
    /// Do not display the background color, so that the text overlay background tracks.
    pub key_text: bool, // 0x4000
}

#[derive(Debug, Clone, Default)]
pub enum TextJustification {
    #[default]
    Left,
    Centre,
    Right,
    Other(i32),
}

#[derive(Debug, Clone, Default)]
pub struct ColorRgb {
    pub red: u16,
    pub green: u16,
    pub blue: u16,
}

#[derive(Debug, Clone, Default)]
pub struct TextBox {
    pub top: u16,
    pub left: u16,
    pub bottom: u16,
    pub right: u16,
}

#[derive(Debug, Clone, Default)]
pub struct FontFace {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub outline: bool,
    pub shadow: bool,
    pub condense: bool,
    pub extend: bool,
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
                SampleEntryData::Audio(audio) => {
                    entry_data.extend(audio.version.to_be_bytes());
                    entry_data.extend([0u8; 6]); // reserved
                    entry_data.extend(audio.channel_count.to_be_bytes());
                    entry_data.extend(audio.sample_size.to_be_bytes());
                    entry_data.extend(2u16.to_be_bytes()); // pre-defined
                    entry_data.extend([0u8; 2]); // reserved
                    entry_data.extend(fixed_point_16x16(audio.sample_rate));
                    audio.extensions.into_iter().for_each(|ext| {
                        let ext_data: Vec<u8> = ext.to_bytes();
                        entry_data.extend_from_slice(&ext_data);
                    });
                }
                SampleEntryData::Text(text) => {
                    entry_data.extend(serialize_text_entry(text));
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

mod serializer {
    pub(crate) mod text {
        use crate::atom::{
            stsd::{ColorRgb, DisplayFlags, FontFace, TextBox, TextJustification, TextSampleEntry},
            util::serializer::{bits::Packer, pascal_string},
        };

        pub fn serialize_text_entry(text: TextSampleEntry) -> Vec<u8> {
            let mut data = Vec::new();

            data.extend(display_flags(text.display_flags));
            data.extend(text_justification(text.text_justification));
            data.extend(color_rgb(text.background_color));
            data.extend(text_box(text.default_text_box));
            data.extend(text.font_number.to_be_bytes());
            data.extend(font_face(text.font_face));
            data.extend(color_rgb(text.foreground_color));
            data.extend(pascal_string(text.font_name));

            text.extensions.into_iter().for_each(|ext| {
                let ext_data: Vec<u8> = ext.to_bytes();
                data.extend_from_slice(&ext_data);
            });

            data.extend(text.trailing_data);

            data
        }

        fn display_flags(d: DisplayFlags) -> [u8; 4] {
            let mut packer = Packer::from(vec![0u8; 2]); // 2 leading empty bytes
            packer.push_n::<1>(0); // 1 leading empty bit
            packer.push_bool(d.key_text);
            packer.push_bool(d.anti_alias);
            packer.push_bool(d.drop_shadow);
            packer.push_n::<2>(0); // 2 padding bits
            packer.push_bool(d.continuous_scroll);
            packer.push_bool(d.reverse_scroll);
            packer.push_bool(d.horizontal_scroll);
            packer.push_bool(d.scroll_out);
            packer.push_bool(d.scroll_in);
            packer.push_n::<1>(0); // 1 padding bit
            packer.push_bool(d.use_movie_background_color);
            packer.push_n::<1>(0); // 1 padding bit
            packer.push_bool(d.disable_auto_scale);
            packer.push_n::<1>(0); // 1 padding bit
            Vec::from(packer)
                .try_into()
                .expect("display_flags is 4 bytes")
        }

        fn text_justification(j: TextJustification) -> [u8; 4] {
            let value: i32 = match j {
                TextJustification::Left => 0,
                TextJustification::Centre => 1,
                TextJustification::Right => -1,
                TextJustification::Other(v) => v,
            };
            value.to_be_bytes()
        }

        fn color_rgb(color: ColorRgb) -> [u8; 6] {
            let mut data = Vec::with_capacity(6);
            data.extend(color.red.to_be_bytes());
            data.extend(color.green.to_be_bytes());
            data.extend(color.blue.to_be_bytes());
            data.try_into().expect("color_rgb is 6 bytes")
        }

        fn text_box(b: TextBox) -> [u8; 8] {
            let mut data = Vec::with_capacity(6);
            data.extend(b.top.to_be_bytes());
            data.extend(b.left.to_be_bytes());
            data.extend(b.bottom.to_be_bytes());
            data.extend(b.right.to_be_bytes());
            data.try_into().expect("text_box is 8 bytes")
        }

        fn font_face(f: FontFace) -> [u8; 2] {
            let mut packer = Packer::from(vec![0u8; 1]); // 1 leading byte
            packer.push_n::<1>(0); // 1 leading bit
            packer.push_bool(f.extend);
            packer.push_bool(f.condense);
            packer.push_bool(f.shadow);
            packer.push_bool(f.outline);
            packer.push_bool(f.underline);
            packer.push_bool(f.italic);
            packer.push_bool(f.bold);
            Vec::from(packer).try_into().expect("font_face is 2 bytes")
        }
    }
}

mod parser {
    use winnow::{
        binary::{be_i32, be_u16, be_u32, bits, length_repeat},
        combinator::seq,
        error::{ContextError, ErrMode, StrContext},
        ModalResult, Parser,
    };

    use crate::atom::{
        stsd::extension::parser::parse_stsd_extensions,
        util::parser::{
            byte_array, combinators::inclusive_length_and_then, fixed_point_16x16, flags3,
            pascal_string, rest_vec, stream, version, Stream,
        },
    };

    use super::*;

    pub fn parse_stsd_data(input: &[u8]) -> Result<SampleDescriptionTableAtom, crate::ParseError> {
        parse_stsd_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    pub fn parse_stsd_data_inner(
        input: &mut Stream<'_>,
    ) -> ModalResult<SampleDescriptionTableAtom> {
        seq!(SampleDescriptionTableAtom {
            version: version.verify(|v| *v == 0),
            flags: flags3,
            entries: length_repeat(be_u32, parse_sample_entry)
                .context(StrContext::Label("entries")),
        })
        .parse_next(input)
    }

    fn parse_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntry> {
        inclusive_length_and_then(
            be_u32,
            seq!(SampleEntry {
                entry_type: byte_array::<4>
                    .map(|v| SampleEntryType::from_bytes(&v))
                    .context(StrContext::Label("entry_type")),
                _: byte_array::<6>.context(StrContext::Label("reserved")), // reserved
                data_reference_index: be_u16.context(StrContext::Label("data_reference_index")),
                data: match entry_type {
                    SampleEntryType::Mp4a | SampleEntryType::Aavd => {
                        parse_audio_sample_entry
                    }
                    SampleEntryType::Text => {
                        parse_text_sample_entry
                    }
                    _ => parse_unknown_sample_entry,
                }.context(StrContext::Label("data")),
            }),
        )
        .parse_next(input)
    }

    pub fn parse_audio_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        seq!(AudioSampleEntry {
            version: be_u16.verify(|v| *v == 0).context(StrContext::Label("version")),
            _: byte_array::<6>.context(StrContext::Label("reserved")),
            channel_count: be_u16.context(StrContext::Label("channel_count")),
            sample_size: be_u16.context(StrContext::Label("sample_size")),
            _: byte_array::<2>.context(StrContext::Label("pre-defined")),
            _: byte_array::<2>.context(StrContext::Label("reserved")),
            sample_rate: fixed_point_16x16.context(StrContext::Label("sample_rate")),
            extensions: parse_stsd_extensions.context(StrContext::Label("extensions")),
        })
        .map(SampleEntryData::Audio)
        .parse_next(input)
    }

    pub fn parse_text_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        fn display_flags(input: &mut (Stream, usize)) -> ModalResult<DisplayFlags> {
            use bits::bool;
            seq!(DisplayFlags {
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(8usize).context(StrContext::Label("leading byte 1")),
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(8usize).context(StrContext::Label("leading byte 2")),
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(1usize).context(StrContext::Label("leading bit")),
                key_text: bool.context(StrContext::Label("key_text")),
                anti_alias: bool.context(StrContext::Label("anti_alias")),
                drop_shadow: bool.context(StrContext::Label("drop_shadow")),
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(2usize).context(StrContext::Label("padding")),
                continuous_scroll: bool.context(StrContext::Label("continuous_scroll")),
                reverse_scroll: bool.context(StrContext::Label("reverse_scroll")),
                horizontal_scroll: bool.context(StrContext::Label("horizontal_scroll")),
                scroll_out: bool.context(StrContext::Label("scroll_out")),
                scroll_in: bool.context(StrContext::Label("scroll_in")),
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(1usize).context(StrContext::Label("padding")),
                use_movie_background_color: bool
                    .context(StrContext::Label("use_movie_background_color")),
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(1usize).context(StrContext::Label("padding")),
                disable_auto_scale: bool.context(StrContext::Label("disable_auto_scale")),
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(1usize).context(StrContext::Label("padding")),
            })
            .parse_next(input)
        }

        fn text_justification(input: &mut Stream<'_>) -> ModalResult<TextJustification> {
            let text_justification = be_i32.parse_next(input)?;
            Ok(match text_justification {
                0 => TextJustification::Left,
                1 => TextJustification::Centre,
                -1 => TextJustification::Right,
                v => TextJustification::Other(v),
            })
        }

        fn color_rgb(input: &mut Stream<'_>) -> ModalResult<ColorRgb> {
            seq!(ColorRgb {
                red: be_u16.context(StrContext::Label("red")),
                green: be_u16.context(StrContext::Label("green")),
                blue: be_u16.context(StrContext::Label("blue")),
            })
            .parse_next(input)
        }

        fn text_box(input: &mut Stream<'_>) -> ModalResult<TextBox> {
            seq!(TextBox {
                top: be_u16.context(StrContext::Label("top")),
                left: be_u16.context(StrContext::Label("left")),
                bottom: be_u16.context(StrContext::Label("bottom")),
                right: be_u16.context(StrContext::Label("right")),
            })
            .parse_next(input)
        }

        fn font_face(input: &mut (Stream, usize)) -> ModalResult<FontFace> {
            use bits::bool;
            seq!(FontFace {
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(8usize).context(StrContext::Label("leading empty byte")),
                _: bits::take::<_, usize, _, ErrMode<ContextError>>(1usize).context(StrContext::Label("leading empty bit")),
                extend: bool.context(StrContext::Label("extend")),
                condense: bool.context(StrContext::Label("condense")),
                shadow: bool.context(StrContext::Label("shadow")),
                outline: bool.context(StrContext::Label("outline")),
                underline: bool.context(StrContext::Label("underline")),
                italic: bool.context(StrContext::Label("italic")),
                bold: bool.context(StrContext::Label("bold")),
            })
            .parse_next(input)
        }

        seq!(TextSampleEntry {
            display_flags: bits::bits(display_flags).context(StrContext::Label("display_flags")),
            text_justification: text_justification.context(StrContext::Label("text_justification")),
            background_color: color_rgb.context(StrContext::Label("background_color")),
            default_text_box: text_box.context(StrContext::Label("default_text_box")),
            font_number: be_u16.context(StrContext::Label("font_number")),
            font_face: bits::bits(font_face).context(StrContext::Label("font_face")),
            foreground_color: color_rgb.context(StrContext::Label("foreground_color")),
            font_name: pascal_string.context(StrContext::Label("text_name")),
            extensions: parse_stsd_extensions.context(StrContext::Label("extensions")),
            trailing_data: rest_vec.context(StrContext::Label("trailing_data")),
        })
        .map(SampleEntryData::Text)
        .parse_next(input)
    }

    pub fn parse_unknown_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        rest_vec.map(SampleEntryData::Other).parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use winnow::Parser;

    use crate::atom::stsd::extension::parser::parse_stsd_extensions;
    use crate::atom::test_utils::test_atom_roundtrip_sync;
    use crate::atom::util::parser::stream;

    use super::*;

    #[test]
    fn test_parse_extensions_round_trip() {
        let extension_data: Vec<u8> = vec![
            0, 0, 0, 51, 101, 115, 100, 115, 0, 0, 0, 0, 3, 128, 128, 128, 34, 0, 1, 0, 4, 128,
            128, 128, 20, 64, 21, 0, 0, 0, 0, 0, 245, 74, 0, 0, 245, 74, 5, 128, 128, 128, 2, 19,
            144, 6, 128, 128, 128, 1, 2, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 245, 74,
            0, 0, 245, 74,
        ];
        let result = parse_stsd_extensions.parse(stream(&extension_data));
        assert!(result.is_ok(), "extensions should parse");
        let result = result.unwrap();

        let round_trip_data: Vec<u8> = result
            .clone()
            .into_iter()
            .flat_map(|ext| ext.to_bytes())
            .collect();

        assert_eq!(
            round_trip_data, extension_data,
            "expected round trip data to equal input data ({result:?})"
        );
    }

    /// Test round-trip for all available stsd test data files
    #[test]
    fn test_stsd_roundtrip() {
        test_atom_roundtrip_sync::<SampleDescriptionTableAtom>(STSD);
    }

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
}
