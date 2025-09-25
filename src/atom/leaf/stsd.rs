use futures_io::AsyncRead;

pub use crate::atom::stsd::extension::{
    btrt::BtrtExtension,
    esds::{DecoderSpecificInfo, EsdsExtension},
    StsdExtension,
};
use crate::{
    atom::{util::read_to_end, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub mod extension;

pub const STSD: &[u8; 4] = b"stsd";

#[derive(Debug, Clone)]
pub enum SampleEntryData {
    Mp4a(Mp4aEntryData),
    Tx3g(Tx3gEntryData),
    Unknown(FourCC, Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct SampleEntry {
    /// Data reference index
    pub data_reference_index: u16,
    /// Raw sample entry data (codec-specific)
    pub data: SampleEntryData,
}

impl SampleEntry {
    fn entry_type(&self) -> &FourCC {
        match &self.data {
            SampleEntryData::Mp4a(_) => &Mp4aEntryData::TYPE,
            SampleEntryData::Tx3g(_) => &Tx3gEntryData::TYPE,
            SampleEntryData::Unknown(typ, _) => typ,
        }
    }

    pub fn is_audio(&self) -> bool {
        match &self.data {
            SampleEntryData::Mp4a(_) => true,
            SampleEntryData::Unknown(typ, _)
                if match &typ.0 {
                    b"aavd" => true,
                    _ => false,
                } =>
            {
                true
            }
            _ => false,
        }
    }
}

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
    const TYPE: FourCC = FourCC::new(b"mp4a");

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

#[derive(Default, Debug, Clone)]
pub struct Tx3gEntryData {
    pub display_flags: u32,
    pub horizontal_justification: i8,
    pub vertical_justification: i8,
    pub background_color: ColorRgba,
    pub default_text_box: TextBox,
    pub default_style_record: StyleRecord,
    pub font_table: Option<Vec<FontTableEntry>>,
}

#[derive(Default, Debug, Clone)]
pub struct ColorRgba {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
    pub alpha: u8,
}

#[derive(Default, Debug, Clone)]
pub struct TextBox {
    pub top: i16,
    pub left: i16,
    pub bottom: i16,
    pub right: i16,
}

#[derive(Default, Debug, Clone)]
pub struct StyleRecord {
    pub start_char: u16,
    pub end_char: u16,
    pub font_id: u16,
    pub font_style_flags: FontStyle,
    pub font_size: u8,
    pub text_color: ColorRgba,
}

#[derive(Default, Debug, Clone)]
pub struct FontStyle {
    pub bold: bool,   // 1 bit
    pub italic: bool, // 1 bit
    pub underline: bool, // 1 bit
                      // 5 reserved bits
}

#[derive(Default, Debug, Clone)]
pub struct FontTableEntry {
    pub font_id: u16,
    pub font_name: String,
}

impl Tx3gEntryData {
    const TYPE: FourCC = FourCC::new(b"tx3g");
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

    /// Finds an audio sample entry matching `pred` or inserts a new one with `default_fn`
    ///
    /// NOTE: audio sample entries that don't match `pred` will be removed.
    pub fn find_or_create_audio_entry<P, D>(&mut self, pred: P, default_fn: D) -> &mut SampleEntry
    where
        P: Fn(&SampleEntry) -> bool,
        D: FnOnce() -> SampleEntry,
    {
        let mut index = None;
        let mut remove_indices: Vec<usize> = Vec::new();
        for (i, entry) in self.entries.iter_mut().enumerate() {
            if !entry.is_audio() {
                continue;
            }
            if pred(entry) {
                index = Some(i);
            } else {
                remove_indices.push(i);
            }
        }
        for i in remove_indices.into_iter() {
            if let Some(index) = index.as_mut() {
                if *index < i {
                    *index -= 1;
                }
            }
            self.entries.remove(i);
        }
        if let Some(index) = index {
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
        serializer::serialize_stsd_data(self)
    }
}

const FTAB: &[u8; 4] = b"ftab";

mod serializer {
    use crate::atom::{
        stsd::{
            ColorRgba, FontStyle, FontTableEntry, Mp4aEntryData, StyleRecord, TextBox,
            Tx3gEntryData, FTAB,
        },
        util::serializer::{be_u32, pascal_string, prepend_size, SizeU32, SizeU32OrU64},
    };

    use super::{SampleDescriptionTableAtom, SampleEntry, SampleEntryData};

    pub fn serialize_stsd_data(stsd: SampleDescriptionTableAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(stsd.version);
        data.extend(stsd.flags);

        data.extend(be_u32(
            stsd.entries
                .len()
                .try_into()
                .expect("stsd entries len should fit in u32"),
        ));

        for entry in stsd.entries {
            data.extend(serialize_stsd_entry(entry))
        }

        data
    }

    fn serialize_stsd_entry(entry: SampleEntry) -> Vec<u8> {
        prepend_size::<SizeU32OrU64, _>(move || {
            let mut data = Vec::new();

            data.extend(entry.entry_type().as_bytes());

            // Reserved (6 bytes)
            data.extend(&[0u8; 6]);

            data.extend(entry.data_reference_index.to_be_bytes());

            data.extend(serialize_entry_data(entry.data));

            data
        })
    }

    fn serialize_entry_data(entry_data: SampleEntryData) -> Vec<u8> {
        match entry_data {
            SampleEntryData::Mp4a(mp4a) => serialize_mp4a_entry_data(mp4a),
            SampleEntryData::Tx3g(tx3g) => serialize_tx3g_entry_data(tx3g),
            SampleEntryData::Unknown(_, raw) => {
                let mut data = Vec::new();
                data.extend(raw);
                data
            }
        }
    }

    fn serialize_mp4a_entry_data(mp4a: Mp4aEntryData) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend(mp4a.version.to_be_bytes());
        data.extend(mp4a.revision_level.to_be_bytes());
        data.extend(mp4a.vendor);
        data.extend(mp4a.channel_count.to_be_bytes());
        data.extend(mp4a.sample_size.to_be_bytes());
        data.extend(mp4a.compression_id.to_be_bytes());
        data.extend(mp4a.packet_size.to_be_bytes());
        data.extend(((mp4a.sample_rate * 65536.0) as u32).to_be_bytes());
        mp4a.extensions.into_iter().for_each(|ext| {
            let ext_data = ext.to_bytes::<SizeU32>();
            data.extend(ext_data);
        });

        data
    }

    fn serialize_tx3g_entry_data(tx3g: Tx3gEntryData) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend(tx3g.display_flags.to_be_bytes());
        data.extend(tx3g.horizontal_justification.to_be_bytes());
        data.extend(tx3g.vertical_justification.to_be_bytes());
        data.extend(serialize_color_rbga(tx3g.background_color));
        data.extend(serialize_text_box(tx3g.default_text_box));
        data.extend(serialize_style_record(tx3g.default_style_record));

        if let Some(font_table) = tx3g.font_table {
            data.extend(prepend_size::<SizeU32OrU64, _>(move || {
                let mut data = Vec::new();
                data.extend(FTAB);
                data.extend(
                    u16::try_from(font_table.len())
                        .expect("font table len must fit in u16")
                        .to_be_bytes(),
                );
                data.extend(font_table.into_iter().flat_map(serialize_font_table_entry));
                data
            }));
        }

        data
    }

    fn serialize_color_rbga(rgba: ColorRgba) -> Vec<u8> {
        vec![rgba.red, rgba.green, rgba.blue, rgba.alpha]
    }

    fn serialize_text_box(text: TextBox) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend(text.top.to_be_bytes());
        data.extend(text.left.to_be_bytes());
        data.extend(text.bottom.to_be_bytes());
        data.extend(text.right.to_be_bytes());

        data
    }

    fn serialize_style_record(style: StyleRecord) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend(style.start_char.to_be_bytes());
        data.extend(style.end_char.to_be_bytes());
        data.extend(style.font_id.to_be_bytes());
        data.push(serialize_font_style_flags(style.font_style_flags));
        data.push(style.font_size);
        data.extend(serialize_color_rbga(style.text_color));

        data
    }

    fn serialize_font_style_flags(flags: FontStyle) -> u8 {
        let bold = (flags.bold as u8) << 7;
        let italic = (flags.italic as u8) << 6;
        let underline = (flags.underline as u8) << 5;
        bold & italic & underline
    }

    fn serialize_font_table_entry(entry: FontTableEntry) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(entry.font_id.to_be_bytes());
        data.extend(pascal_string(entry.font_name));
        data
    }
}

mod parser {
    use winnow::{
        binary::{
            be_i16, be_u16, be_u32,
            bits::{bits, bool},
            i8, length_repeat, u8,
        },
        combinator::{opt, repeat, seq, trace},
        error::{ContextError, ErrMode, StrContext, StrContextValue},
        stream::ToUsize,
        token::{literal, rest},
        ModalResult, Parser,
    };

    use super::{Mp4aEntryData, SampleDescriptionTableAtom, SampleEntry, SampleEntryData};
    use crate::{
        atom::{
            stsd::{
                extension::parser::parse_stsd_extension, ColorRgba, FontStyle, FontTableEntry,
                StsdExtension, StyleRecord, TextBox, Tx3gEntryData, FTAB,
            },
            util::parser::{
                atom_size, byte_array, combinators::inclusive_length_and_then, fixed_point_16x16,
                flags3, fourcc, pascal_string, stream, version, Stream,
            },
        },
        FourCC,
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
                entries: length_repeat(be_u32, inclusive_length_and_then(atom_size, entry))
                    .context(StrContext::Label("entries")),
            })
            .context(StrContext::Label("stsd")),
        )
        .parse_next(input)
    }

    fn entry(input: &mut Stream<'_>) -> ModalResult<SampleEntry> {
        trace("entry", move |input: &mut Stream| {
            let typ = fourcc.parse_next(input)?;
            let _ = reserved.parse_next(input)?;
            let data_reference_index = be_u16
                .context(StrContext::Label("data_reference_index"))
                .parse_next(input)?;
            let data = match typ {
                Mp4aEntryData::TYPE => mp4a_sample_entry.parse_next(input)?,
                Tx3gEntryData::TYPE => tx3g_sample_entry.parse_next(input)?,
                _ => unknown_sample_entry(typ).parse_next(input)?,
            };
            Ok(SampleEntry {
                data_reference_index,
                data,
            })
        })
        .parse_next(input)
    }

    fn reserved(input: &mut Stream<'_>) -> ModalResult<[u8; 6]> {
        trace("reserved", byte_array).parse_next(input)
    }

    fn mp4a_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
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
                extensions: extensions(be_u32).context(StrContext::Label("extensions")),
            })
            .map(SampleEntryData::Mp4a),
        )
        .context(StrContext::Label("mp4a"))
        .parse_next(input)
    }

    fn tx3g_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        trace(
            "tx3g_sample_entry",
            seq!(Tx3gEntryData {
                display_flags: be_u32.context(StrContext::Label("display_flags")),
                horizontal_justification: i8.context(StrContext::Label("horizontal_justification")),
                vertical_justification: i8.context(StrContext::Label("vertical_justification")),
                background_color: color_rgba.context(StrContext::Label("background_color")),
                default_text_box: text_box.context(StrContext::Label("default_text_box")),
                default_style_record: style_record
                    .context(StrContext::Label("default_style_record")),
                font_table: opt(inclusive_length_and_then(
                    atom_size,
                    |input: &mut Stream<'_>| {
                        literal(FTAB).parse_next(input)?;
                        length_repeat(be_u16, font_table_entry).parse_next(input)
                    }
                ))
                .context(StrContext::Label("font_table")),
            })
            .map(SampleEntryData::Tx3g),
        )
        .context(StrContext::Label("mp4a"))
        .parse_next(input)
    }

    fn color_rgba(input: &mut Stream<'_>) -> ModalResult<ColorRgba> {
        trace(
            "color_rbga",
            seq!(ColorRgba {
                red: u8.context(StrContext::Label("red")),
                green: u8.context(StrContext::Label("green")),
                blue: u8.context(StrContext::Label("blue")),
                alpha: u8.context(StrContext::Label("alpha")),
            }),
        )
        .context(StrContext::Expected(StrContextValue::Description(
            "red, green, blue, alpha",
        )))
        .parse_next(input)
    }

    fn text_box(input: &mut Stream<'_>) -> ModalResult<TextBox> {
        trace(
            "text_box",
            seq!(TextBox {
                top: be_i16.context(StrContext::Label("top")),
                left: be_i16.context(StrContext::Label("left")),
                bottom: be_i16.context(StrContext::Label("bottom")),
                right: be_i16.context(StrContext::Label("right")),
            }),
        )
        .context(StrContext::Expected(StrContextValue::Description(
            "top, left, bottom, right",
        )))
        .parse_next(input)
    }

    fn style_record(input: &mut Stream<'_>) -> ModalResult<StyleRecord> {
        trace(
            "style_record",
            seq!(StyleRecord {
                start_char: be_u16.context(StrContext::Label("start_char")),
                end_char: be_u16.context(StrContext::Label("end_char")),
                font_id: be_u16.context(StrContext::Label("font_id")),
                font_style_flags: font_style_flags.context(StrContext::Label("font_style_flags")),
                font_size: u8.context(StrContext::Label("font_size")),
                text_color: color_rgba.context(StrContext::Label("text_color")),
            }),
        )
        .parse_next(input)
    }

    fn font_style_flags(input: &mut Stream<'_>) -> ModalResult<FontStyle> {
        trace(
            "font_style_flags",
            bits(
                move |input: &mut (Stream<'_>, usize)| -> ModalResult<FontStyle> {
                    seq!(FontStyle {
                        bold: bool.context(StrContext::Label("bold")),
                        italic: bool.context(StrContext::Label("italic")),
                        underline: bool.context(StrContext::Label("underline")),
                        // remaining 5 bits are discarded
                    })
                    .parse_next(input)
                },
            ),
        )
        .parse_next(input)
    }

    fn font_table_entry(input: &mut Stream<'_>) -> ModalResult<FontTableEntry> {
        trace(
            "font_table_entry",
            seq!(FontTableEntry {
                font_id: be_u16.context(StrContext::Label("font_id")),
                font_name: pascal_string.context(StrContext::Label("font_name")),
            }),
        )
        .parse_next(input)
    }

    fn unknown_sample_entry<'i>(
        typ: FourCC,
    ) -> impl Parser<Stream<'i>, SampleEntryData, ErrMode<ContextError>> {
        trace("other_sample_entry", move |input: &mut Stream<'_>| {
            rest.map(|buf: &[u8]| buf.to_vec())
                .map(|data| SampleEntryData::Unknown(typ, data))
                .parse_next(input)
        })
        .context(StrContext::Label("unknown"))
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
                inclusive_length_and_then(size_parser.by_ref(), |input: &mut Stream<'i>| {
                    let typ = fourcc.parse_next(input)?;
                    parse_stsd_extension(typ).parse_next(input)
                }),
            )
            .parse_next(input)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    use super::*;

    /// Test round-trip for all available stsd test data files
    #[test]
    fn test_stsd_roundtrip() {
        test_atom_roundtrip_sync::<SampleDescriptionTableAtom>(STSD);
    }
}
