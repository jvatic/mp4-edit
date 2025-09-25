use crate::FourCC;

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
    pub const TYPE: FourCC = FourCC::new(b"tx3g");
}

const FTAB: &[u8; 4] = b"ftab";

pub mod serializer {
    use super::{ColorRgba, FontStyle, FontTableEntry, StyleRecord, TextBox, Tx3gEntryData, FTAB};
    use crate::atom::util::serializer::{pascal_string, prepend_size, SizeU32OrU64};

    pub fn serialize_tx3g_entry_data(tx3g: Tx3gEntryData) -> Vec<u8> {
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

pub mod parser {
    use winnow::{
        binary::{
            be_i16, be_u16, be_u32,
            bits::{bits, bool},
            i8, length_repeat, u8,
        },
        combinator::{opt, seq, trace},
        error::{StrContext, StrContextValue},
        token::literal,
        ModalResult, Parser,
    };

    use super::{ColorRgba, FontStyle, FontTableEntry, StyleRecord, TextBox, Tx3gEntryData, FTAB};
    use crate::atom::{
        stsd::SampleEntryData,
        util::parser::{atom_size, combinators::inclusive_length_and_then, pascal_string, Stream},
    };

    pub fn tx3g_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
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
}
