use bon::Builder;

use crate::atom::util::ColorRgb;

use super::StsdExtension;

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
    #[builder(into)]
    pub font_name: String,
    #[builder(default)]
    pub extensions: Vec<StsdExtension>,
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

pub(super) mod serializer {
    use super::{DisplayFlags, FontFace, TextBox, TextJustification, TextSampleEntry};
    use crate::atom::{
        stsd::extension::serializer::serialize_stsd_extensions,
        util::serializer::{bits::Packer, color_rgb, pascal_string},
    };

    pub fn serialize_text_sample_entry(text: TextSampleEntry) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend(display_flags(text.display_flags));
        data.extend(text_justification(text.text_justification));
        data.extend(color_rgb(text.background_color));
        data.extend(text_box(text.default_text_box));
        data.extend([0u8; 8]); // reserved
        data.extend(text.font_number.to_be_bytes());
        data.extend(font_face(text.font_face));
        data.extend([0u8; 2]); // reserved
        data.extend(color_rgb(text.foreground_color));
        data.extend(pascal_string(text.font_name));
        data.extend(serialize_stsd_extensions(text.extensions));

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

pub(super) mod parser {
    use winnow::{
        binary::{be_i32, be_u16, bits},
        combinator::seq,
        error::{ContextError, ErrMode, StrContext},
        ModalResult, Parser,
    };

    use crate::atom::{
        stsd::{extension::parser::parse_stsd_extensions, SampleEntryData},
        util::parser::{byte_array, color_rgb, pascal_string, Stream},
    };

    use super::*;

    pub fn parse_text_sample_entry(input: &mut Stream<'_>) -> ModalResult<SampleEntryData> {
        seq!(TextSampleEntry {
            display_flags: bits::bits(display_flags).context(StrContext::Label("display_flags")),
            text_justification: text_justification.context(StrContext::Label("text_justification")),
            background_color: color_rgb.context(StrContext::Label("background_color")),
            default_text_box: text_box.context(StrContext::Label("default_text_box")),
            _: byte_array::<8>.context(StrContext::Label("reserved")),
            font_number: be_u16.context(StrContext::Label("font_number")),
            font_face: bits::bits(font_face).context(StrContext::Label("font_face")),
            _: byte_array::<2>.context(StrContext::Label("reserved")),
            foreground_color: color_rgb.context(StrContext::Label("foreground_color")),
            font_name: pascal_string.context(StrContext::Label("text_name")),
            extensions: parse_stsd_extensions.context(StrContext::Label("extensions")),
        })
        .map(SampleEntryData::Text)
        .parse_next(input)
    }

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
}
