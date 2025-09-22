use bon::Builder;
use core::fmt;
use futures_io::AsyncRead;

use crate::{
    atom::{util::read_to_end, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const HDLR: &[u8; 4] = b"hdlr";

// Common handler types
pub const HANDLER_VIDEO: &[u8; 4] = b"vide";
pub const HANDLER_AUDIO: &[u8; 4] = b"soun";
pub const HANDLER_HINT: &[u8; 4] = b"hint";
pub const HANDLER_META: &[u8; 4] = b"meta";
pub const HANDLER_TEXT: &[u8; 4] = b"text";
pub const HANDLER_MDIR: &[u8; 4] = b"mdir";
pub const HANDLER_SUBTITLE: &[u8; 4] = b"subt";
pub const HANDLER_TIMECODE: &[u8; 4] = b"tmcd";

#[derive(Debug, Clone, PartialEq)]
pub enum HandlerType {
    Video,
    Audio,
    Hint,
    Meta,
    Text,
    Mdir,
    Subtitle,
    Timecode,
    Unknown([u8; 4]),
}

impl Default for HandlerType {
    fn default() -> Self {
        Self::Unknown([0u8; 4])
    }
}

impl HandlerType {
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        match bytes {
            HANDLER_VIDEO => HandlerType::Video,
            HANDLER_AUDIO => HandlerType::Audio,
            HANDLER_HINT => HandlerType::Hint,
            HANDLER_META => HandlerType::Meta,
            HANDLER_TEXT => HandlerType::Text,
            HANDLER_MDIR => HandlerType::Mdir,
            HANDLER_SUBTITLE => HandlerType::Subtitle,
            HANDLER_TIMECODE => HandlerType::Timecode,
            _ => HandlerType::Unknown(*bytes),
        }
    }

    pub fn to_bytes(&self) -> [u8; 4] {
        match self {
            HandlerType::Video => *HANDLER_VIDEO,
            HandlerType::Audio => *HANDLER_AUDIO,
            HandlerType::Hint => *HANDLER_HINT,
            HandlerType::Meta => *HANDLER_META,
            HandlerType::Text => *HANDLER_TEXT,
            HandlerType::Mdir => *HANDLER_MDIR,
            HandlerType::Subtitle => *HANDLER_SUBTITLE,
            HandlerType::Timecode => *HANDLER_TIMECODE,
            HandlerType::Unknown(bytes) => *bytes,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            HandlerType::Video => "Video",
            HandlerType::Audio => "Audio",
            HandlerType::Hint => "Hint",
            HandlerType::Meta => "Metadata",
            HandlerType::Text => "Text",
            HandlerType::Mdir => "Mdir",
            HandlerType::Subtitle => "Subtitle",
            HandlerType::Timecode => "Timecode",
            HandlerType::Unknown(_) => "Unknown",
        }
    }

    pub fn is_media_handler(&self) -> bool {
        matches!(
            self,
            HandlerType::Video | HandlerType::Audio | HandlerType::Text | HandlerType::Subtitle
        )
    }
}

#[derive(Default, Debug, Clone, Builder)]
pub struct HandlerReferenceAtom {
    /// Version of the hdlr atom format (0)
    #[builder(default = 0)]
    pub version: u8,
    /// Flags for the hdlr atom (usually all zeros)
    #[builder(default = [0u8; 3])]
    pub flags: [u8; 3],
    /// Component type (pre-defined, usually 0)
    #[builder(default = [0u8; 4])]
    pub component_type: [u8; 4],
    /// Handler type (4CC code indicating the type of media handler)
    pub handler_type: HandlerType,
    /// Component manufacturer (usually 0)
    #[builder(default = [0u8; 4])]
    pub component_manufacturer: [u8; 4],
    /// Component flags (usually 0)
    #[builder(default = 0)]
    pub component_flags: u32,
    /// Component flags mask (usually 0)
    #[builder(default = 0)]
    pub component_flags_mask: u32,
    /// Human-readable name of the handler
    #[builder(into)]
    pub name: Option<HandlerName>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlerName {
    /// just string data
    Raw(String),
    /// length followed by string data
    Pascal(String),
    /// null terminated string
    CString(String),
    /// double-null terminated string
    CString2(String),
}

impl Default for HandlerName {
    fn default() -> Self {
        Self::Raw(String::new())
    }
}

impl fmt::Display for HandlerName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt::Display::fmt(self.as_string(), f)
    }
}

impl From<String> for HandlerName {
    fn from(value: String) -> Self {
        Self::CString(value)
    }
}

impl From<&String> for HandlerName {
    fn from(value: &String) -> Self {
        Self::CString(value.to_owned())
    }
}

impl From<&str> for HandlerName {
    fn from(value: &str) -> Self {
        Self::CString(value.to_owned())
    }
}

impl HandlerName {
    fn as_string(&self) -> &String {
        match self {
            Self::Raw(str) => str,
            Self::Pascal(str) => str,
            Self::CString(str) => str,
            Self::CString2(str) => str,
        }
    }

    pub fn as_str(&self) -> &str {
        self.as_string().as_str()
    }
}

impl ParseAtom for HandlerReferenceAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != HDLR {
            return Err(ParseError::new_unexpected_atom(atom_type, HDLR));
        }
        let data = read_to_end(reader).await?;
        parser::parse_hdlr_data(&data)
    }
}

impl SerializeAtom for HandlerReferenceAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*HDLR)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_hdlr_atom(self)
    }
}

mod serializer {
    use super::{HandlerName, HandlerReferenceAtom, HandlerType};

    pub fn serialize_hdlr_atom(atom: HandlerReferenceAtom) -> Vec<u8> {
        vec![
            version(atom.version),
            flags(atom.flags),
            component_type(atom.component_type),
            handler_type(atom.handler_type),
            component_manufacturer(atom.component_manufacturer),
            component_flags(atom.component_flags),
            component_flags_mask(atom.component_flags_mask),
            atom.name.map(name).unwrap_or_default(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn version(version: u8) -> Vec<u8> {
        vec![version]
    }

    fn flags(flags: [u8; 3]) -> Vec<u8> {
        flags.to_vec()
    }

    fn component_type(component_type: [u8; 4]) -> Vec<u8> {
        component_type.to_vec()
    }

    fn handler_type(handler_type: HandlerType) -> Vec<u8> {
        handler_type.to_bytes().to_vec()
    }

    fn component_manufacturer(manufacturer: [u8; 4]) -> Vec<u8> {
        manufacturer.to_vec()
    }

    fn component_flags(flags: u32) -> Vec<u8> {
        flags.to_be_bytes().to_vec()
    }

    fn component_flags_mask(flags_mask: u32) -> Vec<u8> {
        flags_mask.to_be_bytes().to_vec()
    }

    fn name(name: HandlerName) -> Vec<u8> {
        let mut data = Vec::new();
        match name {
            HandlerName::Pascal(name) => {
                let name_bytes = name.as_bytes();
                let len = u8::try_from(name_bytes.len())
                    .expect("HandlerName::Pascal length must not exceed u8::MAX");
                data.push(len);
                data.extend_from_slice(name_bytes);
            }
            HandlerName::CString(name) => {
                data.extend_from_slice(name.as_bytes());
                data.push(0); // Null terminator
            }
            HandlerName::CString2(name) => {
                data.extend_from_slice(name.as_bytes());
                data.push(0); // 1st null terminator
                data.push(0); // 2nd null terminator
            }
            HandlerName::Raw(name) => {
                data.extend_from_slice(name.as_bytes());
            }
        }
        data
    }
}

mod parser {
    use winnow::{
        binary::{be_u32, length_take, u8},
        combinator::{alt, opt, repeat_till, seq, trace},
        error::StrContext,
        token::{literal, rest},
        ModalResult, Parser,
    };

    use super::{HandlerName, HandlerReferenceAtom, HandlerType};
    use crate::atom::util::parser::{byte_array, flags3, stream, version, Stream};

    pub fn parse_hdlr_data(input: &[u8]) -> Result<HandlerReferenceAtom, crate::ParseError> {
        parse_hdlr_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_hdlr_data_inner(input: &mut Stream<'_>) -> ModalResult<HandlerReferenceAtom> {
        trace(
            "hdlr",
            seq!(HandlerReferenceAtom {
                version: version,
                flags: flags3,
                component_type: component_type,
                handler_type: handler_type,
                component_manufacturer: component_manufacturer,
                component_flags: component_flags,
                component_flags_mask: component_flags_mask,
                name: opt(name),
            })
            .context(StrContext::Label("hdlr")),
        )
        .parse_next(input)
    }

    fn component_type(input: &mut Stream<'_>) -> ModalResult<[u8; 4]> {
        trace(
            "component_type",
            byte_array.context(StrContext::Label("component_type")),
        )
        .parse_next(input)
    }

    fn handler_type(input: &mut Stream<'_>) -> ModalResult<HandlerType> {
        trace(
            "handler_type",
            byte_array
                .map(|fourcc| HandlerType::from_bytes(&fourcc))
                .context(StrContext::Label("handler_type")),
        )
        .parse_next(input)
    }

    fn component_manufacturer(input: &mut Stream<'_>) -> ModalResult<[u8; 4]> {
        trace(
            "component_manufacturer",
            byte_array.context(StrContext::Label("component_manufacturer")),
        )
        .parse_next(input)
    }

    fn component_flags(input: &mut Stream<'_>) -> ModalResult<u32> {
        trace(
            "component_flags",
            be_u32.context(StrContext::Label("component_flags")),
        )
        .parse_next(input)
    }

    fn component_flags_mask(input: &mut Stream<'_>) -> ModalResult<u32> {
        trace(
            "component_flags_mask",
            be_u32.context(StrContext::Label("component_flags_mask")),
        )
        .parse_next(input)
    }

    fn name(input: &mut Stream<'_>) -> ModalResult<HandlerName> {
        trace(
            "name",
            alt((name_cstr, name_pascal, name_raw)).context(StrContext::Label("name")),
        )
        .parse_next(input)
    }

    fn name_cstr(input: &mut Stream<'_>) -> ModalResult<HandlerName> {
        trace(
            "name_cstr",
            repeat_till(1.., u8, null_term).map(|(data, null2): (Vec<u8>, Option<()>)| {
                let str = String::from_utf8_lossy(&data).to_string();
                match null2 {
                    Some(_) => HandlerName::CString2(str),
                    None => HandlerName::CString(str),
                }
            }),
        )
        .parse_next(input)
    }

    fn null_term(input: &mut Stream<'_>) -> ModalResult<Option<()>> {
        trace(
            "null_term",
            (literal(0x00), opt(literal(0x00))).map(|(_, null2)| null2.map(|_| ())),
        )
        .parse_next(input)
    }

    fn name_pascal(input: &mut Stream<'_>) -> ModalResult<HandlerName> {
        trace(
            "name_pascal",
            length_take(u8)
                .map(|data| HandlerName::Pascal(String::from_utf8_lossy(data).to_string())),
        )
        .parse_next(input)
    }

    fn name_raw(input: &mut Stream<'_>) -> ModalResult<HandlerName> {
        trace(
            "name_raw",
            rest.map(|data| HandlerName::Raw(String::from_utf8_lossy(data).to_string())),
        )
        .parse_next(input)
    }

    #[cfg(test)]
    mod tests {

        use super::*;

        #[test]
        fn test_handler_type_from_bytes() {
            assert_eq!(HandlerType::from_bytes(b"vide"), HandlerType::Video);
            assert_eq!(HandlerType::from_bytes(b"soun"), HandlerType::Audio);
            assert_eq!(HandlerType::from_bytes(b"text"), HandlerType::Text);
            assert_eq!(HandlerType::from_bytes(b"meta"), HandlerType::Meta);
            assert_eq!(
                HandlerType::from_bytes(b"abcd"),
                HandlerType::Unknown(*b"abcd")
            );
        }

        #[test]
        fn test_handler_type_methods() {
            let video_handler = HandlerType::Video;
            assert!(video_handler.is_media_handler());
            assert_eq!(video_handler.as_str(), "Video");
            assert_eq!(video_handler.to_bytes(), *b"vide");

            let unknown_handler = HandlerType::Unknown(*b"test");
            assert!(!unknown_handler.is_media_handler());
            assert_eq!(unknown_handler.as_str(), "Unknown");
            assert_eq!(unknown_handler.to_bytes(), *b"test");
        }

        #[test]
        fn test_parse_handler_name_pascal() {
            // Pascal string: length byte followed by string
            let pascal_name = b"\x0CHello World!";
            let result = name.parse(stream(pascal_name)).unwrap();
            assert_eq!(result, HandlerName::Pascal("Hello World!".to_owned()));
        }

        #[test]
        fn test_parse_handler_name_null_terminated() {
            // C-style null-terminated string
            let c_name = b"Hello World!\0";
            let result = name.parse(stream(c_name)).unwrap();
            assert_eq!(result, HandlerName::CString("Hello World!".to_owned()));
        }

        #[test]
        fn test_parse_handler_name_double_null_terminated() {
            // C-style null-terminated string
            let c_name = b"Hello World!\0\0";
            let result = name.parse(stream(c_name)).unwrap();
            assert_eq!(result, HandlerName::CString2("Hello World!".to_owned()));
        }

        #[test]
        fn test_parse_handler_name_raw() {
            // Raw string without null terminator
            let raw_name = b"Hello World!";
            let result = name.parse(stream(raw_name)).unwrap();
            assert_eq!(result, HandlerName::Raw("Hello World!".to_owned()));
        }

        #[test]
        fn test_parse_handler_name_empty() {
            let empty_name = b"";
            let result = name.parse(stream(empty_name)).unwrap();
            assert_eq!(result, HandlerName::Raw(String::new()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available hdlr test data files
    #[test]
    fn test_hdlr_roundtrip() {
        test_atom_roundtrip_sync::<HandlerReferenceAtom>(HDLR);
    }
}
