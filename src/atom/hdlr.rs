use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::io::Read;

use crate::{
    atom::{util::async_to_sync_read, FourCC},
    parser::Parse,
};

pub const HDLR: &[u8; 4] = b"hdlr";

// Common handler types
pub const HANDLER_VIDEO: &[u8; 4] = b"vide";
pub const HANDLER_AUDIO: &[u8; 4] = b"soun";
pub const HANDLER_HINT: &[u8; 4] = b"hint";
pub const HANDLER_META: &[u8; 4] = b"meta";
pub const HANDLER_TEXT: &[u8; 4] = b"text";
pub const HANDLER_SUBTITLE: &[u8; 4] = b"subt";
pub const HANDLER_TIMECODE: &[u8; 4] = b"tmcd";

#[derive(Debug, Clone, PartialEq)]
pub enum HandlerType {
    Video,
    Audio,
    Hint,
    Meta,
    Text,
    Subtitle,
    Timecode,
    Unknown([u8; 4]),
}

impl HandlerType {
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        match bytes {
            HANDLER_VIDEO => HandlerType::Video,
            HANDLER_AUDIO => HandlerType::Audio,
            HANDLER_HINT => HandlerType::Hint,
            HANDLER_META => HandlerType::Meta,
            HANDLER_TEXT => HandlerType::Text,
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

#[derive(Debug, Clone)]
pub struct HandlerReferenceAtom {
    /// Version of the hdlr atom format (0)
    pub version: u8,
    /// Flags for the hdlr atom (usually all zeros)
    pub flags: [u8; 3],
    /// Component type (pre-defined, usually 0)
    pub component_type: [u8; 4],
    /// Handler type (4CC code indicating the type of media handler)
    pub handler_type: HandlerType,
    /// Component manufacturer (usually 0)
    pub component_manufacturer: [u8; 4],
    /// Component flags (usually 0)
    pub component_flags: u32,
    /// Component flags mask (usually 0)
    pub component_flags_mask: u32,
    /// Human-readable name of the handler (null-terminated string)
    pub name: String,
}

impl Parse for HandlerReferenceAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, anyhow::Error> {
        if atom_type != HDLR {
            return Err(anyhow!("Invalid atom type: {}", atom_type));
        }
        parse_hdlr_data(async_to_sync_read(reader).await?)
    }
}

fn parse_hdlr_data<R: Read>(mut reader: R) -> Result<HandlerReferenceAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Validate version
    if version != 0 {
        return Err(anyhow!("unsupported version {}", version));
    }

    // Read component type (4 bytes, pre-defined)
    let mut component_type = [0u8; 4];
    reader
        .read_exact(&mut component_type)
        .context("read component_type")?;

    // Read handler type (4 bytes)
    let mut handler_type_bytes = [0u8; 4];
    reader
        .read_exact(&mut handler_type_bytes)
        .context("read handler_type")?;
    let handler_type = HandlerType::from_bytes(&handler_type_bytes);

    // Read component manufacturer (4 bytes)
    let mut component_manufacturer = [0u8; 4];
    reader
        .read_exact(&mut component_manufacturer)
        .context("read component_manufacturer")?;

    // Read component flags (4 bytes)
    let mut flags_buf = [0u8; 4];
    reader
        .read_exact(&mut flags_buf)
        .context("read component_flags")?;
    let component_flags = u32::from_be_bytes(flags_buf);

    // Read component flags mask (4 bytes)
    reader
        .read_exact(&mut flags_buf)
        .context("read component_flags_mask")?;
    let component_flags_mask = u32::from_be_bytes(flags_buf);

    // Read the remaining data as the name (null-terminated string)
    let mut name_bytes = Vec::new();
    reader
        .read_to_end(&mut name_bytes)
        .context("read handler name")?;

    // Parse the name string
    let name = parse_handler_name(&name_bytes)?;

    Ok(HandlerReferenceAtom {
        version,
        flags,
        component_type,
        handler_type,
        component_manufacturer,
        component_flags,
        component_flags_mask,
        name,
    })
}

fn parse_handler_name(name_bytes: &[u8]) -> Result<String, anyhow::Error> {
    if name_bytes.is_empty() {
        return Ok(String::new());
    }

    // The name can be either:
    // 1. A Pascal string (first byte is length, followed by string data)
    // 2. A C string (null-terminated)
    // 3. Just raw string data

    let name_str = if name_bytes.len() > 1 && name_bytes[0] as usize == name_bytes.len() - 1 {
        // Pascal string format
        let length = name_bytes[0] as usize;
        if length > 0 && length < name_bytes.len() {
            std::str::from_utf8(&name_bytes[1..=length])
                .context("invalid UTF-8 in Pascal string")?
        } else {
            ""
        }
    } else {
        // Try to find null terminator
        let end_pos = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        std::str::from_utf8(&name_bytes[..end_pos]).context("invalid UTF-8 in handler name")?
    };

    Ok(name_str.to_string())
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
        let result = parse_handler_name(pascal_name).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_parse_handler_name_null_terminated() {
        // C-style null-terminated string
        let c_name = b"Hello World!\0";
        let result = parse_handler_name(c_name).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_parse_handler_name_raw() {
        // Raw string without null terminator
        let raw_name = b"Hello World!";
        let result = parse_handler_name(raw_name).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_parse_handler_name_empty() {
        let empty_name = b"";
        let result = parse_handler_name(empty_name).unwrap();
        assert_eq!(result, "");
    }
}

impl From<HandlerReferenceAtom> for Vec<u8> {
    fn from(atom: HandlerReferenceAtom) -> Self {
        let mut data = Vec::new();

        // Version and flags (4 bytes)
        let version_flags = (atom.version as u32) << 24
            | (atom.flags[0] as u32) << 16
            | (atom.flags[1] as u32) << 8
            | (atom.flags[2] as u32);
        data.extend_from_slice(&version_flags.to_be_bytes());

        // Component type (4 bytes)
        data.extend_from_slice(&atom.component_type);

        // Handler type (4 bytes)
        data.extend_from_slice(&atom.handler_type.to_bytes());

        // Component manufacturer (4 bytes)
        data.extend_from_slice(&atom.component_manufacturer);

        // Component flags (4 bytes)
        data.extend_from_slice(&atom.component_flags.to_be_bytes());

        // Component flags mask (4 bytes)
        data.extend_from_slice(&atom.component_flags_mask.to_be_bytes());

        // Name (null-terminated string)
        if !atom.name.is_empty() {
            data.extend_from_slice(atom.name.as_bytes());
        }
        data.push(0); // Null terminator

        data
    }
}
