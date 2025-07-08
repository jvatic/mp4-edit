use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::io::Read;

use crate::{
    atom::{util::async_to_sync_read, FourCC},
    parser::Parse,
    writer::SerializeAtom,
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

impl SerializeAtom for HandlerReferenceAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*HDLR)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version and flags (4 bytes)
        let version_flags = (self.version as u32) << 24
            | (self.flags[0] as u32) << 16
            | (self.flags[1] as u32) << 8
            | (self.flags[2] as u32);
        data.extend_from_slice(&version_flags.to_be_bytes());

        // Component type (4 bytes)
        data.extend_from_slice(&self.component_type);

        // Handler type (4 bytes)
        data.extend_from_slice(&self.handler_type.to_bytes());

        // Component manufacturer (4 bytes)
        data.extend_from_slice(&self.component_manufacturer);

        // Component flags (4 bytes)
        data.extend_from_slice(&self.component_flags.to_be_bytes());

        // Component flags mask (4 bytes)
        data.extend_from_slice(&self.component_flags_mask.to_be_bytes());

        // Handler name (null-terminated string)
        data.extend_from_slice(self.name.as_bytes());
        data.push(0); // Null terminator

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;
    use std::io::Cursor;

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

    #[test]
    fn test_hdlr_mdir_round_trip() {
        // Create the exact hdlr atom that Apple expects for metadata
        let original_hdlr = HandlerReferenceAtom {
            version: 0,
            flags: [0, 0, 0],
            component_type: [0, 0, 0, 0],
            handler_type: HandlerType::Mdir, // This should be 'mdir' for metadata
            component_manufacturer: [97, 112, 112, 108], // 'appl'
            component_flags: 0,
            component_flags_mask: 0,
            name: "".to_string(), // Empty name is common
        };

        // Convert to bytes
        let bytes: Vec<u8> = original_hdlr.clone().into_body_bytes();

        // Parse back from bytes
        let cursor = Cursor::new(&bytes);
        let parsed_hdlr = parse_hdlr_data(cursor).unwrap();

        // Verify all fields match
        assert_eq!(parsed_hdlr.version, original_hdlr.version);
        assert_eq!(parsed_hdlr.flags, original_hdlr.flags);
        assert_eq!(parsed_hdlr.component_type, original_hdlr.component_type);
        assert_eq!(parsed_hdlr.handler_type, original_hdlr.handler_type);
        assert_eq!(
            parsed_hdlr.component_manufacturer,
            original_hdlr.component_manufacturer
        );
        assert_eq!(parsed_hdlr.component_flags, original_hdlr.component_flags);
        assert_eq!(
            parsed_hdlr.component_flags_mask,
            original_hdlr.component_flags_mask
        );
        assert_eq!(parsed_hdlr.name, original_hdlr.name);
    }

    #[test]
    fn test_hdlr_raw_bytes_mdir() {
        // Test with raw bytes that represent a valid Apple metadata hdlr atom
        let raw_hdlr_data = vec![
            // version (1 byte) + flags (3 bytes)
            0x00, 0x00, 0x00, 0x00, // component_type (4 bytes) - usually zeros
            0x00, 0x00, 0x00, 0x00, // handler_type (4 bytes) - 'mdir' in ASCII
            0x6D, 0x64, 0x69, 0x72, // 'm', 'd', 'i', 'r'
            // component_manufacturer (4 bytes) - 'appl' in ASCII
            0x61, 0x70, 0x70, 0x6C, // 'a', 'p', 'p', 'l'
            // component_flags (4 bytes) - usually zero
            0x00, 0x00, 0x00, 0x00, // component_flags_mask (4 bytes) - usually zero
            0x00, 0x00, 0x00, 0x00, // name - empty (null terminated)
            0x00,
        ];

        let cursor = Cursor::new(&raw_hdlr_data);
        let parsed_hdlr = parse_hdlr_data(cursor).unwrap();

        // Verify the handler type is correctly parsed as mdir
        assert_eq!(parsed_hdlr.handler_type, HandlerType::Mdir);
        assert_eq!(parsed_hdlr.version, 0);
        assert_eq!(parsed_hdlr.flags, [0, 0, 0]);
        assert_eq!(parsed_hdlr.component_manufacturer, [0x61, 0x70, 0x70, 0x6C]); // 'appl'
        assert_eq!(parsed_hdlr.name, "");
    }

    #[test]
    fn test_hdlr_write_produces_correct_mdir_bytes() {
        let hdlr = HandlerReferenceAtom {
            version: 0,
            flags: [0, 0, 0],
            component_type: [0, 0, 0, 0],
            handler_type: HandlerType::Mdir,
            component_manufacturer: [97, 112, 112, 108], // 'appl'
            component_flags: 0,
            component_flags_mask: 0,
            name: "".to_string(),
        };

        let bytes: Vec<u8> = hdlr.into_body_bytes();

        // Check that the handler type bytes are exactly 'mdir'
        let handler_type_offset = 8; // version(1) + flags(3) + component_type(4) = 8
        assert_eq!(
            &bytes[handler_type_offset..handler_type_offset + 4],
            &[0x6D, 0x64, 0x69, 0x72] // 'mdir'
        );

        // Check that component manufacturer is 'appl'
        let manufacturer_offset = 12; // handler_type is at offset 8, so manufacturer at 12
        assert_eq!(
            &bytes[manufacturer_offset..manufacturer_offset + 4],
            &[0x61, 0x70, 0x70, 0x6C] // 'appl'
        );
    }

    #[test]
    fn test_hdlr_with_name() {
        let hdlr = HandlerReferenceAtom {
            version: 0,
            flags: [0, 0, 0],
            component_type: [0, 0, 0, 0],
            handler_type: HandlerType::Mdir,
            component_manufacturer: [97, 112, 112, 108], // 'appl'
            component_flags: 0,
            component_flags_mask: 0,
            name: "Apple Metadata Handler".to_string(),
        };

        // Round trip test
        let bytes: Vec<u8> = hdlr.clone().into_body_bytes();
        let cursor = Cursor::new(&bytes);
        let parsed_hdlr = parse_hdlr_data(cursor).unwrap();

        assert_eq!(parsed_hdlr.name, hdlr.name);
        assert_eq!(parsed_hdlr.handler_type, HandlerType::Mdir);
    }

    #[test]
    fn test_hdlr_unknown_handler_type() {
        // Test that unknown handler types are preserved correctly
        let raw_data = vec![
            0x00, 0x00, 0x00, 0x00, // version + flags
            0x00, 0x00, 0x00, 0x00, // component_type
            0x78, 0x79, 0x7A, 0x77, // 'xyzw' - unknown handler type
            0x61, 0x70, 0x70, 0x6C, // 'appl'
            0x00, 0x00, 0x00, 0x00, // component_flags
            0x00, 0x00, 0x00, 0x00, // component_flags_mask
            0x00, // empty name
        ];

        let cursor = Cursor::new(&raw_data);
        let parsed_hdlr = parse_hdlr_data(cursor).unwrap();

        // Should be parsed as Unknown with the correct bytes
        match parsed_hdlr.handler_type {
            HandlerType::Unknown(bytes) => {
                assert_eq!(bytes, [0x78, 0x79, 0x7A, 0x77]); // 'xyzw'
            }
            _ => panic!("Expected Unknown handler type"),
        }

        // Round trip should preserve the unknown type
        let output_bytes: Vec<u8> = parsed_hdlr.into_body_bytes();
        let cursor2 = Cursor::new(&output_bytes);
        let reparsed_hdlr = parse_hdlr_data(cursor2).unwrap();

        match reparsed_hdlr.handler_type {
            HandlerType::Unknown(bytes) => {
                assert_eq!(bytes, [0x78, 0x79, 0x7A, 0x77]);
            }
            _ => panic!("Expected Unknown handler type after round trip"),
        }
    }

    #[test]
    fn test_your_actual_hdlr_data() {
        // This tests the exact data you showed in your debug output
        let hdlr = HandlerReferenceAtom {
            version: 0,
            flags: [0, 0, 0],
            component_type: [0, 0, 0, 0],
            handler_type: HandlerType::Unknown([109, 100, 105, 114]), // Your actual data
            component_manufacturer: [97, 112, 112, 108],              // 'appl'
            component_flags: 0,
            component_flags_mask: 0,
            name: "".to_string(),
        };

        // Convert to bytes and back
        let bytes: Vec<u8> = hdlr.into_body_bytes();
        let cursor = Cursor::new(&bytes);
        let parsed_hdlr = parse_hdlr_data(cursor).unwrap();

        // The key test: those Unknown bytes [109, 100, 105, 114] should be 'mdir'
        // Your parser should recognize this as HandlerType::Mdir, not Unknown
        println!(
            "Handler type after round trip: {:?}",
            parsed_hdlr.handler_type
        );

        // Check the raw bytes in the output
        let handler_type_offset = 8;
        println!(
            "Raw handler type bytes: {:?}",
            &bytes[handler_type_offset..handler_type_offset + 4]
        );

        // These should be [109, 100, 105, 114] which is 'mdir'
        assert_eq!(
            &bytes[handler_type_offset..handler_type_offset + 4],
            &[109, 100, 105, 114]
        );
    }

    /// Test round-trip for all available hdlr test data files
    #[test]
    fn test_ftyp_roundtrip() {
        test_atom_roundtrip_sync::<HandlerReferenceAtom>(HDLR);
    }
}
