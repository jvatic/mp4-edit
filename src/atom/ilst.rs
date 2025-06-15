use anyhow::{anyhow, Context};
use futures_io::AsyncRead;
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::{
    atom::{
        util::{async_to_sync_read, DebugEllipsis},
        FourCC,
    },
    parser::Parse,
};

pub const ILST: &[u8; 4] = b"ilst";

/// Newtype wrapper for metadata items with custom Debug implementation
#[derive(Clone)]
pub struct MetadataItems(pub Vec<MetadataItem>);

impl fmt::Debug for MetadataItems {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() <= 10 {
            return f.debug_list().entries(self.0.iter()).finish();
        }
        f.debug_list()
            .entries(self.0.iter().take(10))
            .entry(&DebugEllipsis(Some(self.0.len() - 10)))
            .finish()
    }
}

impl std::ops::Deref for MetadataItems {
    type Target = Vec<MetadataItem>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for MetadataItems {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Newtype wrapper for raw metadata items with custom Debug implementation
#[derive(Clone)]
pub struct RawMetadataItems(pub Vec<RawMetadataItem>);

impl fmt::Debug for RawMetadataItems {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.len() <= 10 {
            return f.debug_list().entries(self.0.iter()).finish();
        }
        f.debug_list()
            .entries(self.0.iter().take(10))
            .entry(&DebugEllipsis(Some(self.0.len() - 10)))
            .finish()
    }
}

impl std::ops::Deref for RawMetadataItems {
    type Target = Vec<RawMetadataItem>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for RawMetadataItems {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
pub struct ItemListAtom {
    /// Collection of metadata items in order
    pub items: MetadataItems,
    /// Raw data for items we couldn't parse
    pub raw_items: RawMetadataItems,
}

#[derive(Clone)]
pub struct MetadataItem {
    /// The type of metadata (e.g., "©nam" for title, "©ART" for artist)
    pub item_type: String,
    /// The actual metadata value
    pub value: MetadataValue,
    /// Additional metadata about this item
    pub metadata: ItemMetadata,
}

impl fmt::Debug for MetadataItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetadataItem")
            .field("item_type", &self.item_type)
            .field("value", &self.value)
            .field("metadata", &DebugEllipsis(None))
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct ItemMetadata {
    /// Data type flags
    pub type_flags: u32,
    /// Country code (if applicable)
    pub country: u32,
    /// Language code (if applicable)
    pub language: u32,
}

#[derive(Clone)]
pub enum MetadataValue {
    /// UTF-8 text string
    Text(String),
    /// Binary data
    Binary(Vec<u8>),
    /// JPEG image data
    Jpeg(Vec<u8>),
    /// PNG image data
    Png(Vec<u8>),
    /// Unsigned 8-bit integer
    UInt8(u8),
    /// Unsigned 16-bit integer
    UInt16(u16),
    /// Unsigned 32-bit integer
    UInt32(u32),
    /// Unsigned 64-bit integer
    UInt64(u64),
    /// Signed 8-bit integer
    Int8(i8),
    /// Signed 16-bit integer
    Int16(i16),
    /// Signed 32-bit integer
    Int32(i32),
    /// Signed 64-bit integer
    Int64(i64),
    /// Float value
    Float32(f32),
    /// Double value
    Float64(f64),
}

impl fmt::Debug for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataValue::Text(text) => {
                if text.len() <= 50 {
                    f.debug_tuple("Text").field(text).finish()
                } else {
                    let truncated = format!("{}...", &text[..50]);
                    f.debug_tuple("Text").field(&truncated).finish()
                }
            }
            MetadataValue::Binary(data) => f
                .debug_tuple("Binary")
                .field(&format!("<{} bytes>", data.len()))
                .finish(),
            MetadataValue::Jpeg(data) => f
                .debug_tuple("Jpeg")
                .field(&format!("<{} bytes>", data.len()))
                .finish(),
            MetadataValue::Png(data) => f
                .debug_tuple("Png")
                .field(&format!("<{} bytes>", data.len()))
                .finish(),
            MetadataValue::UInt8(val) => f.debug_tuple("UInt8").field(val).finish(),
            MetadataValue::UInt16(val) => f.debug_tuple("UInt16").field(val).finish(),
            MetadataValue::UInt32(val) => f.debug_tuple("UInt32").field(val).finish(),
            MetadataValue::UInt64(val) => f.debug_tuple("UInt64").field(val).finish(),
            MetadataValue::Int8(val) => f.debug_tuple("Int8").field(val).finish(),
            MetadataValue::Int16(val) => f.debug_tuple("Int16").field(val).finish(),
            MetadataValue::Int32(val) => f.debug_tuple("Int32").field(val).finish(),
            MetadataValue::Int64(val) => f.debug_tuple("Int64").field(val).finish(),
            MetadataValue::Float32(val) => f.debug_tuple("Float32").field(val).finish(),
            MetadataValue::Float64(val) => f.debug_tuple("Float64").field(val).finish(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RawMetadataItem {
    /// 4-byte type identifier
    pub item_type: [u8; 4],
    /// Raw item data
    pub data: Vec<u8>,
}

impl Parse for ItemListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, anyhow::Error> {
        if atom_type != ILST {
            return Err(anyhow!(
                "Invalid atom type: expected ilst, got {:?}",
                atom_type
            ));
        }
        parse_ilst_data(async_to_sync_read(reader).await?)
    }
}

impl From<ItemListAtom> for Vec<u8> {
    fn from(atom: ItemListAtom) -> Self {
        let mut data = Vec::new();

        // Serialize parsed metadata items
        for item in atom.items.iter() {
            let mut item_data = Vec::new();

            // Add 'data' atom
            let mut data_atom = Vec::new();

            // Type flags and metadata
            data_atom.extend_from_slice(&item.metadata.type_flags.to_be_bytes());
            data_atom.extend_from_slice(&item.metadata.country.to_be_bytes());
            data_atom.extend_from_slice(&item.metadata.language.to_be_bytes());

            // Value data based on type
            match &item.value {
                MetadataValue::Text(text) => {
                    data_atom.extend_from_slice(text.as_bytes());
                }
                MetadataValue::Binary(bytes) => {
                    data_atom.extend_from_slice(bytes);
                }
                MetadataValue::Jpeg(bytes) => {
                    data_atom.extend_from_slice(bytes);
                }
                MetadataValue::Png(bytes) => {
                    data_atom.extend_from_slice(bytes);
                }
                MetadataValue::UInt8(val) => {
                    data_atom.push(*val);
                }
                MetadataValue::UInt16(val) => {
                    data_atom.extend_from_slice(&val.to_be_bytes());
                }
                MetadataValue::UInt32(val) => {
                    data_atom.extend_from_slice(&val.to_be_bytes());
                }
                MetadataValue::UInt64(val) => {
                    data_atom.extend_from_slice(&val.to_be_bytes());
                }
                MetadataValue::Int8(val) => {
                    data_atom.push(*val as u8);
                }
                MetadataValue::Int16(val) => {
                    data_atom.extend_from_slice(&val.to_be_bytes());
                }
                MetadataValue::Int32(val) => {
                    data_atom.extend_from_slice(&val.to_be_bytes());
                }
                MetadataValue::Int64(val) => {
                    data_atom.extend_from_slice(&val.to_be_bytes());
                }
                MetadataValue::Float32(val) => {
                    data_atom.extend_from_slice(&val.to_be_bytes());
                }
                MetadataValue::Float64(val) => {
                    data_atom.extend_from_slice(&val.to_be_bytes());
                }
            }

            // Write 'data' atom size and type
            let data_atom_size = 8 + data_atom.len();
            item_data.extend_from_slice(&(data_atom_size as u32).to_be_bytes());
            item_data.extend_from_slice(b"data");
            item_data.extend_from_slice(&data_atom);

            // Write item size, type and data
            let item_size = 8 + item_data.len();
            data.extend_from_slice(&(item_size as u32).to_be_bytes());

            // Convert item_type string to 4 bytes using Mac Roman encoding
            let type_bytes = convert_utf8_to_mac_roman(&item.item_type);
            if type_bytes.len() >= 4 {
                data.extend_from_slice(&type_bytes[..4]);
            } else {
                let mut padded = [0u8; 4];
                padded[..type_bytes.len()].copy_from_slice(&type_bytes);
                data.extend_from_slice(&padded);
            }

            data.extend_from_slice(&item_data);
        }

        // Serialize raw metadata items
        for raw_item in atom.raw_items.iter() {
            // Item size and raw data
            let item_size = 8 + raw_item.data.len();
            data.extend_from_slice(&(item_size as u32).to_be_bytes());

            // Item type (4 bytes)
            data.extend_from_slice(&raw_item.item_type);

            data.extend_from_slice(&raw_item.data);
        }

        data
    }
}

fn parse_ilst_data<R: Read>(mut reader: R) -> Result<ItemListAtom, anyhow::Error> {
    let mut items = Vec::new();
    let mut raw_items = Vec::new();

    // Parse individual metadata items
    loop {
        // Try to read item size (4 bytes)
        let mut size_bytes = [0u8; 4];
        match reader.read_exact(&mut size_bytes) {
            Ok(()) => {
                let item_size = u32::from_be_bytes(size_bytes) as usize;

                // Validate size (must be at least 8: 4 for size + 4 for type)
                if item_size < 8 {
                    break;
                }

                // Read the item data (excluding the size field we just read)
                let item_data_size = item_size - 4;
                let mut item_data = vec![0u8; item_data_size];
                reader
                    .read_exact(&mut item_data)
                    .context("read item data")?;

                // Create a cursor for this item's data
                let mut item_cursor = Cursor::new(&item_data);

                match parse_metadata_item(&mut item_cursor, item_data_size) {
                    Ok(item) => {
                        items.push(item);
                    }
                    Err(_) => {
                        // Try to parse as raw item
                        let mut raw_cursor = Cursor::new(&item_data);
                        if let Ok(raw_item) =
                            parse_raw_metadata_item(&mut raw_cursor, item_data_size)
                        {
                            raw_items.push(raw_item);
                        }
                        // Continue regardless of raw parsing result
                    }
                }
            }
            Err(_) => {
                // No more data to read
                break;
            }
        }
    }

    Ok(ItemListAtom {
        items: MetadataItems(items),
        raw_items: RawMetadataItems(raw_items),
    })
}

fn parse_metadata_item<R: Read>(
    reader: &mut R,
    item_size: usize,
) -> Result<MetadataItem, anyhow::Error> {
    if item_size < 8 {
        return Err(anyhow!("Item size too small: {}", item_size));
    }

    // Read item type (4 bytes)
    let mut type_bytes = [0u8; 4];
    reader
        .read_exact(&mut type_bytes)
        .context("read item type")?;

    // Convert item type bytes to string, handling Mac Roman encoding for MP4 metadata
    let item_type = convert_mac_roman_to_utf8(&type_bytes);

    // Read item data (remaining bytes)
    let data_size = item_size - 4; // Subtract type field (size already excluded from slice)
    let mut item_data = vec![0u8; data_size];
    reader
        .read_exact(&mut item_data)
        .context("read item data")?;

    // Parse the data atoms within this item
    let mut data_cursor = Cursor::new(&item_data);
    let (value, metadata) = parse_item_data(&mut data_cursor)?;

    Ok(MetadataItem {
        item_type,
        value,
        metadata,
    })
}

fn parse_item_data<R: Read>(
    reader: &mut R,
) -> Result<(MetadataValue, ItemMetadata), anyhow::Error> {
    // Read data atom size (4 bytes)
    let mut size_bytes = [0u8; 4];
    reader
        .read_exact(&mut size_bytes)
        .context("read data atom size")?;
    let data_size = u32::from_be_bytes(size_bytes) as usize;

    if data_size < 12 {
        return Err(anyhow!("Data atom size too small: {}", data_size));
    }

    // Read data atom type (4 bytes) - should be "data"
    let mut type_bytes = [0u8; 4];
    reader
        .read_exact(&mut type_bytes)
        .context("read data atom type")?;

    if &type_bytes != b"data" {
        return Err(anyhow!("Expected 'data' atom, got {:?}", type_bytes));
    }

    // Read type flags (4 bytes)
    let mut flags_bytes = [0u8; 4];
    reader
        .read_exact(&mut flags_bytes)
        .context("read type flags")?;
    let type_flags = u32::from_be_bytes(flags_bytes);

    // Read country field (4 bytes)
    let mut country_bytes = [0u8; 4];
    reader
        .read_exact(&mut country_bytes)
        .context("read country field")?;
    let country = u32::from_be_bytes(country_bytes);

    // Calculate remaining bytes after reading size + "data" + flags + country = 16 bytes
    let remaining_bytes = data_size - 16;

    // Read all remaining data
    let mut remaining_data = vec![0u8; remaining_bytes];
    reader
        .read_exact(&mut remaining_data)
        .context("read remaining data")?;

    // Try to determine if there's a language field by checking if the first 4 bytes look like language data
    // If the first 4 bytes are all zeros or a small value that could be a language code,
    // and there's more data after, treat them as language field
    let (language, data_bytes) = if remaining_bytes >= 4 {
        let potential_language = u32::from_be_bytes([
            remaining_data[0],
            remaining_data[1],
            remaining_data[2],
            remaining_data[3],
        ]);

        // If the first 4 bytes are zero or a small value that could be a language code,
        // and there's more data after, treat it as language field
        if potential_language == 0 || (potential_language < 0x10000 && remaining_bytes > 4) {
            (potential_language, remaining_data[4..].to_vec())
        } else {
            // No language field, all remaining data is actual data
            (0, remaining_data)
        }
    } else {
        (0, remaining_data)
    };

    let metadata = ItemMetadata {
        type_flags,
        country,
        language,
    };

    // Parse value based on type flags
    let value = match type_flags {
        1 => {
            // UTF-8 text - remove null terminator if present
            let text_data = if !data_bytes.is_empty() && data_bytes[data_bytes.len() - 1] == 0 {
                &data_bytes[..data_bytes.len() - 1]
            } else {
                &data_bytes
            };
            MetadataValue::Text(String::from_utf8_lossy(text_data).to_string())
        }
        13 => {
            // JPEG image data
            MetadataValue::Jpeg(data_bytes)
        }
        14 => {
            // PNG image data
            MetadataValue::Png(data_bytes)
        }
        21 => {
            // Unsigned 8-bit integer
            if !data_bytes.is_empty() {
                MetadataValue::UInt8(data_bytes[0])
            } else {
                MetadataValue::Binary(data_bytes)
            }
        }
        22 => {
            // Unsigned 16-bit integer
            if data_bytes.len() >= 2 {
                MetadataValue::UInt16(u16::from_be_bytes([data_bytes[0], data_bytes[1]]))
            } else {
                MetadataValue::Binary(data_bytes)
            }
        }
        23 => {
            // Unsigned 32-bit integer
            if data_bytes.len() >= 4 {
                MetadataValue::UInt32(u32::from_be_bytes([
                    data_bytes[0],
                    data_bytes[1],
                    data_bytes[2],
                    data_bytes[3],
                ]))
            } else {
                MetadataValue::Binary(data_bytes)
            }
        }
        24 => {
            // Unsigned 64-bit integer
            if data_bytes.len() >= 8 {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&data_bytes[0..8]);
                MetadataValue::UInt64(u64::from_be_bytes(bytes))
            } else {
                MetadataValue::Binary(data_bytes)
            }
        }
        _ => {
            // Unknown type, store as binary
            MetadataValue::Binary(data_bytes)
        }
    };

    Ok((value, metadata))
}

/// Convert Mac Roman encoded bytes to UTF-8 string
/// MP4 metadata item types often use Mac Roman encoding where 0xA9 = ©
fn convert_mac_roman_to_utf8(bytes: &[u8]) -> String {
    let mut result = String::new();
    for &byte in bytes {
        match byte {
            0xA9 => result.push('©'), // Copyright symbol
            0xAE => result.push('®'), // Registered trademark symbol
            0x99 => result.push('™'), // Trademark symbol
            // For other bytes, treat as ASCII if valid, otherwise use replacement char
            b if b.is_ascii() => result.push(b as char),
            _ => result.push('�'),
        }
    }
    result
}

fn convert_utf8_to_mac_roman(text: &str) -> Vec<u8> {
    let mut result = Vec::new();
    for ch in text.chars() {
        match ch {
            '©' => result.push(0xA9), // Copyright symbol
            '®' => result.push(0xAE), // Registered trademark symbol
            '™' => result.push(0x99), // Trademark symbol
            // For ASCII characters, use their byte value
            c if c.is_ascii() => result.push(c as u8),
            // For non-ASCII characters that don't have Mac Roman equivalents, skip them
            _ => {}
        }
    }
    result
}

fn parse_raw_metadata_item<R: Read>(
    reader: &mut R,
    item_size: usize,
) -> Result<RawMetadataItem, anyhow::Error> {
    if item_size < 8 {
        return Err(anyhow!("Raw item size too small: {}", item_size));
    }

    // Read item type (4 bytes)
    let mut type_bytes = [0u8; 4];
    reader
        .read_exact(&mut type_bytes)
        .context("read raw item type")?;

    // Read remaining data
    let data_size = item_size - 4; // Subtract type field (size already excluded from slice)
    let mut data = vec![0u8; data_size];
    reader.read_exact(&mut data).context("read raw item data")?;

    Ok(RawMetadataItem {
        item_type: type_bytes,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_ilst_data() -> Vec<u8> {
        let mut data = Vec::new();

        // Create a simple text metadata item (©nam - title)
        let item_type = b"\xa9nam"; // ©nam in UTF-8
        let text_data = b"Test Title";

        // Calculate sizes
        // The data atom size field contains the total size of the data atom (including the size field itself)
        let data_atom_size = 4 + 4 + 4 + 4 + 4 + text_data.len(); // size + type + flags + country + language + data
        let item_size = 8 + data_atom_size; // 8 bytes item header + data atom

        // Item header
        data.extend_from_slice(&(item_size as u32).to_be_bytes());
        data.extend_from_slice(item_type);

        // Data atom
        data.extend_from_slice(&(data_atom_size as u32).to_be_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&1u32.to_be_bytes()); // Type flags (1 = UTF-8 text)
        data.extend_from_slice(&0u32.to_be_bytes()); // Country
        data.extend_from_slice(&0u32.to_be_bytes()); // Language
        data.extend_from_slice(text_data);

        data
    }

    fn get_expected_item_type() -> String {
        // Convert the item type bytes using proper Mac Roman to UTF-8 conversion
        convert_mac_roman_to_utf8(b"\xa9nam")
    }

    #[test]
    fn test_parse_ilst_data() {
        let data = create_test_ilst_data();
        let result = parse_ilst_data(Cursor::new(&data)).unwrap();

        assert_eq!(result.items.len(), 1);
        let expected_key = get_expected_item_type();
        assert!(result
            .items
            .iter()
            .any(|item| item.item_type == expected_key));

        let title_item = result.items.first().unwrap();
        match &title_item.value {
            MetadataValue::Text(text) => assert_eq!(text, "Test Title"),
            _ => panic!("Expected text value"),
        }
    }

    #[test]
    fn test_binary_metadata_debug() {
        // Test that binary data shows only length in debug output
        let binary_data = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let binary_item = MetadataItem {
            item_type: "covr".to_string(),
            value: MetadataValue::Binary(binary_data.clone()),
            metadata: ItemMetadata {
                type_flags: 13, // JPEG image data
                country: 0,
                language: 0,
            },
        };

        let debug_output = format!("{:?}", binary_item);

        // Should contain the length format, not the raw bytes
        assert!(debug_output.contains("<8 bytes>"));

        // Should NOT contain the actual byte values
        assert!(!debug_output.contains("[1, 2, 3, 4, 5, 6, 7, 8]"));
        assert!(!debug_output.contains("0x01"));

        // Test empty binary data
        let empty_binary_item = MetadataItem {
            item_type: "test".to_string(),
            value: MetadataValue::Binary(vec![]),
            metadata: ItemMetadata {
                type_flags: 13,
                country: 0,
                language: 0,
            },
        };

        let empty_debug = format!("{:?}", empty_binary_item);
        assert!(empty_debug.contains("<0 bytes>"));

        // Test that text values still show normally
        let text_item = MetadataItem {
            item_type: "©nam".to_string(),
            value: MetadataValue::Text("Test Title".to_string()),
            metadata: ItemMetadata {
                type_flags: 1,
                country: 0,
                language: 0,
            },
        };

        let text_debug = format!("{:?}", text_item);
        assert!(text_debug.contains("Test Title"));
        assert!(!text_debug.contains("bytes"));
    }

    #[test]
    fn test_malformed_metadata_parsing() {
        // Create test data that might reproduce the malformed parsing issue
        // This simulates a real-world case where we might see "�pub" and "ble Studios"
        let mut data = Vec::new();

        // Create a publisher metadata item that might get corrupted
        let item_type = b"\xa9pub"; // ©pub publisher field
        let publisher_text = b"Audible Studios";

        // Calculate sizes - this might be where the issue occurs
        let data_atom_size = 4 + 4 + 4 + 4 + 4 + publisher_text.len(); // size + type + flags + country + language + data
        let item_size = 8 + data_atom_size; // 8 bytes item header + data atom

        // Item header
        data.extend_from_slice(&(item_size as u32).to_be_bytes());
        data.extend_from_slice(item_type);

        // Data atom
        data.extend_from_slice(&(data_atom_size as u32).to_be_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&1u32.to_be_bytes()); // Type flags (1 = UTF-8 text)
        data.extend_from_slice(&0u32.to_be_bytes()); // Country
        data.extend_from_slice(&0u32.to_be_bytes()); // Language
        data.extend_from_slice(publisher_text);

        // Parse the data
        let result = parse_ilst_data(Cursor::new(&data));

        match result {
            Ok(ilst) => {
                // Check if we get the correct item type and text
                let pub_key = convert_mac_roman_to_utf8(b"\xa9pub");
                if let Some(publisher_item) =
                    ilst.items.iter().find(|item| item.item_type == pub_key)
                {
                    match &publisher_item.value {
                        MetadataValue::Text(text) => {
                            assert_eq!(text, "Audible Studios", "Text should not be truncated");
                            assert_eq!(
                                publisher_item.item_type, pub_key,
                                "Item type should not be corrupted"
                            );
                        }
                        _ => panic!("Expected text value for publisher"),
                    }
                } else {
                    panic!("Publisher item not found or corrupted");
                }
            }
            Err(_) => {
                // If parsing fails, we want to understand why
                panic!("Failed to parse metadata that should be valid");
            }
        }
    }

    #[test]
    fn test_edge_case_data_size_calculations() {
        // Test different ways data atom size might be calculated in real-world files
        let mut data = Vec::new();
        let item_type = b"\xa9pub";
        let text_data = b"Audible Studios";

        // Test 1: Size field excludes itself (some encoders might do this)
        let data_atom_size_excluding_self = 4 + 4 + 4 + 4 + text_data.len(); // type + flags + country + language + data
        let item_size = 8 + 4 + data_atom_size_excluding_self; // 8 bytes item header + 4 bytes data size + data

        data.extend_from_slice(&(item_size as u32).to_be_bytes());
        data.extend_from_slice(item_type);
        data.extend_from_slice(&(data_atom_size_excluding_self as u32).to_be_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(text_data);

        // This should fail gracefully or be handled by raw item parsing
        let result = parse_ilst_data(Cursor::new(&data));
        match result {
            Ok(ilst) => {
                // If it parses, check that we don't get corrupted data
                if !ilst.items.is_empty() {
                    for item in ilst.items.iter() {
                        if let MetadataValue::Text(text) = &item.value {
                            assert!(
                                !text.contains("�"),
                                "Text should not contain corruption artifacts"
                            );
                            assert!(text.len() > 0, "Text should not be empty");
                        }
                    }
                }
            }
            Err(_) => {
                // Failure is acceptable for malformed data
            }
        }
    }

    #[test]
    fn test_multiple_data_atoms_in_item() {
        // Test an item that contains multiple data atoms (possible in some files)
        let mut data = Vec::new();
        let item_type = b"\xa9pub";
        let text_data1 = b"Audible";
        let text_data2 = b" Studios";

        // First data atom
        let data_atom_size1 = 4 + 4 + 4 + 4 + 4 + text_data1.len();
        // Second data atom
        let data_atom_size2 = 4 + 4 + 4 + 4 + 4 + text_data2.len();

        let item_size = 8 + data_atom_size1 + data_atom_size2;

        // Item header
        data.extend_from_slice(&(item_size as u32).to_be_bytes());
        data.extend_from_slice(item_type);

        // First data atom
        data.extend_from_slice(&(data_atom_size1 as u32).to_be_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(text_data1);

        // Second data atom
        data.extend_from_slice(&(data_atom_size2 as u32).to_be_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(text_data2);

        // This might not parse correctly since we only read the first data atom
        let result = parse_ilst_data(Cursor::new(&data));
        match result {
            Ok(ilst) => {
                // We should get something, but it might only be the first data atom
                let pub_key = String::from_utf8_lossy(item_type).to_string();
                if let Some(item) = ilst.items.iter().find(|item| item.item_type == pub_key) {
                    if let MetadataValue::Text(text) = &item.value {
                        // Should be "Audible", not corrupted
                        assert!(!text.is_empty());
                        assert!(!text.contains("�"));
                    }
                }
            }
            Err(_) => {
                // Acceptable if this format isn't supported
            }
        }
    }

    #[test]
    fn test_padded_data_atoms() {
        // Test data atoms with padding bytes (some encoders add alignment padding)
        let mut data = Vec::new();
        let item_type = b"\xa9pub";
        let text_data = b"Audible Studios";

        // Add padding to align to 4-byte boundary
        let mut padded_text = text_data.to_vec();
        while padded_text.len() % 4 != 0 {
            padded_text.push(0);
        }

        let data_atom_size = 4 + 4 + 4 + 4 + 4 + padded_text.len();
        let item_size = 8 + data_atom_size;

        data.extend_from_slice(&(item_size as u32).to_be_bytes());
        data.extend_from_slice(item_type);
        data.extend_from_slice(&(data_atom_size as u32).to_be_bytes());
        data.extend_from_slice(b"data");
        data.extend_from_slice(&1u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&padded_text);

        let result = parse_ilst_data(Cursor::new(&data));
        match result {
            Ok(ilst) => {
                let pub_key = String::from_utf8_lossy(item_type).to_string();
                if let Some(item) = ilst.items.iter().find(|item| item.item_type == pub_key) {
                    if let MetadataValue::Text(text) = &item.value {
                        // Text might have null padding, but should start correctly
                        assert!(
                            text.starts_with("Audible Studios")
                                || text.trim_end_matches('\0') == "Audible Studios"
                        );
                        assert!(!text.starts_with("�"));
                    }
                }
            }
            Err(_) => {
                // May fail if padding causes size mismatch
            }
        }
    }

    #[test]
    fn test_hexdump_data_parsing() {
        // First few items from the provided hexdump
        let hexdump_data = [
            // First item: ©nam "Opening Credits" (39 bytes total)
            0x00, 0x00, 0x00, 0x27, 0xa9, 0x6e, 0x61, 0x6d, // size + ©nam
            0x00, 0x00, 0x00, 0x1f, 0x64, 0x61, 0x74, 0x61, // data atom size + "data"
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, // flags + country
            0x4f, 0x70, 0x65, 0x6e, 0x69, 0x6e, 0x67, 0x20, // "Opening "
            0x43, 0x72, 0x65, 0x64, 0x69, 0x74, 0x73, 0x00, // "Credits\0"
            // Second item: ©cmt "Opening Credits" (39 bytes total)
            0x00, 0x00, 0x00, 0x27, 0xa9, 0x63, 0x6d, 0x74, // size + ©cmt
            0x00, 0x00, 0x00, 0x1f, 0x64, 0x61, 0x74, 0x61, // data atom size + "data"
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, // flags + country
            0x4f, 0x70, 0x65, 0x6e, 0x69, 0x6e, 0x67, 0x20, // "Opening "
            0x43, 0x72, 0x65, 0x64, 0x69, 0x74, 0x73, 0x00, // "Credits\0"
        ];

        let result = parse_ilst_data(std::io::Cursor::new(&hexdump_data));

        match result {
            Ok(ilst) => {
                // Should have parsed at least some items (regular or raw)
                assert!(
                    ilst.items.len() >= 1 || ilst.raw_items.len() >= 1,
                    "Should have parsed at least 1 item (regular or raw)"
                );
            }
            Err(e) => {
                panic!("Failed to parse hexdump data: {}", e);
            }
        }
    }

    #[test]
    fn test_mac_roman_round_trip() {
        // Test round-trip encoding/decoding of Mac Roman characters in item types
        let original_ilst = ItemListAtom {
            items: MetadataItems(vec![
                MetadataItem {
                    item_type: "©nam".to_string(), // Mac Roman copyright symbol
                    value: MetadataValue::Text("Test Song".to_string()),
                    metadata: ItemMetadata {
                        type_flags: 1,
                        country: 0,
                        language: 0,
                    },
                },
                MetadataItem {
                    item_type: "®pub".to_string(), // Mac Roman registered trademark symbol
                    value: MetadataValue::Text("Test Publisher".to_string()),
                    metadata: ItemMetadata {
                        type_flags: 1,
                        country: 0,
                        language: 0,
                    },
                },
            ]),
            raw_items: RawMetadataItems(vec![]),
        };

        // Convert to bytes
        let encoded_bytes: Vec<u8> = original_ilst.into();

        // Parse back from bytes
        let cursor = std::io::Cursor::new(&encoded_bytes);
        let parsed_ilst = parse_ilst_data(cursor).expect("Failed to parse encoded data");

        // Verify round-trip worked correctly
        assert_eq!(
            parsed_ilst.items.len(),
            2,
            "Should have 2 items after round-trip"
        );

        // Check first item (©nam)
        let nam_item = &parsed_ilst.items[0];
        assert_eq!(nam_item.item_type, "©nam", "First item type should be ©nam");
        match &nam_item.value {
            MetadataValue::Text(text) => assert_eq!(text, "Test Song"),
            _ => panic!("Expected text value for ©nam item"),
        }

        // Check second item (®pub)
        let pub_item = &parsed_ilst.items[1];
        assert_eq!(
            pub_item.item_type, "®pub",
            "Second item type should be ®pub"
        );
        match &pub_item.value {
            MetadataValue::Text(text) => assert_eq!(text, "Test Publisher"),
            _ => panic!("Expected text value for ®pub item"),
        }
    }
}
