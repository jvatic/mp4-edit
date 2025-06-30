use std::io::Read;

#[derive(Debug, Clone, PartialEq)]
pub struct MetaHeader {
    pub version: u8,
    pub flags: [u8; 3],
}

impl MetaHeader {
    /// Parse the META atom's 4-byte header from a byte slice
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ParseError> {
        if bytes.len() < 4 {
            return Err(ParseError::InsufficientData);
        }

        Ok(MetaHeader {
            version: bytes[0],
            flags: [bytes[1], bytes[2], bytes[3]],
        })
    }

    /// Parse the META atom's 4-byte header from a reader
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self, ParseError> {
        let mut header_bytes = [0u8; 4];
        reader
            .read_exact(&mut header_bytes)
            .map_err(|_| ParseError::ReadError)?;

        Ok(MetaHeader {
            version: header_bytes[0],
            flags: [header_bytes[1], header_bytes[2], header_bytes[3]],
        })
    }

    /// Get flags as a single u32 value (big-endian)
    pub fn flags_as_u32(&self) -> u32 {
        u32::from_be_bytes([0, self.flags[0], self.flags[1], self.flags[2]])
    }

    /// Check if this is a valid META header (version should be 0 for Apple compatibility)
    pub fn is_valid_for_apple(&self) -> bool {
        self.version == 0 && self.flags == [0, 0, 0]
    }

    /// Convert back to 4 bytes
    pub fn to_bytes(&self) -> [u8; 4] {
        [self.version, self.flags[0], self.flags[1], self.flags[2]]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    InsufficientData,
    ReadError,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::InsufficientData => write!(f, "Insufficient data to parse META header"),
            ParseError::ReadError => write!(f, "Failed to read META header data"),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_parse_valid_apple_meta_header() {
        let bytes = [0x00, 0x00, 0x00, 0x00]; // Version 0, flags all 0
        let header = MetaHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.version, 0);
        assert_eq!(header.flags, [0, 0, 0]);
        assert!(header.is_valid_for_apple());
        assert_eq!(header.flags_as_u32(), 0);
    }

    #[test]
    fn test_parse_non_apple_meta_header() {
        let bytes = [0x01, 0x00, 0x00, 0x01]; // Version 1, some flags set
        let header = MetaHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.version, 1);
        assert_eq!(header.flags, [0, 0, 1]);
        assert!(!header.is_valid_for_apple());
        assert_eq!(header.flags_as_u32(), 1);
    }

    #[test]
    fn test_parse_from_reader() {
        let data = [0x00, 0x00, 0x00, 0x00];
        let mut cursor = Cursor::new(&data);
        let header = MetaHeader::from_reader(&mut cursor).unwrap();

        assert_eq!(header.version, 0);
        assert_eq!(header.flags, [0, 0, 0]);
    }

    #[test]
    fn test_insufficient_data() {
        let bytes = [0x00, 0x00]; // Only 2 bytes
        let result = MetaHeader::from_bytes(&bytes);

        assert!(matches!(result, Err(ParseError::InsufficientData)));
    }

    #[test]
    fn test_round_trip() {
        let original = MetaHeader {
            version: 0,
            flags: [0x12, 0x34, 0x56],
        };

        let bytes = original.to_bytes();
        let parsed = MetaHeader::from_bytes(&bytes).unwrap();

        assert_eq!(original, parsed);
    }

    #[test]
    fn test_flags_as_u32() {
        let header = MetaHeader {
            version: 0,
            flags: [0x01, 0x02, 0x03],
        };

        assert_eq!(header.flags_as_u32(), 0x010203);
    }
}
