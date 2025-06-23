use std::io::{Cursor, Read};

use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct BtrtExtension {
    /// Buffer size in bytes
    pub buffer_size_db: u32,
    /// Maximum bitrate in bits per second
    pub max_bitrate: u32,
    /// Average bitrate in bits per second
    pub avg_bitrate: u32,
}

#[derive(Debug, Error)]
pub enum BtrtParseError {
    #[error("Invalid data length: expected 12 bytes, got {0}")]
    InvalidLength(usize),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl BtrtExtension {
    /// Parse btrt extension from raw bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, BtrtParseError> {
        if data.len() != 12 {
            return Err(BtrtParseError::InvalidLength(data.len()));
        }

        let mut cursor = Cursor::new(data);
        let mut buf = [0u8; 4];

        // Read buffer size (big-endian u32)
        cursor.read_exact(&mut buf)?;
        let buffer_size_db = u32::from_be_bytes(buf);

        // Read max bitrate (big-endian u32)
        cursor.read_exact(&mut buf)?;
        let max_bitrate = u32::from_be_bytes(buf);

        // Read avg bitrate (big-endian u32)
        cursor.read_exact(&mut buf)?;
        let avg_bitrate = u32::from_be_bytes(buf);

        Ok(BtrtExtension {
            buffer_size_db,
            max_bitrate,
            avg_bitrate,
        })
    }
}

impl From<BtrtExtension> for Vec<u8> {
    fn from(btrt: BtrtExtension) -> Self {
        let mut result = Vec::with_capacity(12);

        // Write buffer size (big-endian u32)
        result.extend_from_slice(&btrt.buffer_size_db.to_be_bytes());

        // Write max bitrate (big-endian u32)
        result.extend_from_slice(&btrt.max_bitrate.to_be_bytes());

        // Write avg bitrate (big-endian u32)
        result.extend_from_slice(&btrt.avg_bitrate.to_be_bytes());

        result
    }
}
