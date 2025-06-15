use derive_more::Display;
use futures_io::AsyncWrite;
use futures_util::AsyncWriteExt;
use thiserror::Error;

use crate::atom::FourCC;

#[derive(Debug, Error)]
#[error("{kind}{}", self.source.as_ref().map(|e| format!(" ({e})")).unwrap_or(String::new()))]
pub struct WriteError {
    /// The kind of error that occurred during writing.
    kind: WriteErrorKind,
    /// The source error that caused this error.
    #[source]
    source: Option<Box<dyn std::error::Error + Send + Sync>>,
}

#[derive(Debug, Display)]
pub enum WriteErrorKind {
    #[display("I/O error")]
    Io,
}

pub struct Mp4Writer {}

impl Mp4Writer {
    pub async fn write_atom<W: AsyncWrite + Unpin>(
        mut writer: W,
        atom_type: FourCC,
        data: impl Into<Vec<u8>>,
    ) -> Result<(), WriteError> {
        // Serialize the atom data first to know its size
        let data_bytes = data.into();

        let header_size = 8u64; // 4 bytes size + 4 bytes type
        let total_size = header_size + data_bytes.len() as u64;

        // Check if we need extended 64-bit size
        if total_size > u32::MAX as u64 {
            // Extended size format: size=1, type, extended_size, data
            writer
                .write_all(&1u32.to_be_bytes())
                .await
                .map_err(|e| WriteError {
                    kind: WriteErrorKind::Io,
                    source: Some(Box::new(e)),
                })?;

            writer
                .write_all(&atom_type.0)
                .await
                .map_err(|e| WriteError {
                    kind: WriteErrorKind::Io,
                    source: Some(Box::new(e)),
                })?;

            let extended_total_size = 16u64 + data_bytes.len() as u64; // 16 bytes header + data
            writer
                .write_all(&extended_total_size.to_be_bytes())
                .await
                .map_err(|e| WriteError {
                    kind: WriteErrorKind::Io,
                    source: Some(Box::new(e)),
                })?;
        } else {
            // Standard 32-bit size format
            writer
                .write_all(&(total_size as u32).to_be_bytes())
                .await
                .map_err(|e| WriteError {
                    kind: WriteErrorKind::Io,
                    source: Some(Box::new(e)),
                })?;

            writer
                .write_all(&atom_type.0)
                .await
                .map_err(|e| WriteError {
                    kind: WriteErrorKind::Io,
                    source: Some(Box::new(e)),
                })?;
        }

        // Write the atom data
        if !data_bytes.is_empty() {
            writer
                .write_all(&data_bytes)
                .await
                .map_err(|e| WriteError {
                    kind: WriteErrorKind::Io,
                    source: Some(Box::new(e)),
                })?;
        }

        writer.flush().await.map_err(|e| WriteError {
            kind: WriteErrorKind::Io,
            source: Some(Box::new(e)),
        })?;

        Ok(())
    }
}
