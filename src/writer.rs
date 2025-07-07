use std::future::Future;

use derive_more::Display;
use futures_io::AsyncWrite;
use futures_util::AsyncWriteExt;
use thiserror::Error;

use crate::{atom::FourCC, Atom, AtomData};

#[derive(Debug, Error)]
#[error("{kind}{}", self.source.as_ref().map(|e| format!(" ({e})")).unwrap_or_default())]
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

pub trait WriteAtom: Sized {
    /// Write full atom (and any children) to the provided writer
    fn write_atom<W: AsyncWrite + Unpin + Send>(
        &self,
        writer: W,
    ) -> impl Future<Output = Result<(), anyhow::Error>> + Send;
}

pub struct Mp4Writer<W> {
    writer: W,
    offset: usize,
}

impl<W: AsyncWrite + Unpin> Mp4Writer<W> {
    pub fn new(writer: W) -> Self {
        Self { writer, offset: 0 }
    }

    pub fn current_offset(&self) -> usize {
        self.offset
    }

    pub async fn flush(&mut self) -> Result<(), WriteError> {
        self.writer.flush().await.map_err(|e| WriteError {
            kind: WriteErrorKind::Io,
            source: Some(Box::new(e)),
        })
    }

    pub async fn write_atom_header(
        &mut self,
        atom_type: FourCC,
        data_size: usize,
    ) -> Result<(), WriteError> {
        // Write atom header
        let header_bytes = serialize_atom_header(atom_type, data_size as u64);
        self.writer
            .write_all(&header_bytes)
            .await
            .map_err(|e| WriteError {
                kind: WriteErrorKind::Io,
                source: Some(Box::new(e)),
            })?;
        self.offset += header_bytes.len();
        Ok(())
    }

    pub async fn write_leaf_atom(
        &mut self,
        atom_type: FourCC,
        data: AtomData,
    ) -> Result<(), WriteError> {
        let data_bytes: Vec<u8> = data.into();
        self.write_atom_header(atom_type, data_bytes.len()).await?;
        self.writer
            .write_all(&data_bytes)
            .await
            .map_err(|e| WriteError {
                kind: WriteErrorKind::Io,
                source: Some(Box::new(e)),
            })?;
        self.offset += data_bytes.len();
        Ok(())
    }

    pub async fn write_atom(&mut self, atom: Atom) -> Result<(), WriteError> {
        // Serialize the entire atom tree into bytes
        let bytes = serialize_atom(&atom);

        // Write all bytes at once
        self.writer
            .write_all(&bytes)
            .await
            .map_err(|e| WriteError {
                kind: WriteErrorKind::Io,
                source: Some(Box::new(e)),
            })?;

        self.offset += bytes.len();

        Ok(())
    }

    pub async fn write_raw(&mut self, data: &[u8]) -> Result<(), WriteError> {
        self.writer.write_all(data).await.map_err(|e| WriteError {
            kind: WriteErrorKind::Io,
            source: Some(Box::new(e)),
        })?;

        self.offset += data.len();
        Ok(())
    }
}

fn serialize_atom_header(atom_type: FourCC, data_size: u64) -> Vec<u8> {
    let mut result = Vec::new();

    // Determine if we need 64-bit size
    let total_size_with_32bit_header = 8u64 + data_size;
    let use_64bit = total_size_with_32bit_header > u32::MAX as u64;

    if use_64bit {
        // Extended 64-bit size format: size=1 (4 bytes) + type (4 bytes) + extended_size (8 bytes) + content
        let total_size = 16u64 + data_size;

        // Write size=1 to indicate extended format
        result.extend_from_slice(&1u32.to_be_bytes());

        // Write atom type
        result.extend_from_slice(&atom_type.0);

        // Write extended size
        result.extend_from_slice(&total_size.to_be_bytes());
    } else {
        // Standard 32-bit size format: size (4 bytes) + type (4 bytes) + content
        let total_size = total_size_with_32bit_header as u32;

        // Write size
        result.extend_from_slice(&total_size.to_be_bytes());

        // Write atom type
        result.extend_from_slice(&atom_type.0);
    }

    result
}

/// Serialize an atom and all its children into bytes
pub fn serialize_atom(atom: &Atom) -> Vec<u8> {
    let mut result = Vec::new();

    // Get atom data bytes
    let data_bytes = if let Some(data) = &atom.data {
        let bytes: Vec<u8> = data.clone().into();
        bytes
    } else {
        Vec::new()
    };

    // Serialize all children
    let mut children_bytes = Vec::new();
    for child in &atom.children {
        let child_bytes = serialize_atom(child);
        children_bytes.extend(child_bytes);
    }

    // Calculate total content size (data + children)
    let data_size = data_bytes.len() as u64 + children_bytes.len() as u64;

    // Write atom header
    result.extend(serialize_atom_header(atom.header.atom_type, data_size));

    // Write atom data
    result.extend_from_slice(&data_bytes);

    // Write children
    result.extend_from_slice(&children_bytes);

    result
}
