use anyhow::anyhow;
use anyhow::{bail, Result};
use core::fmt;
use derive_more::Deref;
use futures_io::AsyncRead;
use std::io::Read;

use crate::atom::util::DebugEllipsis;
use crate::{
    atom::{util::async_to_sync_read, FourCC},
    parser::Parse,
};

pub const ILST: &[u8; 4] = b"ilst";

#[derive(Clone, Deref)]
pub struct RawData(Vec<u8>);

impl fmt::Debug for RawData {
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

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ListItemData {
    Text(String),
    Jpeg(RawData),
    Raw(RawData),
}

impl ListItemData {
    fn new(data_type: u32, data: Vec<u8>) -> Self {
        match data_type {
            1 => String::from_utf8(data)
                .map(Self::Text)
                .unwrap_or_else(|e| Self::Raw(RawData(e.into_bytes()))),
            13 => Self::Jpeg(RawData(data)),
            _ => Self::Raw(RawData(data)),
        }
    }
}

impl From<ListItemData> for Vec<u8> {
    fn from(value: ListItemData) -> Self {
        use ListItemData::*;
        match value {
            Text(s) => s.into_bytes(),
            Jpeg(data) => data.0,
            Raw(data) => data.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DataAtom {
    pub data_type: u32,
    pub reserved: u32,
    pub data: ListItemData,
}

#[derive(Debug, Clone)]
pub struct MetadataItem {
    pub item_type: FourCC,
    pub data_atoms: Vec<DataAtom>,
}

#[derive(Debug, Clone)]
pub struct ItemListAtom {
    pub items: Vec<MetadataItem>,
}

impl Parse for ItemListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(atom_type: FourCC, reader: R) -> Result<Self> {
        if atom_type != ILST {
            return Err(anyhow!("Invalid atom type: {}", atom_type));
        }
        parse_ilst_data(async_to_sync_read(reader).await?)
    }
}

impl From<ItemListAtom> for Vec<u8> {
    fn from(atom: ItemListAtom) -> Self {
        let mut output = Vec::new();

        for item in atom.items {
            // Calculate item data
            let mut item_data = Vec::new();

            for data_atom in item.data_atoms {
                let data: Vec<u8> = data_atom.data.into();
                let data_size = 16 + data.len() as u32; // header + type flags + reserved + data

                // Write data atom
                item_data.extend_from_slice(&data_size.to_be_bytes());
                item_data.extend_from_slice(b"data");
                item_data.extend_from_slice(&data_atom.data_type.to_be_bytes());
                item_data.extend_from_slice(&data_atom.reserved.to_be_bytes());
                item_data.extend_from_slice(&data);
            }

            let item_size = 8 + item_data.len() as u32; // size + type + data

            // Write item
            output.extend_from_slice(&item_size.to_be_bytes());
            output.extend_from_slice(&item.item_type.into_bytes());
            output.extend_from_slice(&item_data);
        }

        output
    }
}

fn parse_ilst_data<R: std::io::Read>(mut reader: R) -> Result<ItemListAtom> {
    let mut items = Vec::new();

    loop {
        // Try to read size
        let size = match read_u32(&mut reader) {
            Ok(s) => s,
            Err(_) => break, // End of stream
        };

        if size == 0 {
            break;
        }

        let actual_size = if size == 1 {
            read_u64(&mut reader)?
        } else {
            size as u64
        };

        let item_type = read_fourcc(&mut reader)?;

        // Calculate remaining bytes for this item
        let header_size = if size == 1 { 16 } else { 8 }; // size + type + optional extended size
        let remaining_size = actual_size - header_size;

        // Read the item data
        let item_data = read_bytes(&mut reader, remaining_size as usize)?;
        let mut item_reader = std::io::Cursor::new(&item_data);

        // Parse data atoms within this item
        let mut data_atoms = Vec::new();

        while item_reader.position() < item_data.len() as u64 {
            let data_size = match read_u32(&mut item_reader) {
                Ok(s) => s,
                Err(_) => break,
            };

            if data_size == 0 {
                break;
            }

            let data_actual_size = if data_size == 1 {
                read_u64(&mut item_reader)?
            } else {
                data_size as u64
            };

            let atom_type = read_fourcc(&mut item_reader)?;
            if atom_type != b"data" {
                bail!("Expected 'data' atom, got '{}'", atom_type);
            }

            let remaining_data_size = data_actual_size - if data_size == 1 { 16 } else { 8 };
            let data_atom = parse_data_atom(&mut item_reader, remaining_data_size)?;
            data_atoms.push(data_atom);
        }

        items.push(MetadataItem {
            item_type,
            data_atoms,
        });
    }

    Ok(ItemListAtom { items })
}

fn parse_data_atom<R: Read>(reader: &mut R, size: u64) -> Result<DataAtom> {
    if size < 8 {
        bail!("Data atom too small");
    }

    let data_type = read_u32(reader)?;
    let reserved = read_u32(reader)?;
    let data_size = size - 8; // size - data_type - reserved
    let data = read_bytes(reader, data_size as usize)?;

    Ok(DataAtom {
        data_type,
        reserved,
        data: ListItemData::new(data_type, data),
    })
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

fn read_fourcc<R: Read>(reader: &mut R) -> Result<FourCC> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(FourCC::from(buf))
}

fn read_bytes<R: Read>(reader: &mut R, count: usize) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; count];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Test round-trip for binary data
    #[test]
    fn test_binary_metadata_roundtrip() {
        const BINARY_DATA: &[u8] = include_bytes!("../../test-data/ilst.bin");

        // exclude the ilst atom size and fourcc
        let ilst_data = &BINARY_DATA[8..];

        let decoded =
            parse_ilst_data(Cursor::new(ilst_data)).expect("failed to parse encoded data");
        assert!(!decoded.items.is_empty());
        let re_encoded: Vec<u8> = decoded.into();

        // check each chunk for equality to make any variations easier to debug
        const CHUNK_SIZE: usize = 200;
        for ((i, left), right) in re_encoded
            .chunks(CHUNK_SIZE)
            .enumerate()
            .zip(ilst_data.chunks(CHUNK_SIZE))
        {
            assert_eq!(
                left,
                right,
                "round-trip failed for binary metadata at range [{}..{}] (left.len()={}, right.len()={})",
                i * CHUNK_SIZE,
                ((i + 1) * CHUNK_SIZE).min(ilst_data.len()),
                re_encoded.len(),
                ilst_data.len(),
            );
        }
    }
}
