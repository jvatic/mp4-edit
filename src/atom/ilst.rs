use anyhow::bail;
use bon::bon;
use core::fmt;
use derive_more::Deref;
use futures_io::AsyncRead;
use std::io::Read;

use crate::atom::util::DebugEllipsis;
use crate::ParseError;
use crate::{
    atom::{util::async_to_sync_read, FourCC},
    parser::ParseAtom,
    writer::SerializeAtom,
};

pub const ILST: &[u8; 4] = b"ilst";

const DATA_TYPE_TEXT: u32 = 1;
const DATA_TYPE_JPEG: u32 = 13;

#[derive(Clone, Deref)]
pub struct RawData(Vec<u8>);

impl RawData {
    pub fn new(data: impl Into<Vec<u8>>) -> Self {
        RawData(data.into())
    }
}

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
            DATA_TYPE_TEXT => String::from_utf8(data)
                .map_or_else(|e| Self::Raw(RawData(e.into_bytes())), Self::Text),
            DATA_TYPE_JPEG => Self::Jpeg(RawData(data)),
            _ => Self::Raw(RawData(data)),
        }
    }

    fn to_bytes(self: ListItemData) -> Vec<u8> {
        use ListItemData::{Jpeg, Raw, Text};
        match self {
            Text(s) => s.into_bytes(),
            Jpeg(data) | Raw(data) => data.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DataAtom {
    pub data_type: u32,
    pub reserved: u32,
    pub data: ListItemData,
}

impl DataAtom {
    pub fn new(data: ListItemData) -> Self {
        Self {
            data_type: 0,
            reserved: 0,
            data,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetadataItem {
    pub item_type: FourCC,
    pub mean: Option<Vec<u8>>,
    pub name: Option<Vec<u8>>,
    pub data_atoms: Vec<DataAtom>,
}

#[bon]
impl MetadataItem {
    #[builder]
    pub fn new(
        #[builder(into, start_fn)] item_type: FourCC,
        #[builder(into)] data_atoms: Vec<DataAtom>,
    ) -> Self {
        Self {
            item_type,
            mean: None,
            name: None,
            data_atoms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ItemListAtom {
    pub items: Vec<MetadataItem>,
}

impl ParseAtom for ItemListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != ILST {
            return Err(ParseError::new_unexpected_atom(atom_type, ILST));
        }
        parse_ilst_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
}

impl SerializeAtom for ItemListAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*ILST)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut output = Vec::new();

        for item in self.items {
            // Calculate item data
            let mut item_data = Vec::new();

            if let Some(mean_data) = &item.mean {
                let mean_size =
                    u32::try_from(8 + mean_data.len()).expect("mean size should fit in u32");
                item_data.extend_from_slice(&mean_size.to_be_bytes());
                item_data.extend_from_slice(b"mean");
                item_data.extend_from_slice(mean_data);
            }

            if let Some(name_data) = &item.name {
                let name_size =
                    u32::try_from(8 + name_data.len()).expect("name size should fit in u32");
                item_data.extend_from_slice(&name_size.to_be_bytes());
                item_data.extend_from_slice(b"name");
                item_data.extend_from_slice(name_data);
            }

            for data_atom in item.data_atoms {
                let data_type = match &data_atom.data {
                    ListItemData::Text(_) => DATA_TYPE_TEXT,
                    ListItemData::Jpeg(_) => DATA_TYPE_JPEG,
                    _ => data_atom.data_type,
                };
                let data: Vec<u8> = data_atom.data.to_bytes();
                let data_size =
                    u32::try_from(16 + data.len()).expect("data size should fit in u32"); // header + type flags + reserved + data

                // Write data atom
                item_data.extend_from_slice(&data_size.to_be_bytes());
                item_data.extend_from_slice(b"data");
                item_data.extend_from_slice(&data_type.to_be_bytes());
                item_data.extend_from_slice(&data_atom.reserved.to_be_bytes());
                item_data.extend_from_slice(&data);
            }

            let item_size =
                u32::try_from(8 + item_data.len()).expect("item size should fit in u32"); // size + type + data

            // Write item
            output.extend_from_slice(&item_size.to_be_bytes());
            output.extend_from_slice(&item.item_type.into_bytes());
            output.extend_from_slice(&item_data);
        }

        output
    }
}

fn parse_ilst_data<R: std::io::Read>(mut reader: R) -> anyhow::Result<ItemListAtom> {
    let mut items = Vec::new();

    loop {
        // Try to read size
        let Ok(size) = read_u32(&mut reader) else {
            break;
        };

        if size == 0 {
            break;
        }

        let actual_size = if size == 1 {
            read_u64(&mut reader)?
        } else {
            u64::from(size)
        };

        let item_type = read_fourcc(&mut reader)?;

        // Calculate remaining bytes for this item
        let header_size = if size == 1 { 16 } else { 8 }; // size + type + optional extended size
        let remaining_size = actual_size - header_size;

        // Read the item data
        let item_data = read_bytes(
            &mut reader,
            usize::try_from(remaining_size).expect("u64 should fit in usize"),
        )?;
        let mut item_reader = std::io::Cursor::new(&item_data);

        let mut data_atoms = Vec::new();
        let mut mean_data = None;
        let mut name_data = None;

        while item_reader.position() < item_data.len() as u64 {
            let Ok(data_size) = read_u32(&mut item_reader) else {
                break;
            };

            if data_size == 0 {
                break;
            }

            let data_actual_size = if data_size == 1 {
                read_u64(&mut item_reader)?
            } else {
                u64::from(data_size)
            };

            let atom_type = read_fourcc(&mut item_reader)?;
            let remaining_atom_size = data_actual_size - if data_size == 1 { 16 } else { 8 };

            match atom_type.as_slice() {
                b"data" => {
                    let data_atom = parse_data_atom(&mut item_reader, remaining_atom_size)?;
                    data_atoms.push(data_atom);
                }
                b"mean" => {
                    // Parse mean atom (for ---- items)
                    if item_type == b"----" {
                        let mean_bytes = read_bytes(
                            &mut item_reader,
                            usize::try_from(remaining_atom_size).expect("u64 should fit in usize"),
                        )?;
                        mean_data = Some(mean_bytes);
                    } else {
                        bail!("unexpected 'mean' atom in non-'----' item");
                    }
                }
                b"name" => {
                    // Parse name atom (for ---- items)
                    if item_type == b"----" {
                        let name_bytes = read_bytes(
                            &mut item_reader,
                            usize::try_from(remaining_atom_size).expect("u64 should fit in usize"),
                        )?;
                        name_data = Some(name_bytes);
                    } else {
                        bail!("unexpected 'name' atom in non-'----' item");
                    }
                }
                _ => {
                    bail!("unexpected atom type '{atom_type}' in {item_type} item");
                }
            }
        }

        items.push(MetadataItem {
            item_type,
            mean: mean_data,
            name: name_data,
            data_atoms,
        });
    }

    Ok(ItemListAtom { items })
}

fn parse_data_atom<R: Read>(reader: &mut R, size: u64) -> anyhow::Result<DataAtom> {
    if size < 8 {
        bail!("Data atom too small");
    }

    let data_type = read_u32(reader)?;
    let reserved = read_u32(reader)?;
    let data_size = size - 8; // size - data_type - reserved
    let data = read_bytes(
        reader,
        usize::try_from(data_size).expect("u64 should fit in usize"),
    )?;

    Ok(DataAtom {
        data_type,
        reserved,
        data: ListItemData::new(data_type, data),
    })
}

fn read_u32<R: Read>(reader: &mut R) -> anyhow::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_u64<R: Read>(reader: &mut R) -> anyhow::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_be_bytes(buf))
}

fn read_fourcc<R: Read>(reader: &mut R) -> anyhow::Result<FourCC> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(FourCC::from(buf))
}

fn read_bytes<R: Read>(reader: &mut R, count: usize) -> anyhow::Result<Vec<u8>> {
    let mut buf = vec![0u8; count];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available ilst test data files
    #[test]
    fn test_ilst_roundtrip() {
        test_atom_roundtrip_sync::<ItemListAtom>(ILST);
    }
}
