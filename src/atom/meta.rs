use anyhow::{anyhow, Context};
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis, FourCC};

pub const META: &[u8; 4] = b"meta";

#[derive(Clone)]
pub struct MetadataAtom {
    /// Version of the meta atom format (usually 0)
    pub version: u8,
    /// Flags for the meta atom (usually all zeros)
    pub flags: [u8; 3],
    /// Raw child atom data (contains various metadata atoms)
    pub child_data: Vec<u8>,
    /// Parsed child atoms for easier access
    pub child_atoms: Vec<ChildAtom>,
}

impl fmt::Debug for MetadataAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetadataAtom")
            .field("version", &self.version)
            .field("flags", &self.flags)
            .field("child_data", &DebugEllipsis(Some(self.child_data.len())))
            .field("child_atoms", &self.child_atoms)
            .finish()
    }
}

#[derive(Clone)]
pub struct ChildAtom {
    /// Atom type (fourcc)
    pub atom_type: FourCC,
    /// Atom size
    pub size: u32,
    /// Atom data (excluding header)
    pub data: Vec<u8>,
}

impl fmt::Debug for ChildAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChildAtom")
            .field("atom_type", &self.atom_type)
            .field("size", &self.size)
            .field("data", &DebugEllipsis(Some(self.data.len())))
            .finish()
    }
}

impl MetadataAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_meta_atom(reader)
    }

    /// Get the handler type from hdlr child atom if present
    pub fn get_handler_type(&self) -> Option<FourCC> {
        for child in &self.child_atoms {
            if child.atom_type == b"hdlr" && child.data.len() >= 12 {
                // hdlr has version(1) + flags(3) + pre_defined(4) + handler_type(4)
                let handler_type: [u8; 4] = child.data[8..12].try_into().ok()?;
                return Some(FourCC(handler_type));
            }
        }
        None
    }

    /// Get the handler type as a string
    pub fn get_handler_type_string(&self) -> Option<String> {
        self.get_handler_type().map(|ht| ht.to_string())
    }

    /// Check if this is a picture metadata handler
    pub fn is_picture_handler(&self) -> bool {
        self.get_handler_type().map_or(false, |ht| ht == b"pict")
    }

    /// Check if this is a general metadata handler
    pub fn is_metadata_handler(&self) -> bool {
        self.get_handler_type().map_or(false, |ht| ht == b"mdta")
    }

    /// Get the primary item ID from pitm child atom if present
    pub fn get_primary_item_id(&self) -> Option<u32> {
        for child in &self.child_atoms {
            if child.atom_type == b"pitm" {
                return self.parse_pitm_data(&child.data).ok();
            }
        }
        None
    }

    /// Find child atoms of a specific type
    pub fn find_child_atoms(&self, atom_type: &[u8; 4]) -> Vec<&ChildAtom> {
        self.child_atoms
            .iter()
            .filter(|child| child.atom_type == atom_type)
            .collect()
    }

    /// Check if a specific child atom type exists
    pub fn has_child_atom(&self, atom_type: &[u8; 4]) -> bool {
        self.child_atoms
            .iter()
            .any(|child| child.atom_type == atom_type)
    }

    /// Get a list of all child atom types
    pub fn get_child_atom_types(&self) -> Vec<String> {
        self.child_atoms
            .iter()
            .map(|child| child.atom_type.to_string())
            .collect()
    }

    /// Get summary information about the metadata
    pub fn get_summary(&self) -> MetadataSummary {
        MetadataSummary {
            version: self.version,
            flags: self.flags,
            handler_type: self.get_handler_type_string(),
            primary_item_id: self.get_primary_item_id(),
            child_atom_count: self.child_atoms.len(),
            child_atom_types: self.get_child_atom_types(),
            has_item_info: self.has_child_atom(b"iinf"),
            has_item_location: self.has_child_atom(b"iloc"),
            has_item_properties: self.has_child_atom(b"iprp"),
            has_keys: self.has_child_atom(b"keys"),
            has_item_list: self.has_child_atom(b"ilst"),
        }
    }

    /// Validate the meta atom structure
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        // Version should typically be 0
        if self.version > 1 {
            return Err(anyhow!(
                "Unusual version: {} (expected 0 or 1)",
                self.version
            ));
        }

        // Should have at least an hdlr atom
        if !self.has_child_atom(b"hdlr") {
            return Err(anyhow!("Missing required hdlr (handler) atom"));
        }

        // Validate handler type
        if let Some(handler) = self.get_handler_type() {
            // Check for common handler types
            if handler == b"pict" || handler == b"mdta" || handler == b"mdir" {
                // Valid handlers
            } else {
                // Unknown handler, but not necessarily invalid
            }
        }

        Ok(())
    }

    /// Parse pitm (primary item) data
    fn parse_pitm_data(&self, data: &[u8]) -> Result<u32, anyhow::Error> {
        if data.len() < 6 {
            return Err(anyhow!("pitm data too short"));
        }

        let version = data[0];
        // flags are data[1..4]

        match version {
            0 => {
                if data.len() < 6 {
                    return Err(anyhow!("pitm v0 data too short"));
                }
                Ok(u16::from_be_bytes([data[4], data[5]]) as u32)
            }
            1 => {
                if data.len() < 8 {
                    return Err(anyhow!("pitm v1 data too short"));
                }
                Ok(u32::from_be_bytes([data[4], data[5], data[6], data[7]]))
            }
            _ => Err(anyhow!("Unsupported pitm version: {}", version)),
        }
    }

    /// Create a new MetadataAtom with default values
    pub fn new() -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            child_data: Vec::new(),
            child_atoms: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetadataSummary {
    pub version: u8,
    pub flags: [u8; 3],
    pub handler_type: Option<String>,
    pub primary_item_id: Option<u32>,
    pub child_atom_count: usize,
    pub child_atom_types: Vec<String>,
    pub has_item_info: bool,
    pub has_item_location: bool,
    pub has_item_properties: bool,
    pub has_keys: bool,
    pub has_item_list: bool,
}

impl Default for MetadataAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl TryFrom<&[u8]> for MetadataAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_meta_atom(reader)
    }
}

fn parse_meta_atom<R: Read>(reader: R) -> Result<MetadataAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != META {
        return Err(anyhow!("Invalid atom type: {}", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_meta_data(&mut cursor)
}

fn parse_meta_data<R: Read>(mut reader: R) -> Result<MetadataAtom, anyhow::Error> {
    // Read version and flags (4 bytes total)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Read remaining data (contains child atoms)
    let mut child_data = Vec::new();
    reader
        .read_to_end(&mut child_data)
        .context("read child data")?;

    // Parse child atoms from the remaining data
    let child_atoms = parse_child_atoms(&child_data)?;

    Ok(MetadataAtom {
        version,
        flags,
        child_data,
        child_atoms,
    })
}

fn parse_child_atoms(data: &[u8]) -> Result<Vec<ChildAtom>, anyhow::Error> {
    let mut atoms = Vec::new();
    let mut offset = 0;

    while offset + 8 <= data.len() {
        // Read atom header (8 bytes minimum)
        let size = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);

        let atom_type = FourCC([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);

        if size == 0 {
            // Size 0 means "rest of data"
            let atom_data = data[offset + 8..].to_vec();
            atoms.push(ChildAtom {
                atom_type,
                size,
                data: atom_data,
            });
            break;
        } else if size == 1 {
            // Extended size (64-bit) - more complex, skip for now
            return Err(anyhow!(
                "Extended size atoms not yet supported in meta children"
            ));
        } else if size < 8 {
            return Err(anyhow!("Invalid child atom size: {}", size));
        }

        let data_size = (size as usize).saturating_sub(8);
        let end_offset = offset + 8 + data_size;

        if end_offset > data.len() {
            return Err(anyhow!("Child atom extends beyond data bounds"));
        }

        let atom_data = data[offset + 8..end_offset].to_vec();
        atoms.push(ChildAtom {
            atom_type,
            size,
            data: atom_data,
        });

        offset = end_offset;
    }

    Ok(atoms)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_meta_data() -> Vec<u8> {
        let mut data = Vec::new();

        // Version and flags
        data.push(0); // version
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Add a simple hdlr atom
        let hdlr_size = 25u32; // 8 header + 17 data
        data.extend_from_slice(&hdlr_size.to_be_bytes());
        data.extend_from_slice(b"hdlr");
        // hdlr data: version(1) + flags(3) + pre_defined(4) + handler_type(4) + name(5)
        data.push(0); // version
        data.extend_from_slice(&[0, 0, 0]); // flags
        data.extend_from_slice(&[0, 0, 0, 0]); // pre_defined
        data.extend_from_slice(b"pict"); // handler_type
        data.extend_from_slice(b"test\0"); // null-terminated name

        data
    }

    #[test]
    fn test_parse_meta_data() {
        let data = create_test_meta_data();
        let result = parse_meta_data(Cursor::new(&data)).unwrap();

        assert_eq!(result.version, 0);
        assert_eq!(result.flags, [0, 0, 0]);
        assert_eq!(result.child_atoms.len(), 1);
        assert_eq!(result.child_atoms[0].atom_type, FourCC(*b"hdlr"));
        assert_eq!(result.get_handler_type(), Some(FourCC(*b"pict")));
        assert!(result.is_picture_handler());
        assert!(!result.is_metadata_handler());
    }

    #[test]
    fn test_default_meta() {
        let meta = MetadataAtom::default();
        assert_eq!(meta.version, 0);
        assert_eq!(meta.flags, [0, 0, 0]);
        assert!(meta.child_data.is_empty());
        assert!(meta.child_atoms.is_empty());
    }

    #[test]
    fn test_child_atom_operations() {
        let data = create_test_meta_data();
        let meta = parse_meta_data(Cursor::new(&data)).unwrap();

        assert!(meta.has_child_atom(b"hdlr"));
        assert!(!meta.has_child_atom(b"keys"));

        let hdlr_atoms = meta.find_child_atoms(b"hdlr");
        assert_eq!(hdlr_atoms.len(), 1);

        let types = meta.get_child_atom_types();
        assert_eq!(types, vec!["hdlr"]);
    }

    #[test]
    fn test_validation() {
        let data = create_test_meta_data();
        let meta = parse_meta_data(Cursor::new(&data)).unwrap();

        assert!(meta.validate().is_ok());

        // Test meta without hdlr
        let meta_no_hdlr = MetadataAtom::new();
        assert!(meta_no_hdlr.validate().is_err());
    }

    #[test]
    fn test_summary() {
        let data = create_test_meta_data();
        let meta = parse_meta_data(Cursor::new(&data)).unwrap();

        let summary = meta.get_summary();
        assert_eq!(summary.version, 0);
        assert_eq!(summary.handler_type, Some("pict".to_string()));
        assert_eq!(summary.child_atom_count, 1);
        assert_eq!(summary.child_atom_types, vec!["hdlr"]);
        assert!(!summary.has_keys);
        assert!(!summary.has_item_list);
    }

    #[test]
    fn test_pitm_parsing() {
        let meta = MetadataAtom::new();

        // Test pitm version 0 data
        let pitm_v0_data = vec![0, 0, 0, 0, 0x12, 0x34]; // version 0, flags, item_id=0x1234
        assert_eq!(meta.parse_pitm_data(&pitm_v0_data).unwrap(), 0x1234);

        // Test pitm version 1 data
        let pitm_v1_data = vec![1, 0, 0, 0, 0x12, 0x34, 0x56, 0x78]; // version 1, flags, item_id=0x12345678
        assert_eq!(meta.parse_pitm_data(&pitm_v1_data).unwrap(), 0x12345678);
    }

    #[test]
    fn test_parse_child_atoms_empty() {
        let atoms = parse_child_atoms(&[]).unwrap();
        assert!(atoms.is_empty());
    }

    #[test]
    fn test_parse_child_atoms_invalid_size() {
        let data = vec![0, 0, 0, 4, b'h', b'd', b'l', b'r']; // size=4 (too small)
        assert!(parse_child_atoms(&data).is_err());
    }

    #[test]
    fn test_handler_type_detection() {
        let mut meta = MetadataAtom::new();

        // Add hdlr atom with mdta handler
        let mut hdlr_data = Vec::new();
        hdlr_data.push(0); // version
        hdlr_data.extend_from_slice(&[0, 0, 0]); // flags
        hdlr_data.extend_from_slice(&[0, 0, 0, 0]); // pre_defined
        hdlr_data.extend_from_slice(b"mdta"); // handler_type

        meta.child_atoms.push(ChildAtom {
            atom_type: FourCC(*b"hdlr"),
            size: (8 + hdlr_data.len()) as u32,
            data: hdlr_data,
        });

        assert!(meta.is_metadata_handler());
        assert!(!meta.is_picture_handler());
        assert_eq!(meta.get_handler_type_string(), Some("mdta".to_string()));
    }
}
