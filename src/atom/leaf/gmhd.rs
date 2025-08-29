use crate::atom::FourCC;
use crate::writer::SerializeAtom;

pub const GMHD: &[u8; 4] = b"gmhd";

/// Generic Media Header Atom (gmhd)
///
/// Used for media tracks that don't fit into standard categories like video or audio.
/// This is commonly used for text, subtitle, or chapter tracks.
#[derive(Debug, Clone, PartialEq)]
pub struct GenericMediaHeaderAtom {
    pub version: u8,
    pub flags: [u8; 3],
    /// Generic media information data - typically empty for basic implementations
    pub data: Vec<u8>,
}

impl GenericMediaHeaderAtom {
    pub fn new() -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            data: Vec::new(),
        }
    }

    pub fn with_data(data: Vec<u8>) -> Self {
        Self {
            version: 0,
            flags: [0, 0, 0],
            data,
        }
    }
}

impl Default for GenericMediaHeaderAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl SerializeAtom for GenericMediaHeaderAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*GMHD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Version (1 byte) and flags (3 bytes)
        bytes.push(self.version);
        bytes.extend_from_slice(&self.flags);

        // Additional data if any
        bytes.extend(self.data);

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmhd_creation() {
        let gmhd = GenericMediaHeaderAtom::new();
        assert_eq!(gmhd.version, 0);
        assert_eq!(gmhd.flags, [0, 0, 0]);
        assert!(gmhd.data.is_empty());
    }

    #[test]
    fn test_gmhd_with_data() {
        let data = vec![1, 2, 3, 4];
        let gmhd = GenericMediaHeaderAtom::with_data(data.clone());
        assert_eq!(gmhd.data, data);
    }

    #[test]
    fn test_gmhd_serialization() {
        let gmhd = GenericMediaHeaderAtom::new();
        let bytes = gmhd.into_body_bytes();

        // Should be 4 bytes: version + 3 flags bytes
        assert_eq!(bytes.len(), 4);
        assert_eq!(bytes, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_gmhd_serialization_with_data() {
        let data = vec![0xAB, 0xCD];
        let gmhd = GenericMediaHeaderAtom::with_data(data);
        let bytes = gmhd.into_body_bytes();

        // Should be 6 bytes: version + 3 flags bytes + 2 data bytes
        assert_eq!(bytes.len(), 6);
        assert_eq!(bytes, vec![0, 0, 0, 0, 0xAB, 0xCD]);
    }

    #[test]
    fn test_atom_type() {
        let gmhd = GenericMediaHeaderAtom::new();
        assert_eq!(gmhd.atom_type(), FourCC(*GMHD));
    }
}
