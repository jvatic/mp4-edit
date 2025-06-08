use anyhow::anyhow;
use std::{
    fmt,
    io::{Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis, FourCC};

pub const FREE: &[u8; 4] = b"free";
pub const SKIP: &[u8; 4] = b"skip";

#[derive(Clone)]
pub struct FreeAtom {
    /// The atom type (either 'free' or 'skip')
    pub atom_type: FourCC,
    /// Size of the free space data
    pub data_size: usize,
    /// The actual free space data (usually ignored)
    pub data: Vec<u8>,
}

impl fmt::Debug for FreeAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreeAtom")
            .field("atom_type", &self.atom_type)
            .field("data_size", &self.data_size)
            .field("data", &DebugEllipsis(Some(self.data.len())))
            .finish()
    }
}

impl FreeAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_free_atom(reader)
    }

    /// Check if this is a 'free' type atom
    pub fn is_free(&self) -> bool {
        self.atom_type == FREE
    }

    /// Check if this is a 'skip' type atom
    pub fn is_skip(&self) -> bool {
        self.atom_type == SKIP
    }

    /// Get the atom type as a string
    pub fn atom_type_string(&self) -> String {
        self.atom_type.to_string()
    }

    /// Check if the free space contains only zero bytes
    pub fn is_zeroed(&self) -> bool {
        self.data.iter().all(|&b| b == 0)
    }

    /// Check if the free space appears to contain random/garbage data
    pub fn appears_random(&self) -> bool {
        if self.data.is_empty() {
            return false;
        }
        
        // Simple heuristic: if we have a reasonable variety of byte values,
        // it's probably not structured data
        let mut byte_counts = [0u32; 256];
        for &byte in &self.data {
            byte_counts[byte as usize] += 1;
        }
        
        let unique_bytes = byte_counts.iter().filter(|&&count| count > 0).count();
        unique_bytes > 16 // If we see more than 16 different byte values, likely random
    }

    /// Get a summary of the free space content
    pub fn get_content_summary(&self) -> FreeSpaceSummary {
        FreeSpaceSummary {
            atom_type: self.atom_type_string(),
            size: self.data_size,
            is_zeroed: self.is_zeroed(),
            appears_random: self.appears_random(),
            first_bytes: if self.data.len() >= 16 {
                Some(self.data[..16].to_vec())
            } else {
                Some(self.data.clone())
            },
        }
    }

    /// Create a new FreeAtom with zero-filled data
    pub fn new_zeroed(size: usize) -> Self {
        Self {
            atom_type: FourCC(*FREE),
            data_size: size,
            data: vec![0u8; size],
        }
    }

    /// Create a new FreeAtom with specific data
    pub fn new_with_data(data: Vec<u8>) -> Self {
        let size = data.len();
        Self {
            atom_type: FourCC(*FREE),
            data_size: size,
            data,
        }
    }

    /// Create a new SkipAtom (functionally identical to free)
    pub fn new_skip(size: usize) -> Self {
        Self {
            atom_type: FourCC(*SKIP),
            data_size: size,
            data: vec![0u8; size],
        }
    }

    /// Clear the data (replace with zeros) while maintaining size
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Resize the free space
    pub fn resize(&mut self, new_size: usize) {
        self.data.resize(new_size, 0);
        self.data_size = new_size;
    }
}

#[derive(Debug, Clone)]
pub struct FreeSpaceSummary {
    pub atom_type: String,
    pub size: usize,
    pub is_zeroed: bool,
    pub appears_random: bool,
    pub first_bytes: Option<Vec<u8>>,
}

impl Default for FreeAtom {
    fn default() -> Self {
        Self::new_zeroed(0)
    }
}

impl TryFrom<&[u8]> for FreeAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_free_atom(reader)
    }
}

fn parse_free_atom<R: Read>(reader: R) -> Result<FreeAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    
    // Verify this is a free or skip atom
    if atom_type != FREE && atom_type != SKIP {
        return Err(anyhow!("Invalid atom type: {} (expected 'free' or 'skip')", atom_type));
    }

    Ok(FreeAtom {
        atom_type,
        data_size: data.len(),
        data,
    })
}

/// Check if an atom type represents free space
pub fn is_free_space_atom(atom_type: &[u8; 4]) -> bool {
    atom_type == FREE || atom_type == SKIP
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_free_data(atom_type: &[u8; 4], content: &[u8]) -> Vec<u8> {
        let mut data = Vec::new();
        let total_size = 8 + content.len();
        
        data.extend_from_slice(&(total_size as u32).to_be_bytes());
        data.extend_from_slice(atom_type);
        data.extend_from_slice(content);
        
        data
    }

    #[test]
    fn test_parse_free_atom() {
        let test_content = vec![0, 1, 2, 3, 4, 5];
        let data = create_test_free_data(FREE, &test_content);
        let result = parse_free_atom(Cursor::new(&data)).unwrap();

        assert_eq!(result.atom_type, FourCC(*FREE));
        assert_eq!(result.data_size, test_content.len());
        assert_eq!(result.data, test_content);
        assert!(result.is_free());
        assert!(!result.is_skip());
    }

    #[test]
    fn test_parse_skip_atom() {
        let test_content = vec![0xff; 100];
        let data = create_test_free_data(SKIP, &test_content);
        let result = parse_free_atom(Cursor::new(&data)).unwrap();

        assert_eq!(result.atom_type, FourCC(*SKIP));
        assert_eq!(result.data_size, test_content.len());
        assert_eq!(result.data, test_content);
        assert!(!result.is_free());
        assert!(result.is_skip());
    }

    #[test]
    fn test_zeroed_detection() {
        let free_atom = FreeAtom::new_zeroed(100);
        assert!(free_atom.is_zeroed());
        assert!(!free_atom.appears_random());

        let random_data = (0..100).map(|i| (i * 7) as u8).collect();
        let random_atom = FreeAtom::new_with_data(random_data);
        assert!(!random_atom.is_zeroed());
        assert!(random_atom.appears_random());
    }

    #[test]
    fn test_content_summary() {
        let test_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
        let atom = FreeAtom::new_with_data(test_data.clone());
        let summary = atom.get_content_summary();

        assert_eq!(summary.atom_type, "free");
        assert_eq!(summary.size, test_data.len());
        assert!(!summary.is_zeroed);
        assert_eq!(summary.first_bytes.unwrap(), test_data[..16]);
    }

    #[test]
    fn test_atom_manipulation() {
        let mut atom = FreeAtom::new_zeroed(50);
        assert_eq!(atom.data_size, 50);
        assert!(atom.is_zeroed());

        atom.resize(100);
        assert_eq!(atom.data_size, 100);
        assert_eq!(atom.data.len(), 100);

        // Modify some data
        atom.data[10] = 0xff;
        assert!(!atom.is_zeroed());

        atom.clear();
        assert!(atom.is_zeroed());
    }

    #[test]
    fn test_default_atom() {
        let atom = FreeAtom::default();
        assert_eq!(atom.atom_type, FourCC(*FREE));
        assert_eq!(atom.data_size, 0);
        assert!(atom.data.is_empty());
        assert!(atom.is_zeroed());
    }

    #[test]
    fn test_new_constructors() {
        let free_atom = FreeAtom::new_zeroed(10);
        assert!(free_atom.is_free());
        assert_eq!(free_atom.data_size, 10);

        let skip_atom = FreeAtom::new_skip(20);
        assert!(skip_atom.is_skip());
        assert_eq!(skip_atom.data_size, 20);

        let data_atom = FreeAtom::new_with_data(vec![1, 2, 3]);
        assert_eq!(data_atom.data, vec![1, 2, 3]);
        assert_eq!(data_atom.data_size, 3);
    }

    #[test]
    fn test_is_free_space_atom() {
        assert!(is_free_space_atom(FREE));
        assert!(is_free_space_atom(SKIP));
        assert!(!is_free_space_atom(b"moov"));
        assert!(!is_free_space_atom(b"trak"));
    }

    #[test]
    fn test_invalid_atom_type() {
        let data = create_test_free_data(b"moov", &[1, 2, 3]);
        let result = parse_free_atom(Cursor::new(&data));
        assert!(result.is_err());
    }

    #[test]
    fn test_appears_random_heuristic() {
        // Empty data should not appear random
        let empty_atom = FreeAtom::new_with_data(vec![]);
        assert!(!empty_atom.appears_random());

        // Repeating pattern should not appear random
        let pattern_atom = FreeAtom::new_with_data(vec![0xAA; 100]);
        assert!(!pattern_atom.appears_random());

        // Varied data should appear random
        let varied_data: Vec<u8> = (0..100).map(|i| ((i * 13 + 7) % 256) as u8).collect();
        let varied_atom = FreeAtom::new_with_data(varied_data);
        assert!(varied_atom.appears_random());
    }
}