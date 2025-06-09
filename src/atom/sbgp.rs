use anyhow::{anyhow, Context};
use std::io::{Cursor, Read};

use crate::atom::util::{parse_fixed_size_atom, FourCC};

pub const SBGP: &[u8; 4] = b"sbgp";

/// Sample-to-Group Atom (sbgp) - ISO/IEC 14496-12
/// This atom maps samples to sample groups defined in the corresponding sgpd atom.
#[derive(Debug, Clone)]
pub struct SampleToGroupAtom {
    /// Version of the sbgp atom format (0 or 1+)
    pub version: u8,
    /// Flags for the atom
    pub flags: [u8; 3],
    /// Grouping type - identifies the type of grouping (must match corresponding sgpd)
    pub grouping_type: FourCC,
    /// Grouping type parameter (version >= 1 only)
    pub grouping_type_parameter: Option<u32>,
    /// Sample-to-group mapping entries
    pub entries: Vec<SampleToGroupEntry>,
}

/// A single sample-to-group mapping entry
#[derive(Debug, Clone)]
pub struct SampleToGroupEntry {
    /// Number of consecutive samples that belong to the same group
    pub sample_count: u32,
    /// Index into the sample group description table (1-based, 0 means no group assignment)
    pub group_description_index: u32,
}

impl SampleToGroupAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_sample_to_group_atom(reader)
    }

    /// Get the number of sample-to-group entries
    pub fn entry_count(&self) -> u32 {
        self.entries.len() as u32
    }

    /// Check if the atom has any entries
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the grouping type as a string
    pub fn grouping_type_string(&self) -> String {
        self.grouping_type.to_string()
    }

    /// Get an entry by index
    pub fn get_entry(&self, index: usize) -> Option<&SampleToGroupEntry> {
        self.entries.get(index)
    }

    /// Get the total number of samples covered by all entries
    pub fn total_sample_count(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.sample_count as u64)
            .sum()
    }

    /// Find the group description index for a specific sample (1-based sample index)
    pub fn get_group_for_sample(&self, sample_index: u32) -> Option<u32> {
        if sample_index == 0 {
            return None; // Sample indices are 1-based
        }

        let mut current_sample = 1;
        for entry in &self.entries {
            let end_sample = current_sample + entry.sample_count - 1;
            if sample_index >= current_sample && sample_index <= end_sample {
                return Some(entry.group_description_index);
            }
            current_sample += entry.sample_count;
        }
        None
    }

    /// Get all samples that belong to a specific group
    pub fn get_samples_in_group(&self, group_description_index: u32) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        let mut current_sample = 1;

        for entry in &self.entries {
            if entry.group_description_index == group_description_index {
                let start_sample = current_sample;
                let end_sample = current_sample + entry.sample_count - 1;
                result.push((start_sample, end_sample));
            }
            current_sample += entry.sample_count;
        }
        result
    }

    /// Get statistics about the sample grouping
    pub fn get_statistics(&self) -> SampleToGroupStatistics {
        let entry_count = self.entries.len() as u32;
        let total_samples = self.total_sample_count();
        
        let mut unique_groups = std::collections::HashSet::new();
        let mut min_sample_count = u32::MAX;
        let mut max_sample_count = 0u32;
        let mut ungrouped_samples = 0u64;

        for entry in &self.entries {
            if entry.group_description_index > 0 {
                unique_groups.insert(entry.group_description_index);
            } else {
                ungrouped_samples += entry.sample_count as u64;
            }
            
            if entry.sample_count < min_sample_count {
                min_sample_count = entry.sample_count;
            }
            if entry.sample_count > max_sample_count {
                max_sample_count = entry.sample_count;
            }
        }

        if self.entries.is_empty() {
            min_sample_count = 0;
        }

        let average_samples_per_entry = if entry_count > 0 {
            total_samples as f64 / entry_count as f64
        } else {
            0.0
        };

        SampleToGroupStatistics {
            entry_count,
            total_samples,
            unique_groups: unique_groups.len() as u32,
            min_samples_per_entry: min_sample_count,
            max_samples_per_entry: max_sample_count,
            average_samples_per_entry,
            ungrouped_samples,
        }
    }

    /// Validate the atom structure
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        // Check version-specific constraints
        match self.version {
            0 => {
                if self.grouping_type_parameter.is_some() {
                    return Err(anyhow!("Version 0 should not have grouping_type_parameter"));
                }
            }
            _ => {
                // Version 1+
                if self.grouping_type_parameter.is_none() {
                    return Err(anyhow!(
                        "Version {} should have grouping_type_parameter",
                        self.version
                    ));
                }
            }
        }

        // Validate entries
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.sample_count == 0 {
                return Err(anyhow!("Entry {} has zero sample_count", i));
            }
            // Note: group_description_index of 0 is valid and means "no group assignment"
        }

        Ok(())
    }

    /// Check if grouping type matches another atom (for use with sgpd)
    pub fn matches_grouping_type(&self, other_grouping_type: &FourCC) -> bool {
        self.grouping_type == *other_grouping_type
    }

    /// Check if any samples are ungrouped (have group_description_index == 0)
    pub fn has_ungrouped_samples(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.group_description_index == 0)
    }
}

/// Statistics about sample-to-group mapping
#[derive(Debug, Clone)]
pub struct SampleToGroupStatistics {
    /// Number of entries in the atom
    pub entry_count: u32,
    /// Total number of samples covered
    pub total_samples: u64,
    /// Number of unique groups referenced
    pub unique_groups: u32,
    /// Minimum samples per entry
    pub min_samples_per_entry: u32,
    /// Maximum samples per entry
    pub max_samples_per_entry: u32,
    /// Average samples per entry
    pub average_samples_per_entry: f64,
    /// Number of ungrouped samples (group_description_index == 0)
    pub ungrouped_samples: u64,
}

impl TryFrom<&[u8]> for SampleToGroupAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_sample_to_group_atom(reader)
    }
}

fn parse_sample_to_group_atom<R: Read>(reader: R) -> Result<SampleToGroupAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;

    // Verify this is an sbgp atom
    if atom_type != SBGP {
        return Err(anyhow!(
            "Invalid atom type: {} (expected 'sbgp')",
            atom_type
        ));
    }

    parse_sbgp_data(&data)
}

fn parse_sbgp_data(data: &[u8]) -> Result<SampleToGroupAtom, anyhow::Error> {
    let mut cursor = Cursor::new(data);
    let mut buffer = [0u8; 4];

    // Read version and flags (4 bytes total)
    cursor
        .read_exact(&mut buffer)
        .context("Failed to read version and flags")?;
    let version = buffer[0];
    let flags = [buffer[1], buffer[2], buffer[3]];

    // Read grouping_type (4 bytes)
    cursor
        .read_exact(&mut buffer)
        .context("Failed to read grouping_type")?;
    let grouping_type = FourCC(buffer);

    // Version-dependent fields
    let mut grouping_type_parameter = None;
    if version >= 1 {
        cursor
            .read_exact(&mut buffer)
            .context("Failed to read grouping_type_parameter")?;
        grouping_type_parameter = Some(u32::from_be_bytes(buffer));
    }

    // Read entry_count
    cursor
        .read_exact(&mut buffer)
        .context("Failed to read entry_count")?;
    let entry_count = u32::from_be_bytes(buffer);

    // Read entries
    let mut entries = Vec::new();
    for i in 0..entry_count {
        // Read sample_count
        cursor.read_exact(&mut buffer).with_context(|| {
            format!("Failed to read sample_count for entry {}", i)
        })?;
        let sample_count = u32::from_be_bytes(buffer);

        // Read group_description_index
        cursor.read_exact(&mut buffer).with_context(|| {
            format!("Failed to read group_description_index for entry {}", i)
        })?;
        let group_description_index = u32::from_be_bytes(buffer);

        entries.push(SampleToGroupEntry {
            sample_count,
            group_description_index,
        });
    }

    let atom = SampleToGroupAtom {
        version,
        flags,
        grouping_type,
        grouping_type_parameter,
        entries,
    };

    // Validate the parsed atom
    atom.validate()?;

    Ok(atom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sbgp_data_v0(
        grouping_type: &[u8; 4],
        entries: &[(u32, u32)],
    ) -> Vec<u8> {
        let mut data = Vec::new();
        let total_size = 8 + 4 + 4 + 4 + (entries.len() * 8); // header + grouping_type + entry_count + entries

        // Atom header
        data.extend_from_slice(&(total_size as u32).to_be_bytes());
        data.extend_from_slice(SBGP);

        // Version and flags
        data.push(0); // version 0
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Grouping type
        data.extend_from_slice(grouping_type);

        // Entry count
        data.extend_from_slice(&(entries.len() as u32).to_be_bytes());

        // Entries
        for (sample_count, group_description_index) in entries {
            data.extend_from_slice(&sample_count.to_be_bytes());
            data.extend_from_slice(&group_description_index.to_be_bytes());
        }

        data
    }

    fn create_test_sbgp_data_v1(
        grouping_type: &[u8; 4],
        grouping_type_parameter: u32,
        entries: &[(u32, u32)],
    ) -> Vec<u8> {
        let mut data = Vec::new();
        let total_size = 8 + 4 + 4 + 4 + 4 + (entries.len() * 8); // header + grouping_type + parameter + entry_count + entries

        // Atom header
        data.extend_from_slice(&(total_size as u32).to_be_bytes());
        data.extend_from_slice(SBGP);

        // Version and flags
        data.push(1); // version 1
        data.extend_from_slice(&[0, 0, 0]); // flags

        // Grouping type
        data.extend_from_slice(grouping_type);

        // Grouping type parameter
        data.extend_from_slice(&grouping_type_parameter.to_be_bytes());

        // Entry count
        data.extend_from_slice(&(entries.len() as u32).to_be_bytes());

        // Entries
        for (sample_count, group_description_index) in entries {
            data.extend_from_slice(&sample_count.to_be_bytes());
            data.extend_from_slice(&group_description_index.to_be_bytes());
        }

        data
    }

    #[test]
    fn test_parse_sbgp_version_0() {
        let entries = [(10, 1), (5, 2), (8, 1)];
        let data = create_test_sbgp_data_v0(b"test", &entries);
        let result = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        assert_eq!(result.version, 0);
        assert_eq!(result.grouping_type, FourCC(*b"test"));
        assert_eq!(result.grouping_type_parameter, None);
        assert_eq!(result.entries.len(), 3);
        assert_eq!(result.entries[0].sample_count, 10);
        assert_eq!(result.entries[0].group_description_index, 1);
        assert_eq!(result.entries[1].sample_count, 5);
        assert_eq!(result.entries[1].group_description_index, 2);
        assert_eq!(result.entries[2].sample_count, 8);
        assert_eq!(result.entries[2].group_description_index, 1);
    }

    #[test]
    fn test_parse_sbgp_version_1() {
        let entries = [(20, 1), (15, 0)]; // Second entry is ungrouped
        let data = create_test_sbgp_data_v1(b"grp1", 0x12345678, &entries);
        let result = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        assert_eq!(result.version, 1);
        assert_eq!(result.grouping_type, FourCC(*b"grp1"));
        assert_eq!(result.grouping_type_parameter, Some(0x12345678));
        assert_eq!(result.entries.len(), 2);
        assert_eq!(result.entries[0].sample_count, 20);
        assert_eq!(result.entries[0].group_description_index, 1);
        assert_eq!(result.entries[1].sample_count, 15);
        assert_eq!(result.entries[1].group_description_index, 0);
    }

    #[test]
    fn test_total_sample_count() {
        let entries = [(10, 1), (5, 2), (8, 1)];
        let data = create_test_sbgp_data_v0(b"test", &entries);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        assert_eq!(atom.total_sample_count(), 23);
    }

    #[test]
    fn test_get_group_for_sample() {
        let entries = [(10, 1), (5, 2), (8, 1)];
        let data = create_test_sbgp_data_v0(b"test", &entries);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        // First group: samples 1-10
        assert_eq!(atom.get_group_for_sample(1), Some(1));
        assert_eq!(atom.get_group_for_sample(5), Some(1));
        assert_eq!(atom.get_group_for_sample(10), Some(1));

        // Second group: samples 11-15
        assert_eq!(atom.get_group_for_sample(11), Some(2));
        assert_eq!(atom.get_group_for_sample(15), Some(2));

        // Third group: samples 16-23
        assert_eq!(atom.get_group_for_sample(16), Some(1));
        assert_eq!(atom.get_group_for_sample(23), Some(1));

        // Out of range
        assert_eq!(atom.get_group_for_sample(0), None);
        assert_eq!(atom.get_group_for_sample(24), None);
    }

    #[test]
    fn test_get_samples_in_group() {
        let entries = [(10, 1), (5, 2), (8, 1)];
        let data = create_test_sbgp_data_v0(b"test", &entries);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        let group1_samples = atom.get_samples_in_group(1);
        assert_eq!(group1_samples, vec![(1, 10), (16, 23)]);

        let group2_samples = atom.get_samples_in_group(2);
        assert_eq!(group2_samples, vec![(11, 15)]);

        let no_samples = atom.get_samples_in_group(99);
        assert!(no_samples.is_empty());
    }

    #[test]
    fn test_statistics() {
        let entries = [(10, 1), (5, 2), (8, 0)]; // Last entry is ungrouped
        let data = create_test_sbgp_data_v0(b"test", &entries);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        let stats = atom.get_statistics();
        assert_eq!(stats.entry_count, 3);
        assert_eq!(stats.total_samples, 23);
        assert_eq!(stats.unique_groups, 2); // Groups 1 and 2, not counting 0
        assert_eq!(stats.min_samples_per_entry, 5);
        assert_eq!(stats.max_samples_per_entry, 10);
        assert!((stats.average_samples_per_entry - (23.0 / 3.0)).abs() < 0.001);
        assert_eq!(stats.ungrouped_samples, 8);
    }

    #[test]
    fn test_has_ungrouped_samples() {
        let entries_with_ungrouped = [(10, 1), (5, 0)];
        let data = create_test_sbgp_data_v0(b"test", &entries_with_ungrouped);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();
        assert!(atom.has_ungrouped_samples());

        let entries_all_grouped = [(10, 1), (5, 2)];
        let data = create_test_sbgp_data_v0(b"test", &entries_all_grouped);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();
        assert!(!atom.has_ungrouped_samples());
    }

    #[test]
    fn test_validation() {
        let entries = [(10, 1), (5, 2)];
        let data = create_test_sbgp_data_v0(b"test", &entries);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();
        
        // Should validate successfully
        atom.validate().unwrap();
    }

    #[test]
    fn test_invalid_sample_count() {
        let mut data = create_test_sbgp_data_v0(b"test", &[(10, 1)]);
        
        // Corrupt the sample_count to 0 (invalid)
        let sample_count_offset = 8 + 4 + 4 + 4; // header + grouping_type + entry_count
        data[sample_count_offset..sample_count_offset + 4].copy_from_slice(&0u32.to_be_bytes());
        
        let result = parse_sample_to_group_atom(Cursor::new(&data));
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_atom() {
        let data = create_test_sbgp_data_v0(b"empt", &[]);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        assert!(atom.is_empty());
        assert_eq!(atom.entry_count(), 0);
        assert_eq!(atom.total_sample_count(), 0);
        assert!(atom.get_entry(0).is_none());
        assert!(!atom.has_ungrouped_samples());
    }

    #[test]
    fn test_invalid_atom_type() {
        let data = create_test_sbgp_data_v0(b"test", &[(1, 1)]);
        let mut modified_data = data;
        // Change atom type from 'sbgp' to 'badd'
        modified_data[4..8].copy_from_slice(b"badd");

        let result = parse_sample_to_group_atom(Cursor::new(&modified_data));
        assert!(result.is_err());
    }

    #[test]
    fn test_try_from_trait() {
        let entries = [(7, 3), (8, 1)];
        let data = create_test_sbgp_data_v0(b"from", &entries);
        let result = SampleToGroupAtom::try_from(data.as_slice()).unwrap();

        assert_eq!(result.grouping_type_string(), "from");
        assert_eq!(result.entries.len(), 2);
        assert_eq!(result.entries[0].sample_count, 7);
        assert_eq!(result.entries[0].group_description_index, 3);
    }

    #[test]
    fn test_matches_grouping_type() {
        let entries = [(5, 1)];
        let data = create_test_sbgp_data_v0(b"test", &entries);
        let atom = parse_sample_to_group_atom(Cursor::new(&data)).unwrap();

        assert!(atom.matches_grouping_type(&FourCC(*b"test")));
        assert!(!atom.matches_grouping_type(&FourCC(*b"diff")));
    }
}