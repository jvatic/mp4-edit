use anyhow::{anyhow, Context};
use derive_more::Deref;
use std::{
    fmt,
    io::{BufRead, BufReader, Cursor, Read},
};

use crate::atom::util::{parse_fixed_size_atom, DebugEllipsis};

pub const CHPL: &[u8; 4] = b"chpl";

#[derive(Clone, Deref)]
pub struct ChapterEntries(Vec<ChapterEntry>);

impl fmt::Debug for ChapterEntries {
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

/// Chapter entry containing start time and title
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterEntry {
    /// Start time of the chapter in 100-nanosecond units
    pub start_time: u64,
    /// Chapter title as UTF-8 string
    pub title: String,
}

/// Chapter List Atom - contains chapter information for media
#[derive(Debug, Clone)]
pub struct ChapterListAtom {
    /// Version of the chpl atom format (1)
    pub version: u8,
    /// Flags for the chpl atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of chapter entries
    pub chapters: ChapterEntries,
}

impl ChapterListAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_chapter_list_atom(reader)
    }

    /// Get the number of chapters
    pub fn chapter_count(&self) -> usize {
        self.chapters.len()
    }

    /// Check if there are no chapters
    pub fn is_empty(&self) -> bool {
        self.chapters.is_empty()
    }

    /// Get the total duration covered by chapters (from first to last)
    /// Returns None if no chapters exist
    pub fn total_duration(&self) -> Option<u64> {
        if self.chapters.is_empty() {
            return None;
        }

        let first_start = self.chapters.first()?.start_time;
        let last_start = self.chapters.last()?.start_time;

        if last_start >= first_start {
            Some(last_start - first_start)
        } else {
            Some(0)
        }
    }

    /// Find chapter at a specific time
    /// Returns the chapter that contains the given time
    pub fn find_chapter_at_time(&self, time: u64) -> Option<&ChapterEntry> {
        if self.chapters.is_empty() {
            return None;
        }

        // Find the last chapter that starts at or before the given time
        let mut result = None;
        for chapter in self.chapters.iter() {
            if chapter.start_time <= time {
                result = Some(chapter);
            } else {
                break;
            }
        }
        result
    }

    /// Get chapter by index
    pub fn get_chapter(&self, index: usize) -> Option<&ChapterEntry> {
        self.chapters.get(index)
    }

    /// Get all chapter titles
    pub fn get_titles(&self) -> Vec<&str> {
        self.chapters.iter().map(|ch| ch.title.as_str()).collect()
    }

    /// Get chapters within a time range
    pub fn get_chapters_in_range(&self, start_time: u64, end_time: u64) -> Vec<&ChapterEntry> {
        self.chapters
            .iter()
            .filter(|ch| ch.start_time >= start_time && ch.start_time < end_time)
            .collect()
    }

    /// Convert time from 100-nanosecond units to seconds
    pub fn time_to_seconds(time_units: u64) -> f64 {
        time_units as f64 / 10_000_000.0
    }

    /// Convert time from seconds to 100-nanosecond units
    pub fn seconds_to_time(seconds: f64) -> u64 {
        (seconds * 10_000_000.0) as u64
    }

    /// Get chapter start times in seconds
    pub fn get_start_times_seconds(&self) -> Vec<f64> {
        self.chapters
            .iter()
            .map(|ch| Self::time_to_seconds(ch.start_time))
            .collect()
    }

    /// Find the index of the chapter containing the given time
    pub fn find_chapter_index_at_time(&self, time: u64) -> Option<usize> {
        if self.chapters.is_empty() {
            return None;
        }

        for (index, chapter) in self.chapters.iter().enumerate().rev() {
            if chapter.start_time <= time {
                return Some(index);
            }
        }
        None
    }

    /// Check if chapters are sorted by start time
    pub fn is_sorted(&self) -> bool {
        if self.chapters.len() <= 1 {
            return true;
        }

        for window in self.chapters.windows(2) {
            if window[0].start_time > window[1].start_time {
                return false;
            }
        }
        true
    }

    /// Get statistics about the chapters
    pub fn get_statistics(&self) -> ChapterStatistics {
        if self.chapters.is_empty() {
            return ChapterStatistics::default();
        }

        let chapter_count = self.chapters.len();
        let total_duration = self.total_duration().unwrap_or(0);

        let min_title_length = self
            .chapters
            .iter()
            .map(|ch| ch.title.len())
            .min()
            .unwrap_or(0);
        let max_title_length = self
            .chapters
            .iter()
            .map(|ch| ch.title.len())
            .max()
            .unwrap_or(0);
        let avg_title_length = if chapter_count > 0 {
            self.chapters.iter().map(|ch| ch.title.len()).sum::<usize>() / chapter_count
        } else {
            0
        };

        let is_sorted = self.is_sorted();
        let has_empty_titles = self.chapters.iter().any(|ch| ch.title.is_empty());

        ChapterStatistics {
            chapter_count,
            total_duration,
            min_title_length,
            max_title_length,
            avg_title_length,
            is_sorted,
            has_empty_titles,
        }
    }

    /// Validate the chapter list for consistency
    pub fn validate(&self) -> Result<(), String> {
        if !self.is_sorted() {
            return Err("Chapters are not sorted by start time".to_string());
        }

        // Check for duplicate start times
        let mut seen_times = std::collections::HashSet::new();
        for (i, chapter) in self.chapters.iter().enumerate() {
            if !seen_times.insert(chapter.start_time) {
                return Err(format!(
                    "Duplicate start time at chapter {}: {}",
                    i, chapter.start_time
                ));
            }
        }

        // Check title encoding (basic UTF-8 validation is done by String)
        for (i, chapter) in self.chapters.iter().enumerate() {
            if chapter.title.len() > 255 {
                return Err(format!(
                    "Chapter {} title too long: {} bytes",
                    i,
                    chapter.title.len()
                ));
            }
        }

        Ok(())
    }

    /// Create a new chapter list with sorted chapters
    pub fn sorted(mut self) -> Self {
        self.chapters.0.sort_by_key(|ch| ch.start_time);
        self
    }
}

/// Statistics about the chapter list
#[derive(Debug, Clone, Default)]
pub struct ChapterStatistics {
    /// Number of chapters
    pub chapter_count: usize,
    /// Total duration from first to last chapter
    pub total_duration: u64,
    /// Minimum title length in bytes
    pub min_title_length: usize,
    /// Maximum title length in bytes
    pub max_title_length: usize,
    /// Average title length in bytes
    pub avg_title_length: usize,
    /// Whether chapters are sorted by start time
    pub is_sorted: bool,
    /// Whether any chapters have empty titles
    pub has_empty_titles: bool,
}

impl TryFrom<&[u8]> for ChapterListAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_chapter_list_atom(reader)
    }
}

fn parse_chapter_list_atom<R: Read>(reader: R) -> Result<ChapterListAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != CHPL {
        return Err(anyhow!("Invalid atom type: {} (expected chpl)", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_chpl_data(&mut cursor)
}

fn parse_chpl_data<R: Read>(mut reader: R) -> Result<ChapterListAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Validate version (chpl typically uses version 1)
    if version != 1 {
        return Err(anyhow!("unsupported chpl version {}", version));
    }

    // Read reserved field (4 bytes) - should be zero
    let mut reserved = [0u8; 8];
    reader
        .read_exact(&mut reserved)
        .context("read reserved field")?;

    let mut chapters = Vec::new();

    let mut reader = BufReader::new(reader);

    // Read start time (8 bytes, big-endian)
    let mut start_time_buf = [0u8; 8];
    let mut i = -1;
    while let Ok(()) = reader.read_exact(&mut start_time_buf) {
        i += 1;
        let start_time = u64::from_be_bytes(start_time_buf);

        let mut title_buf = Vec::new();
        reader
            .read_until(0x00, &mut title_buf)
            .context(format!("read title for chapter {}", i))?;
        title_buf.pop(); // Remove trailing null byte

        let title = String::from_utf8(title_buf)
            .context(format!("invalid UTF-8 in chapter {} title", i))?;

        chapters.push(ChapterEntry { start_time, title });
    }

    Ok(ChapterListAtom {
        version,
        flags,
        chapters: ChapterEntries(chapters),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_chpl() -> ChapterListAtom {
        ChapterListAtom {
            version: 1,
            flags: [0; 3],
            chapters: ChapterEntries(vec![
                ChapterEntry {
                    start_time: 0,
                    title: "Introduction".to_string(),
                },
                ChapterEntry {
                    start_time: 300_000_000, // 30 seconds
                    title: "Chapter 1".to_string(),
                },
                ChapterEntry {
                    start_time: 600_000_000, // 60 seconds
                    title: "Chapter 2".to_string(),
                },
                ChapterEntry {
                    start_time: 1_200_000_000, // 120 seconds
                    title: "Conclusion".to_string(),
                },
            ]),
        }
    }

    #[test]
    fn test_basic_properties() {
        let chpl = create_test_chpl();

        assert_eq!(chpl.chapter_count(), 4);
        assert!(!chpl.is_empty());
        assert_eq!(chpl.total_duration(), Some(1_200_000_000)); // From 0 to 120 seconds
    }

    #[test]
    fn test_find_chapter_at_time() {
        let chpl = create_test_chpl();

        // At start
        let chapter = chpl.find_chapter_at_time(0).unwrap();
        assert_eq!(chapter.title, "Introduction");

        // In middle of first chapter
        let chapter = chpl.find_chapter_at_time(150_000_000).unwrap(); // 15 seconds
        assert_eq!(chapter.title, "Introduction");

        // At start of second chapter
        let chapter = chpl.find_chapter_at_time(300_000_000).unwrap(); // 30 seconds
        assert_eq!(chapter.title, "Chapter 1");

        // In middle of second chapter
        let chapter = chpl.find_chapter_at_time(450_000_000).unwrap(); // 45 seconds
        assert_eq!(chapter.title, "Chapter 1");

        // At last chapter
        let chapter = chpl.find_chapter_at_time(1_200_000_000).unwrap(); // 120 seconds
        assert_eq!(chapter.title, "Conclusion");

        // Beyond last chapter
        let chapter = chpl.find_chapter_at_time(1_500_000_000).unwrap(); // 150 seconds
        assert_eq!(chapter.title, "Conclusion");
    }

    #[test]
    fn test_time_conversion() {
        assert_eq!(ChapterListAtom::time_to_seconds(100_000_000), 10.0);
        assert_eq!(ChapterListAtom::time_to_seconds(50_000_000), 5.0);

        assert_eq!(ChapterListAtom::seconds_to_time(10.0), 100_000_000);
        assert_eq!(ChapterListAtom::seconds_to_time(5.5), 55_000_000);
    }

    #[test]
    fn test_get_titles() {
        let chpl = create_test_chpl();
        let titles = chpl.get_titles();

        assert_eq!(
            titles,
            vec!["Introduction", "Chapter 1", "Chapter 2", "Conclusion"]
        );
    }

    #[test]
    fn test_get_start_times_seconds() {
        let chpl = create_test_chpl();
        let times = chpl.get_start_times_seconds();

        assert_eq!(times, vec![0.0, 30.0, 60.0, 120.0]);
    }

    #[test]
    fn test_find_chapter_index_at_time() {
        let chpl = create_test_chpl();

        assert_eq!(chpl.find_chapter_index_at_time(0), Some(0));
        assert_eq!(chpl.find_chapter_index_at_time(150_000_000), Some(0)); // 15 seconds
        assert_eq!(chpl.find_chapter_index_at_time(300_000_000), Some(1)); // 30 seconds
        assert_eq!(chpl.find_chapter_index_at_time(450_000_000), Some(1)); // 45 seconds
        assert_eq!(chpl.find_chapter_index_at_time(1_200_000_000), Some(3)); // 120 seconds
        assert_eq!(chpl.find_chapter_index_at_time(1_500_000_000), Some(3)); // 150 seconds
    }

    #[test]
    fn test_chapters_in_range() {
        let chpl = create_test_chpl();

        let chapters = chpl.get_chapters_in_range(250_000_000, 700_000_000); // 25-70 seconds
        assert_eq!(chapters.len(), 2);
        assert_eq!(chapters[0].title, "Chapter 1");
        assert_eq!(chapters[1].title, "Chapter 2");
    }

    #[test]
    fn test_is_sorted() {
        let chpl = create_test_chpl();
        assert!(chpl.is_sorted());

        let unsorted_chpl = ChapterListAtom {
            version: 1,
            flags: [0; 3],
            chapters: ChapterEntries(vec![
                ChapterEntry {
                    start_time: 300_000_000,
                    title: "Chapter 1".to_string(),
                },
                ChapterEntry {
                    start_time: 0,
                    title: "Introduction".to_string(),
                },
            ]),
        };
        assert!(!unsorted_chpl.is_sorted());
    }

    #[test]
    fn test_statistics() {
        let chpl = create_test_chpl();
        let stats = chpl.get_statistics();

        assert_eq!(stats.chapter_count, 4);
        assert_eq!(stats.total_duration, 1_200_000_000);
        assert_eq!(stats.min_title_length, 9); // "Chapter 1" or "Chapter 2"
        assert_eq!(stats.max_title_length, 12); // "Introduction"
        assert!(stats.is_sorted);
        assert!(!stats.has_empty_titles);
    }

    #[test]
    fn test_validation() {
        let valid_chpl = create_test_chpl();
        assert!(valid_chpl.validate().is_ok());

        // Test unsorted chapters
        let unsorted_chpl = ChapterListAtom {
            version: 1,
            flags: [0; 3],
            chapters: ChapterEntries(vec![
                ChapterEntry {
                    start_time: 100,
                    title: "Second".to_string(),
                },
                ChapterEntry {
                    start_time: 50,
                    title: "First".to_string(),
                },
            ]),
        };
        assert!(unsorted_chpl.validate().is_err());

        // Test duplicate start times
        let duplicate_chpl = ChapterListAtom {
            version: 1,
            flags: [0; 3],
            chapters: ChapterEntries(vec![
                ChapterEntry {
                    start_time: 100,
                    title: "First".to_string(),
                },
                ChapterEntry {
                    start_time: 100,
                    title: "Duplicate".to_string(),
                },
            ]),
        };
        assert!(duplicate_chpl.validate().is_err());
    }

    #[test]
    fn test_sorted_method() {
        let unsorted_chpl = ChapterListAtom {
            version: 1,
            flags: [0; 3],
            chapters: ChapterEntries(vec![
                ChapterEntry {
                    start_time: 300_000_000,
                    title: "Chapter 1".to_string(),
                },
                ChapterEntry {
                    start_time: 0,
                    title: "Introduction".to_string(),
                },
            ]),
        };

        assert!(!unsorted_chpl.is_sorted());

        let sorted_chpl = unsorted_chpl.sorted();
        assert!(sorted_chpl.is_sorted());
        assert_eq!(sorted_chpl.chapters[0].title, "Introduction");
        assert_eq!(sorted_chpl.chapters[1].title, "Chapter 1");
    }

    #[test]
    fn test_empty_chapter_list() {
        let empty_chpl = ChapterListAtom {
            version: 1,
            flags: [0; 3],
            chapters: ChapterEntries(vec![]),
        };

        assert_eq!(empty_chpl.chapter_count(), 0);
        assert!(empty_chpl.is_empty());
        assert_eq!(empty_chpl.total_duration(), None);
        assert_eq!(empty_chpl.find_chapter_at_time(0), None);
        assert_eq!(empty_chpl.find_chapter_index_at_time(0), None);
        assert!(empty_chpl.is_sorted());
        assert!(empty_chpl.validate().is_ok());
    }
}
