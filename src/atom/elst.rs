use std::{io::Read, time::Duration};

use anyhow::{anyhow, Context};
use bon::bon;
use futures_io::AsyncRead;

use crate::{
    atom::{
        util::{async_to_sync_read, time::scaled_duration},
        FourCC,
    },
    parser::Parse,
    writer::SerializeAtom,
    ParseError,
};

pub const ELST: &[u8; 4] = b"elst";

#[derive(Default, Debug, Clone)]
pub struct EditListAtom {
    /// Version of the elst atom format (0 or 1)
    pub version: u8,
    /// Flags for the elst atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of edit entries
    pub entries: Vec<EditEntry>,
}

impl EditListAtom {
    pub fn new(entries: impl Into<Vec<EditEntry>>) -> Self {
        Self {
            entries: entries.into(),
            ..Default::default()
        }
    }
}

pub struct SegmentDuration {
    duration: Duration,
    movie_timescale: u32,
}

#[bon]
impl SegmentDuration {
    #[builder]
    pub fn new(duration: Duration, movie_timescale: u32) -> Self {
        Self {
            duration,
            movie_timescale,
        }
    }

    pub fn scaled(&self) -> u64 {
        scaled_duration(self.duration, self.movie_timescale as u64)
    }
}

#[derive(Default)]
pub struct MediaDuration {
    duration: Duration,
    media_timescale: u32,
}

#[bon]
impl MediaDuration {
    #[builder]
    pub fn new(duration: Duration, media_timescale: u32) -> Self {
        Self {
            duration,
            media_timescale,
        }
    }

    pub fn scaled(&self) -> i64 {
        scaled_duration(self.duration, self.media_timescale as u64) as i64
    }
}

#[derive(Debug, Clone)]
pub struct EditEntry {
    /// Duration of this edit segment (in movie timescale units)
    pub segment_duration: u64,
    /// Starting time within the media (in media timescale units)
    /// -1 indicates an empty edit (no media displayed)
    pub media_time: i64,
    /// Playback rate for this segment (1.0 = normal speed)
    pub media_rate: f32,
}

#[bon]
impl EditEntry {
    #[builder]
    pub fn new(
        segment_duration: SegmentDuration,
        #[builder(default = Default::default())] media_time: MediaDuration,
        #[builder(default = 1.0)] media_rate: f32,
    ) -> Self {
        Self {
            segment_duration: segment_duration.scaled(),
            media_time: media_time.scaled(),
            media_rate,
        }
    }
}

impl Parse for EditListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != ELST {
            return Err(ParseError::new_unexpected_atom(atom_type, ELST));
        }
        parse_elst_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
}

impl SerializeAtom for EditListAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*ELST)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Entry count (4 bytes, big-endian)
        data.extend_from_slice(&(self.entries.len() as u32).to_be_bytes());

        // Entries
        for entry in self.entries {
            match self.version {
                0 => {
                    // Version 0: 32-bit fields
                    data.extend_from_slice(&(entry.segment_duration as u32).to_be_bytes());
                    data.extend_from_slice(&(entry.media_time as i32).to_be_bytes());
                    // Convert f32 to fixed-point 16.16
                    let rate_fixed = (entry.media_rate * 65536.0) as u32;
                    data.extend_from_slice(&rate_fixed.to_be_bytes());
                }
                1 => {
                    // Version 1: 64-bit fields for duration and time
                    data.extend_from_slice(&entry.segment_duration.to_be_bytes());
                    data.extend_from_slice(&entry.media_time.to_be_bytes());
                    // Convert f32 to fixed-point 16.16
                    let rate_fixed = (entry.media_rate * 65536.0) as u32;
                    data.extend_from_slice(&rate_fixed.to_be_bytes());
                }
                _ => {
                    // Fallback to version 0 format for unknown versions
                    data.extend_from_slice(&(entry.segment_duration as u32).to_be_bytes());
                    data.extend_from_slice(&(entry.media_time as i32).to_be_bytes());
                    let rate_fixed = (entry.media_rate * 65536.0) as u32;
                    data.extend_from_slice(&rate_fixed.to_be_bytes());
                }
            }
        }

        data
    }
}

fn parse_elst_data<R: Read>(mut reader: R) -> Result<EditListAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Read entry count
    let mut count_buf = [0u8; 4];
    reader
        .read_exact(&mut count_buf)
        .context("read entry count")?;
    let entry_count = u32::from_be_bytes(count_buf);

    // Validate entry count
    if entry_count > 65535 {
        return Err(anyhow!("Too many edit entries: {}", entry_count));
    }

    let entries = match version {
        0 => parse_elst_entries_v0(&mut reader, entry_count as usize)?,
        1 => parse_elst_entries_v1(&mut reader, entry_count as usize)?,
        v => return Err(anyhow!("unsupported version {v}")),
    };

    Ok(EditListAtom {
        version,
        flags,
        entries,
    })
}

fn parse_elst_entries_v0<R: Read>(
    mut reader: R,
    entry_count: usize,
) -> Result<Vec<EditEntry>, anyhow::Error> {
    let mut entries = Vec::with_capacity(entry_count);
    let mut buf = [0u8; 4];

    for i in 0..entry_count {
        // Segment duration (32-bit)
        reader
            .read_exact(&mut buf)
            .context(format!("read segment_duration for entry {}", i))?;
        let segment_duration = u32::from_be_bytes(buf) as u64;

        // Media time (32-bit signed)
        reader
            .read_exact(&mut buf)
            .context(format!("read media_time for entry {}", i))?;
        let media_time = i32::from_be_bytes(buf) as i64;

        // Media rate (fixed-point 16.16)
        reader
            .read_exact(&mut buf)
            .context(format!("read media_rate for entry {}", i))?;
        let rate_fixed = u32::from_be_bytes(buf);
        let media_rate = (rate_fixed as f32) / 65536.0;

        entries.push(EditEntry {
            segment_duration,
            media_time,
            media_rate,
        });
    }

    Ok(entries)
}

fn parse_elst_entries_v1<R: Read>(
    mut reader: R,
    entry_count: usize,
) -> Result<Vec<EditEntry>, anyhow::Error> {
    let mut entries = Vec::with_capacity(entry_count);
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    for i in 0..entry_count {
        // Segment duration (64-bit)
        reader
            .read_exact(&mut buf8)
            .context(format!("read segment_duration for entry {}", i))?;
        let segment_duration = u64::from_be_bytes(buf8);

        // Media time (64-bit signed)
        reader
            .read_exact(&mut buf8)
            .context(format!("read media_time for entry {}", i))?;
        let media_time = i64::from_be_bytes(buf8);

        // Media rate (fixed-point 16.16)
        reader
            .read_exact(&mut buf4)
            .context(format!("read media_rate for entry {}", i))?;
        let rate_fixed = u32::from_be_bytes(buf4);
        let media_rate = (rate_fixed as f32) / 65536.0;

        entries.push(EditEntry {
            segment_duration,
            media_time,
            media_rate,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available elst test data files
    #[test]
    fn test_elst_roundtrip() {
        test_atom_roundtrip_sync::<EditListAtom>(ELST);
    }
}
