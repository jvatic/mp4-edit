use anyhow::{anyhow, Context};

use std::io::{Cursor, Read};

use crate::atom::util::parse_fixed_size_atom;

pub const ELST: &[u8; 4] = b"elst";

#[derive(Debug, Clone)]
pub struct EditListAtom {
    /// Version of the elst atom format (0 or 1)
    pub version: u8,
    /// Flags for the elst atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of edit entries
    pub entries: Vec<EditEntry>,
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

impl EditListAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_elst_atom(reader)
    }
}

impl EditEntry {
    /// Check if this is an empty edit (no media displayed)
    pub fn is_empty_edit(&self) -> bool {
        self.media_time == -1
    }

    /// Check if this edit plays at normal speed
    pub fn is_normal_speed(&self) -> bool {
        (self.media_rate - 1.0).abs() < f32::EPSILON
    }

    /// Check if this edit is a dwell edit (media_rate == 0.0)
    pub fn is_dwell_edit(&self) -> bool {
        self.media_rate.abs() < f32::EPSILON
    }
}

impl TryFrom<&[u8]> for EditListAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_elst_atom(reader)
    }
}

fn parse_elst_atom<R: Read>(reader: R) -> Result<EditListAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != ELST {
        return Err(anyhow!("Invalid atom type: {}", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_elst_data(&mut cursor)
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
