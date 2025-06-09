use anyhow::{anyhow, Context};

use std::io::{Cursor, Read};

use crate::atom::util::parse_fixed_size_atom;

pub const MVHD: &[u8; 4] = b"mvhd";

#[derive(Debug, Clone)]
pub struct MovieHeaderAtom {
    /// Version of the mvhd atom format (0 or 1)
    pub version: u8,
    /// Flags for the mvhd atom (usually all zeros)
    pub flags: [u8; 3],
    /// When the movie was created (seconds since Jan 1, 1904 UTC)
    pub creation_time: u64,
    /// When the movie was last modified (seconds since Jan 1, 1904 UTC)
    pub modification_time: u64,
    /// Number of time units per second (e.g., 90000 for 90kHz)
    pub timescale: u32,
    /// Duration of the movie in timescale units
    pub duration: u64,
    /// Playback rate (1.0 = normal speed, 2.0 = double speed)
    pub rate: f32,
    /// Audio volume level (1.0 = full volume, 0.0 = muted)
    pub volume: f32,
    /// 3x3 transformation matrix for video display positioning/rotation
    pub matrix: Option<[i32; 9]>,
    /// Time when preview starts (in timescale units)
    pub preview_time: u32,
    /// Duration of the preview (in timescale units)
    pub preview_duration: u32,
    /// Time of poster frame to display when movie is not playing
    pub poster_time: u32,
    /// Start time of current selection (in timescale units)
    pub selection_time: u32,
    /// Duration of current selection (in timescale units)
    pub selection_duration: u32,
    /// Current playback time position (in timescale units)
    pub current_time: u32,
    /// ID to use for the next track added to this movie
    pub next_track_id: u32,
}

impl MovieHeaderAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_mvhd_atom(reader)
    }

    /// Convert duration to seconds using the timescale
    pub fn duration_seconds(&self) -> f64 {
        if self.timescale == 0 {
            0.0
        } else {
            self.duration as f64 / self.timescale as f64
        }
    }
}

impl TryFrom<&[u8]> for MovieHeaderAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_mvhd_atom(reader)
    }
}

fn parse_mvhd_atom<R: Read>(reader: R) -> Result<MovieHeaderAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != MVHD {
        return Err(anyhow!("Invalid atom type: {}", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_mvhd_data(&mut cursor)
}

fn parse_mvhd_data<R: Read>(mut reader: R) -> Result<MovieHeaderAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    match version {
        0 => parse_mvhd_v0(reader, flags),
        1 => parse_mvhd_v1(reader, flags),
        v => Err(anyhow!("unsupported version {v}")),
    }
}

fn parse_mvhd_v0<R: Read>(mut reader: R, flags: [u8; 3]) -> Result<MovieHeaderAtom, anyhow::Error> {
    let mut buf = [0u8; 4];

    // Creation time (32-bit)
    reader.read_exact(&mut buf).context("creation_time")?;
    let creation_time = u32::from_be_bytes(buf) as u64;

    // Modification time (32-bit)
    reader.read_exact(&mut buf).context("modification_time")?;
    let modification_time = u32::from_be_bytes(buf) as u64;

    // Timescale
    reader.read_exact(&mut buf).context("timescale")?;
    let timescale = u32::from_be_bytes(buf);

    // Duration (32-bit)
    reader.read_exact(&mut buf).context("duration")?;
    let duration = u32::from_be_bytes(buf) as u64;

    // Rate (fixed-point 16.16)
    reader.read_exact(&mut buf).context("rate")?;
    let rate_fixed = u32::from_be_bytes(buf);
    let rate = (rate_fixed as f32) / 65536.0;

    // Volume (fixed-point 8.8) - stored in 16 bits
    let mut vol_buf = [0u8; 2];
    reader.read_exact(&mut vol_buf).context("volume")?;
    let volume_fixed = u16::from_be_bytes(vol_buf);
    let volume = (volume_fixed as f32) / 256.0;

    // Reserved (10 bytes)
    let mut reserved = [0u8; 10];
    reader.read_exact(&mut reserved).context("reserved")?;

    // Matrix (9 x 32-bit values)
    let mut matrix = [0i32; 9];
    for item in &mut matrix {
        reader.read_exact(&mut buf).context("matrix")?;
        *item = i32::from_be_bytes(buf);
    }
    let matrix = if is_empty_matrix(&matrix) {
        None
    } else {
        Some(matrix)
    };

    // Preview time
    reader.read_exact(&mut buf).context("preview_time")?;
    let preview_time = u32::from_be_bytes(buf);

    // Preview duration
    reader.read_exact(&mut buf).context("preview_duration")?;
    let preview_duration = u32::from_be_bytes(buf);

    // Poster time
    reader.read_exact(&mut buf).context("poster_time")?;
    let poster_time = u32::from_be_bytes(buf);

    // Selection time
    reader.read_exact(&mut buf).context("selection_time")?;
    let selection_time = u32::from_be_bytes(buf);

    // Selection duration
    reader.read_exact(&mut buf).context("selection_duration")?;
    let selection_duration = u32::from_be_bytes(buf);

    // Current time
    reader.read_exact(&mut buf).context("current_time")?;
    let current_time = u32::from_be_bytes(buf);

    // Next track ID
    reader.read_exact(&mut buf).context("next_track_id")?;
    let next_track_id = u32::from_be_bytes(buf);

    Ok(MovieHeaderAtom {
        version: 0,
        flags,
        creation_time,
        modification_time,
        timescale,
        duration,
        rate,
        volume,
        matrix,
        preview_time,
        preview_duration,
        poster_time,
        selection_time,
        selection_duration,
        current_time,
        next_track_id,
    })
}

fn parse_mvhd_v1<R: Read>(mut reader: R, flags: [u8; 3]) -> Result<MovieHeaderAtom, anyhow::Error> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Creation time (64-bit)
    reader.read_exact(&mut buf8).context("creation time")?;
    let creation_time = u64::from_be_bytes(buf8);

    // Modification time (64-bit)
    reader.read_exact(&mut buf8).context("modification time")?;
    let modification_time = u64::from_be_bytes(buf8);

    // Timescale
    reader.read_exact(&mut buf4).context("timescale")?;
    let timescale = u32::from_be_bytes(buf4);

    // Duration (64-bit)
    reader.read_exact(&mut buf8).context("duration")?;
    let duration = u64::from_be_bytes(buf8);

    // Rate (fixed-point 16.16)
    reader.read_exact(&mut buf4).context("rate")?;
    let rate_fixed = u32::from_be_bytes(buf4);
    let rate = (rate_fixed as f32) / 65536.0;

    // Volume (fixed-point 8.8) - stored in 16 bits
    let mut vol_buf = [0u8; 2];
    reader.read_exact(&mut vol_buf).context("volume")?;
    let volume_fixed = u16::from_be_bytes(vol_buf);
    let volume = (volume_fixed as f32) / 256.0;

    // Reserved (10 bytes)
    let mut reserved = [0u8; 10];
    reader.read_exact(&mut reserved).context("reserved")?;

    // Matrix (9 x 32-bit values)
    let mut matrix = [0i32; 9];
    for item in &mut matrix {
        reader.read_exact(&mut buf4).context("matrix")?;
        *item = i32::from_be_bytes(buf4);
    }
    let matrix = if is_empty_matrix(&matrix) {
        None
    } else {
        Some(matrix)
    };

    // Preview time
    reader.read_exact(&mut buf4).context("preview_time")?;
    let preview_time = u32::from_be_bytes(buf4);

    // Preview duration
    reader.read_exact(&mut buf4).context("preview_duration")?;
    let preview_duration = u32::from_be_bytes(buf4);

    // Poster time
    reader.read_exact(&mut buf4).context("poster_time")?;
    let poster_time = u32::from_be_bytes(buf4);

    // Selection time
    reader.read_exact(&mut buf4).context("selection_time")?;
    let selection_time = u32::from_be_bytes(buf4);

    // Selection duration
    reader.read_exact(&mut buf4).context("selection_duration")?;
    let selection_duration = u32::from_be_bytes(buf4);

    // Current time
    reader.read_exact(&mut buf4).context("current_time")?;
    let current_time = u32::from_be_bytes(buf4);

    // Next track ID
    reader.read_exact(&mut buf4).context("next_track_id")?;
    let next_track_id = u32::from_be_bytes(buf4);

    Ok(MovieHeaderAtom {
        version: 1,
        flags,
        creation_time,
        modification_time,
        timescale,
        duration,
        rate,
        volume,
        matrix,
        preview_time,
        preview_duration,
        poster_time,
        selection_time,
        selection_duration,
        current_time,
        next_track_id,
    })
}

fn is_empty_matrix(matrix: &[i32; 9]) -> bool {
    let empty = [0; 9];
    let identity = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
    matrix == &empty || matrix == &identity
}
