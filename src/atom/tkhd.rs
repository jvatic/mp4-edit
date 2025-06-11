use anyhow::{anyhow, Context};

use futures_io::AsyncRead;
use std::io::Read;

use crate::{
    atom::{util::async_to_sync_read, FourCC},
    parser::Parse,
};

pub const TKHD: &[u8; 4] = b"tkhd";

#[derive(Debug, Clone)]
pub struct TrackHeaderAtom {
    /// Version of the tkhd atom format (0 or 1)
    pub version: u8,
    /// Flags for the tkhd atom (bit flags for track properties)
    pub flags: [u8; 3],
    /// When the track was created (seconds since Jan 1, 1904 UTC)
    pub creation_time: u64,
    /// When the track was last modified (seconds since Jan 1, 1904 UTC)
    pub modification_time: u64,
    /// Unique identifier for this track within the movie
    pub track_id: u32,
    /// Duration of the track in movie timescale units
    pub duration: u64,
    /// Playback layer (lower numbers are closer to viewer)
    pub layer: i16,
    /// Audio balance or stereo balance (-1.0 = left, 0.0 = center, 1.0 = right)
    pub alternate_group: i16,
    /// Audio volume level (1.0 = full volume, 0.0 = muted)
    pub volume: f32,
    /// 3x3 transformation matrix for video display positioning/rotation
    pub matrix: Option<[i32; 9]>,
    /// Track width in pixels (fixed-point 16.16)
    pub width: f32,
    /// Track height in pixels (fixed-point 16.16)
    pub height: f32,
}

impl Parse for TrackHeaderAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, anyhow::Error> {
        if atom_type != TKHD {
            return Err(anyhow!("Invalid atom type: {}", atom_type));
        }
        parse_tkhd_data(async_to_sync_read(reader).await?)
    }
}

fn parse_tkhd_data<R: Read>(mut reader: R) -> Result<TrackHeaderAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    match version {
        0 => parse_tkhd_v0(reader, flags),
        1 => parse_tkhd_v1(reader, flags),
        v => Err(anyhow!("unsupported version {v}")),
    }
}

fn parse_tkhd_v0<R: Read>(mut reader: R, flags: [u8; 3]) -> Result<TrackHeaderAtom, anyhow::Error> {
    let mut buf = [0u8; 4];

    // Creation time (32-bit)
    reader.read_exact(&mut buf).context("creation_time")?;
    let creation_time = u32::from_be_bytes(buf) as u64;

    // Modification time (32-bit)
    reader.read_exact(&mut buf).context("modification_time")?;
    let modification_time = u32::from_be_bytes(buf) as u64;

    // Track ID
    reader.read_exact(&mut buf).context("track_id")?;
    let track_id = u32::from_be_bytes(buf);

    // Reserved (4 bytes)
    reader.read_exact(&mut buf).context("reserved1")?;

    // Duration (32-bit)
    reader.read_exact(&mut buf).context("duration")?;
    let duration = u32::from_be_bytes(buf) as u64;

    // Reserved (8 bytes)
    let mut reserved = [0u8; 8];
    reader.read_exact(&mut reserved).context("reserved2")?;

    // Layer (16-bit signed)
    let mut layer_buf = [0u8; 2];
    reader.read_exact(&mut layer_buf).context("layer")?;
    let layer = i16::from_be_bytes(layer_buf);

    // Alternate group (16-bit signed)
    let mut alt_buf = [0u8; 2];
    reader.read_exact(&mut alt_buf).context("alternate_group")?;
    let alternate_group = i16::from_be_bytes(alt_buf);

    // Volume (fixed-point 8.8) - stored in 16 bits
    let mut vol_buf = [0u8; 2];
    reader.read_exact(&mut vol_buf).context("volume")?;
    let volume_fixed = u16::from_be_bytes(vol_buf);
    let volume = (volume_fixed as f32) / 256.0;

    // Reserved (2 bytes)
    let mut reserved2 = [0u8; 2];
    reader.read_exact(&mut reserved2).context("reserved3")?;

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

    // Width (fixed-point 16.16)
    reader.read_exact(&mut buf).context("width")?;
    let width_fixed = u32::from_be_bytes(buf);
    let width = (width_fixed as f32) / 65536.0;

    // Height (fixed-point 16.16)
    reader.read_exact(&mut buf).context("height")?;
    let height_fixed = u32::from_be_bytes(buf);
    let height = (height_fixed as f32) / 65536.0;

    Ok(TrackHeaderAtom {
        version: 0,
        flags,
        creation_time,
        modification_time,
        track_id,
        duration,
        layer,
        alternate_group,
        volume,
        matrix,
        width,
        height,
    })
}

fn parse_tkhd_v1<R: Read>(mut reader: R, flags: [u8; 3]) -> Result<TrackHeaderAtom, anyhow::Error> {
    let mut buf4 = [0u8; 4];
    let mut buf8 = [0u8; 8];

    // Creation time (64-bit)
    reader.read_exact(&mut buf8).context("creation_time")?;
    let creation_time = u64::from_be_bytes(buf8);

    // Modification time (64-bit)
    reader.read_exact(&mut buf8).context("modification_time")?;
    let modification_time = u64::from_be_bytes(buf8);

    // Track ID
    reader.read_exact(&mut buf4).context("track_id")?;
    let track_id = u32::from_be_bytes(buf4);

    // Reserved (4 bytes)
    reader.read_exact(&mut buf4).context("reserved1")?;

    // Duration (64-bit)
    reader.read_exact(&mut buf8).context("duration")?;
    let duration = u64::from_be_bytes(buf8);

    // Reserved (8 bytes)
    let mut reserved = [0u8; 8];
    reader.read_exact(&mut reserved).context("reserved2")?;

    // Layer (16-bit signed)
    let mut layer_buf = [0u8; 2];
    reader.read_exact(&mut layer_buf).context("layer")?;
    let layer = i16::from_be_bytes(layer_buf);

    // Alternate group (16-bit signed)
    let mut alt_buf = [0u8; 2];
    reader.read_exact(&mut alt_buf).context("alternate_group")?;
    let alternate_group = i16::from_be_bytes(alt_buf);

    // Volume (fixed-point 8.8) - stored in 16 bits
    let mut vol_buf = [0u8; 2];
    reader.read_exact(&mut vol_buf).context("volume")?;
    let volume_fixed = u16::from_be_bytes(vol_buf);
    let volume = (volume_fixed as f32) / 256.0;

    // Reserved (2 bytes)
    let mut reserved2 = [0u8; 2];
    reader.read_exact(&mut reserved2).context("reserved3")?;

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

    // Width (fixed-point 16.16)
    reader.read_exact(&mut buf4).context("width")?;
    let width_fixed = u32::from_be_bytes(buf4);
    let width = (width_fixed as f32) / 65536.0;

    // Height (fixed-point 16.16)
    reader.read_exact(&mut buf4).context("height")?;
    let height_fixed = u32::from_be_bytes(buf4);
    let height = (height_fixed as f32) / 65536.0;

    Ok(TrackHeaderAtom {
        version: 1,
        flags,
        creation_time,
        modification_time,
        track_id,
        duration,
        layer,
        alternate_group,
        volume,
        matrix,
        width,
        height,
    })
}

fn is_empty_matrix(matrix: &[i32; 9]) -> bool {
    let empty = [0; 9];
    let identity = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
    matrix == &empty || matrix == &identity
}
