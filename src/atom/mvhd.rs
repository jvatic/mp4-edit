use anyhow::{anyhow, Context};

use bon::Builder;
use futures_io::AsyncRead;
use std::{io::Read, time::Duration};

use crate::{
    atom::{
        util::{async_to_sync_read, mp4_timestamp_now, scaled_duration, unscaled_duration},
        FourCC,
    },
    parser::Parse,
    writer::SerializeAtom,
    ParseError,
};

pub const MVHD: &[u8; 4] = b"mvhd";

#[derive(Debug, Clone, Builder)]
pub struct MovieHeaderAtom {
    /// Version of the mvhd atom format (0 or 1)
    #[builder(default = 0)]
    pub version: u8,
    /// Flags for the mvhd atom (usually all zeros)
    #[builder(default = [0u8; 3])]
    pub flags: [u8; 3],
    /// When the movie was created (seconds since Jan 1, 1904 UTC)
    #[builder(default = mp4_timestamp_now())]
    pub creation_time: u64,
    /// When the movie was last modified (seconds since Jan 1, 1904 UTC)
    #[builder(default = mp4_timestamp_now())]
    pub modification_time: u64,
    /// Number of time units per second (e.g., 90000 for 90kHz)
    pub timescale: u32,
    /// Duration of the movie in timescale units
    pub duration: u64,
    /// Playback rate (1.0 = normal speed, 2.0 = double speed)
    #[builder(default = 1.0)]
    pub rate: f32,
    /// Audio volume level (1.0 = full volume, 0.0 = muted)
    #[builder(default = 1.0)]
    pub volume: f32,
    /// 3x3 transformation matrix for video display positioning/rotation
    pub matrix: Option<[i32; 9]>,
    /// Time when preview starts (in timescale units)
    #[builder(default = 0)]
    pub preview_time: u32,
    /// Duration of the preview (in timescale units)
    #[builder(default = 0)]
    pub preview_duration: u32,
    /// Time of poster frame to display when movie is not playing
    #[builder(default = 0)]
    pub poster_time: u32,
    /// Start time of current selection (in timescale units)
    #[builder(default = 0)]
    pub selection_time: u32,
    /// Duration of current selection (in timescale units)
    #[builder(default = 0)]
    pub selection_duration: u32,
    /// Current playback time position (in timescale units)
    #[builder(default = 0)]
    pub current_time: u32,
    /// ID to use for the next track added to this movie
    pub next_track_id: u32,
}

impl MovieHeaderAtom {
    pub fn update_duration<F>(&mut self, mut closure: F) -> &mut Self
    where
        F: FnMut(Duration) -> Duration,
    {
        self.duration = scaled_duration(
            closure(unscaled_duration(self.duration, u64::from(self.timescale))),
            u64::from(self.timescale),
        );
        self
    }

    pub fn duration(&self) -> Duration {
        unscaled_duration(self.duration, u64::from(self.timescale))
    }
}

impl Parse for MovieHeaderAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != MVHD {
            return Err(ParseError::new_unexpected_atom(atom_type, MVHD));
        }
        parse_mvhd_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
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
    let creation_time = u64::from(u32::from_be_bytes(buf));

    // Modification time (32-bit)
    reader.read_exact(&mut buf).context("modification_time")?;
    let modification_time = u64::from(u32::from_be_bytes(buf));

    // Timescale
    reader.read_exact(&mut buf).context("timescale")?;
    let timescale = u32::from_be_bytes(buf);

    // Duration (32-bit)
    reader.read_exact(&mut buf).context("duration")?;
    let duration = u64::from(u32::from_be_bytes(buf));

    // Rate (fixed-point 16.16)
    reader.read_exact(&mut buf).context("rate")?;
    let rate_fixed = u32::from_be_bytes(buf);
    let rate = (rate_fixed as f32) / 65536.0;

    // Volume (fixed-point 8.8) - stored in 16 bits
    let mut vol_buf = [0u8; 2];
    reader.read_exact(&mut vol_buf).context("volume")?;
    let volume_fixed = u16::from_be_bytes(vol_buf);
    let volume = f32::from(volume_fixed) / 256.0;

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
    let volume = f32::from(volume_fixed) / 256.0;

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

impl SerializeAtom for MovieHeaderAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*MVHD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Determine version based on whether values fit in 32-bit
        let needs_64_bit = self.creation_time > u64::from(u32::MAX)
            || self.modification_time > u64::from(u32::MAX)
            || self.duration > u64::from(u32::MAX);

        let version = i32::from(needs_64_bit);

        // Version and flags (4 bytes)
        let version_flags = (version as u32) << 24
            | u32::from(self.flags[0]) << 16
            | u32::from(self.flags[1]) << 8
            | u32::from(self.flags[2]);
        data.extend_from_slice(&version_flags.to_be_bytes());

        match version {
            0 => {
                // Creation time (32-bit)
                data.extend_from_slice(
                    &(u32::try_from(self.creation_time).expect("creation time should fit in u32"))
                        .to_be_bytes(),
                );
                // Modification time (32-bit)
                data.extend_from_slice(
                    &(u32::try_from(self.modification_time)
                        .expect("modification time should fit in u32"))
                    .to_be_bytes(),
                );
                // Timescale (32-bit)
                data.extend_from_slice(&self.timescale.to_be_bytes());
                // Duration (32-bit)
                data.extend_from_slice(
                    &(u32::try_from(self.duration).expect("duration should fit in u32"))
                        .to_be_bytes(),
                );
            }
            1 => {
                // Creation time (64-bit)
                data.extend_from_slice(&self.creation_time.to_be_bytes());
                // Modification time (64-bit)
                data.extend_from_slice(&self.modification_time.to_be_bytes());
                // Timescale (32-bit)
                data.extend_from_slice(&self.timescale.to_be_bytes());
                // Duration (64-bit)
                data.extend_from_slice(&self.duration.to_be_bytes());
            }
            _ => {} // Should not happen due to validation during parsing
        }

        // Rate (fixed-point 16.16)
        let rate_fixed = (self.rate * 65536.0) as u32;
        data.extend_from_slice(&rate_fixed.to_be_bytes());

        // Volume (fixed-point 8.8)
        let volume_fixed = (self.volume * 256.0) as u16;
        data.extend_from_slice(&volume_fixed.to_be_bytes());

        // Reserved (10 bytes)
        data.extend_from_slice(&[0u8; 10]);

        // Matrix (9 x 32-bit values)
        let matrix = self
            .matrix
            .unwrap_or([0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000]);
        for value in matrix {
            data.extend_from_slice(&value.to_be_bytes());
        }

        // Preview time
        data.extend_from_slice(&self.preview_time.to_be_bytes());
        // Preview duration
        data.extend_from_slice(&self.preview_duration.to_be_bytes());
        // Poster time
        data.extend_from_slice(&self.poster_time.to_be_bytes());
        // Selection time
        data.extend_from_slice(&self.selection_time.to_be_bytes());
        // Selection duration
        data.extend_from_slice(&self.selection_duration.to_be_bytes());
        // Current time
        data.extend_from_slice(&self.current_time.to_be_bytes());
        // Next track ID
        data.extend_from_slice(&self.next_track_id.to_be_bytes());

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available mvhd test data files
    #[test]
    fn test_mvhd_roundtrip() {
        test_atom_roundtrip_sync::<MovieHeaderAtom>(MVHD);
    }
}
