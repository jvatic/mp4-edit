use anyhow::{anyhow, Context};
use derive_more::Display;
use std::io::{Cursor, Read};

use crate::atom::util::{parse_fixed_size_atom, FourCC};

pub const STSD: &[u8; 4] = b"stsd";

// Common sample entry types
pub const SAMPLE_ENTRY_AVC1: &[u8; 4] = b"avc1"; // H.264 video
pub const SAMPLE_ENTRY_HVC1: &[u8; 4] = b"hvc1"; // H.265/HEVC video
pub const SAMPLE_ENTRY_MP4A: &[u8; 4] = b"mp4a"; // AAC audio
pub const SAMPLE_ENTRY_MP4V: &[u8; 4] = b"mp4v"; // MPEG-4 video
pub const SAMPLE_ENTRY_TX3G: &[u8; 4] = b"tx3g"; // 3GPP text
pub const SAMPLE_ENTRY_WVTT: &[u8; 4] = b"wvtt"; // WebVTT text
pub const SAMPLE_ENTRY_STPP: &[u8; 4] = b"stpp"; // Subtitle
pub const SAMPLE_ENTRY_AAVD: &[u8; 4] = b"aavd"; // Apple Audio Video Data

#[derive(Debug, Clone, Display, PartialEq)]
#[display("{}", self.as_str())]
pub enum SampleEntryType {
    /// H.264/AVC video
    Avc1,
    /// H.265/HEVC video
    Hvc1,
    /// AAC audio
    Mp4a,
    /// MPEG-4 video
    Mp4v,
    /// 3GPP text
    Tx3g,
    /// WebVTT text
    Wvtt,
    /// Subtitle
    Stpp,
    /// Apple Audio Video Data
    Aavd,
    /// Unknown sample entry type
    Unknown(FourCC),
}

impl SampleEntryType {
    pub fn from_bytes(bytes: &[u8; 4]) -> Self {
        match bytes {
            SAMPLE_ENTRY_AVC1 => SampleEntryType::Avc1,
            SAMPLE_ENTRY_HVC1 => SampleEntryType::Hvc1,
            SAMPLE_ENTRY_MP4A => SampleEntryType::Mp4a,
            SAMPLE_ENTRY_MP4V => SampleEntryType::Mp4v,
            SAMPLE_ENTRY_TX3G => SampleEntryType::Tx3g,
            SAMPLE_ENTRY_WVTT => SampleEntryType::Wvtt,
            SAMPLE_ENTRY_STPP => SampleEntryType::Stpp,
            SAMPLE_ENTRY_AAVD => SampleEntryType::Aavd,
            _ => SampleEntryType::Unknown(FourCC(*bytes)),
        }
    }

    pub fn as_bytes(&self) -> &[u8; 4] {
        match self {
            SampleEntryType::Avc1 => SAMPLE_ENTRY_AVC1,
            SampleEntryType::Hvc1 => SAMPLE_ENTRY_HVC1,
            SampleEntryType::Mp4a => SAMPLE_ENTRY_MP4A,
            SampleEntryType::Mp4v => SAMPLE_ENTRY_MP4V,
            SampleEntryType::Tx3g => SAMPLE_ENTRY_TX3G,
            SampleEntryType::Wvtt => SAMPLE_ENTRY_WVTT,
            SampleEntryType::Stpp => SAMPLE_ENTRY_STPP,
            SampleEntryType::Aavd => SAMPLE_ENTRY_AAVD,
            SampleEntryType::Unknown(bytes) => &bytes.0,
        }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(self.as_bytes()).unwrap_or("????")
    }

    pub fn is_video(&self) -> bool {
        matches!(
            self,
            SampleEntryType::Avc1
                | SampleEntryType::Hvc1
                | SampleEntryType::Mp4v
                | SampleEntryType::Aavd
        )
    }

    pub fn is_audio(&self) -> bool {
        matches!(self, SampleEntryType::Mp4a)
    }

    pub fn is_text(&self) -> bool {
        matches!(
            self,
            SampleEntryType::Tx3g | SampleEntryType::Wvtt | SampleEntryType::Stpp
        )
    }
}

#[derive(Debug, Clone)]
pub enum SampleEntryData {
    Video(VideoSampleEntry),
    Audio(AudioSampleEntry),
    Other(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct SampleEntry {
    /// Sample entry type (4CC code)
    pub entry_type: SampleEntryType,
    /// Data reference index
    pub data_reference_index: u16,
    /// Raw sample entry data (codec-specific)
    pub data: SampleEntryData,
}

#[derive(Debug, Clone)]
pub struct VideoSampleEntry {
    /// Version (usually 0)
    pub version: u16,
    /// Revision level (usually 0)
    pub revision_level: u16,
    /// Vendor identifier
    pub vendor: [u8; 4],
    /// Temporal quality (0-1023)
    pub temporal_quality: u32,
    /// Spatial quality (0-1024)
    pub spatial_quality: u32,
    /// Width in pixels
    pub width: u16,
    /// Height in pixels
    pub height: u16,
    /// Horizontal resolution (pixels per inch, 16.16 fixed point)
    pub horizresolution: f32,
    /// Vertical resolution (pixels per inch, 16.16 fixed point)
    pub vertresolution: f32,
    /// Reserved (should be 0)
    pub entry_data_size: u32,
    /// Frame count (usually 1)
    pub frame_count: u16,
    /// Compressor name (32 bytes, Pascal string)
    pub compressor_name: String,
    /// Bit depth (usually 24)
    pub depth: u16,
    /// Color table ID (usually -1)
    pub color_table_id: i16,
    /// Extension data (codec-specific atoms)
    pub extensions: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct AudioSampleEntry {
    /// Version (usually 0)
    pub version: u16,
    /// Revision level (usually 0)
    pub revision_level: u16,
    /// Vendor identifier
    pub vendor: [u8; 4],
    /// Number of audio channels
    pub channel_count: u16,
    /// Sample size in bits
    pub sample_size: u16,
    /// Compression ID (usually 0)
    pub compression_id: u16,
    /// Packet size (usually 0)
    pub packet_size: u16,
    /// Sample rate (16.16 fixed point)
    pub sample_rate: f32,
    /// Extension data (codec-specific atoms)
    pub extensions: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SampleDescriptionTableAtom {
    /// Version of the stsd atom format (0)
    pub version: u8,
    /// Flags for the stsd atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of sample entries
    pub entries: Vec<SampleEntry>,
}

impl SampleDescriptionTableAtom {
    pub fn parse<R: Read>(reader: R) -> Result<Self, anyhow::Error> {
        parse_stsd_atom(reader)
    }

    /// Get the number of sample entries
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the first sample entry (most common case)
    pub fn first_entry(&self) -> Option<&SampleEntry> {
        self.entries.first()
    }

    /// Get all video sample entries
    pub fn video_entries(&self) -> Vec<&SampleEntry> {
        self.entries
            .iter()
            .filter(|e| e.entry_type.is_video())
            .collect()
    }

    /// Get all audio sample entries
    pub fn audio_entries(&self) -> Vec<&SampleEntry> {
        self.entries
            .iter()
            .filter(|e| e.entry_type.is_audio())
            .collect()
    }

    /// Get all text sample entries
    pub fn text_entries(&self) -> Vec<&SampleEntry> {
        self.entries
            .iter()
            .filter(|e| e.entry_type.is_text())
            .collect()
    }

    /// Check if this table contains video samples
    pub fn has_video(&self) -> bool {
        self.entries.iter().any(|e| e.entry_type.is_video())
    }

    /// Check if this table contains audio samples
    pub fn has_audio(&self) -> bool {
        self.entries.iter().any(|e| e.entry_type.is_audio())
    }

    /// Check if this table contains text samples
    pub fn has_text(&self) -> bool {
        self.entries.iter().any(|e| e.entry_type.is_text())
    }
}

impl SampleEntry {
    /// Get the sample entry type as a 4-character string
    pub fn type_string(&self) -> String {
        let bytes = self.entry_type.as_bytes();
        std::str::from_utf8(bytes).unwrap_or("????").to_string()
    }
}

impl VideoSampleEntry {
    /// Get the video dimensions as a tuple
    pub fn dimensions(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    /// Get the aspect ratio
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    /// Check if this is high definition video (720p or higher)
    pub fn is_hd(&self) -> bool {
        self.height >= 720
    }

    /// Check if this is 4K video
    pub fn is_4k(&self) -> bool {
        self.width >= 3840 || self.height >= 2160
    }
}

impl AudioSampleEntry {
    /// Check if this is stereo audio
    pub fn is_stereo(&self) -> bool {
        self.channel_count == 2
    }

    /// Check if this is mono audio
    pub fn is_mono(&self) -> bool {
        self.channel_count == 1
    }

    /// Get sample rate as Hz
    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate as u32
    }
}

impl TryFrom<&[u8]> for SampleDescriptionTableAtom {
    type Error = anyhow::Error;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        let reader = Cursor::new(data);
        parse_stsd_atom(reader)
    }
}

fn parse_stsd_atom<R: Read>(reader: R) -> Result<SampleDescriptionTableAtom, anyhow::Error> {
    let (atom_type, data) = parse_fixed_size_atom(reader)?;
    if atom_type != STSD {
        return Err(anyhow!("Invalid atom type: {}", atom_type));
    }
    let mut cursor = Cursor::new(data);
    parse_stsd_data(&mut cursor)
}

fn parse_stsd_data<R: Read>(mut reader: R) -> Result<SampleDescriptionTableAtom, anyhow::Error> {
    // Read version and flags (4 bytes)
    let mut version_flags = [0u8; 4];
    reader
        .read_exact(&mut version_flags)
        .context("read version and flags")?;

    let version = version_flags[0];
    let flags = [version_flags[1], version_flags[2], version_flags[3]];

    // Validate version
    if version != 0 {
        return Err(anyhow!("unsupported version {}", version));
    }

    // Read entry count
    let mut count_buf = [0u8; 4];
    reader
        .read_exact(&mut count_buf)
        .context("read entry count")?;
    let entry_count = u32::from_be_bytes(count_buf);

    // Validate entry count
    if entry_count > 1000 {
        return Err(anyhow!("Too many sample entries: {}", entry_count));
    }

    let mut entries = Vec::with_capacity(entry_count as usize);

    for i in 0..entry_count {
        let entry = parse_sample_entry(&mut reader).context(format!("parse sample entry {}", i))?;
        entries.push(entry);
    }

    Ok(SampleDescriptionTableAtom {
        version,
        flags,
        entries,
    })
}

fn parse_sample_entry<R: Read>(mut reader: R) -> Result<SampleEntry, anyhow::Error> {
    // Read sample entry size (4 bytes)
    let mut size_buf = [0u8; 4];
    reader
        .read_exact(&mut size_buf)
        .context("read sample entry size")?;
    let size = u32::from_be_bytes(size_buf);

    if size < 16 {
        return Err(anyhow!("Invalid sample entry size: {}", size));
    }

    // Read sample entry type (4 bytes)
    let mut type_buf = [0u8; 4];
    reader
        .read_exact(&mut type_buf)
        .context("read sample entry type")?;
    let entry_type = SampleEntryType::from_bytes(&type_buf);

    // Read reserved (6 bytes)
    let mut reserved = [0u8; 6];
    reader.read_exact(&mut reserved).context("read reserved")?;
    if reserved != [0u8; 6] {
        return Err(anyhow!("Invalid reserved bytes: {:?}", reserved));
    }

    // Read data reference index (2 bytes)
    let mut dref_buf = [0u8; 2];
    reader
        .read_exact(&mut dref_buf)
        .context("read data reference index")?;
    let data_reference_index = u16::from_be_bytes(dref_buf);

    // Read remaining data
    let remaining_size = size - 16; // 4 + 4 + 6 + 2 = 16 bytes already read
    let mut data = vec![0u8; remaining_size as usize];
    reader
        .read_exact(&mut data)
        .context("read sample entry data")?;

    let data = match entry_type {
        SampleEntryType::Avc1
        | SampleEntryType::Hvc1
        | SampleEntryType::Mp4v
        | SampleEntryType::Aavd => SampleEntryData::Video(parse_video_sample_entry(&data)?),
        SampleEntryType::Mp4a => SampleEntryData::Audio(parse_audio_sample_entry(&data)?),
        _ => SampleEntryData::Other(data),
    };

    Ok(SampleEntry {
        entry_type,
        data_reference_index,
        data,
    })
}

fn parse_video_sample_entry(data: &[u8]) -> Result<VideoSampleEntry, anyhow::Error> {
    let mut cursor = Cursor::new(data);

    // Read version (2 bytes)
    let mut buf2 = [0u8; 2];
    cursor.read_exact(&mut buf2).context("read version")?;
    let version = u16::from_be_bytes(buf2);

    // Read revision level (2 bytes)
    cursor
        .read_exact(&mut buf2)
        .context("read revision level")?;
    let revision_level = u16::from_be_bytes(buf2);

    // Read vendor (4 bytes)
    let mut vendor = [0u8; 4];
    cursor.read_exact(&mut vendor).context("read vendor")?;

    // Read temporal quality (4 bytes)
    let mut buf4 = [0u8; 4];
    cursor
        .read_exact(&mut buf4)
        .context("read temporal quality")?;
    let temporal_quality = u32::from_be_bytes(buf4);

    // Read spatial quality (4 bytes)
    cursor
        .read_exact(&mut buf4)
        .context("read spatial quality")?;
    let spatial_quality = u32::from_be_bytes(buf4);

    // Read width (2 bytes)
    cursor.read_exact(&mut buf2).context("read width")?;
    let width = u16::from_be_bytes(buf2);

    // Read height (2 bytes)
    cursor.read_exact(&mut buf2).context("read height")?;
    let height = u16::from_be_bytes(buf2);

    // Read horizontal resolution (4 bytes, 16.16 fixed point)
    cursor
        .read_exact(&mut buf4)
        .context("read horizontal resolution")?;
    let horizresolution = u32::from_be_bytes(buf4) as f32 / 65536.0;

    // Read vertical resolution (4 bytes, 16.16 fixed point)
    cursor
        .read_exact(&mut buf4)
        .context("read vertical resolution")?;
    let vertresolution = u32::from_be_bytes(buf4) as f32 / 65536.0;

    // Read entry data size (4 bytes)
    cursor
        .read_exact(&mut buf4)
        .context("read entry data size")?;
    let entry_data_size = u32::from_be_bytes(buf4);

    // Read frame count (2 bytes)
    cursor.read_exact(&mut buf2).context("read frame count")?;
    let frame_count = u16::from_be_bytes(buf2);

    // Read compressor name (32 bytes, Pascal string)
    let mut compressor_bytes = [0u8; 32];
    cursor
        .read_exact(&mut compressor_bytes)
        .context("read compressor name")?;
    let compressor_name = parse_pascal_string(&compressor_bytes);

    // Read depth (2 bytes)
    cursor.read_exact(&mut buf2).context("read depth")?;
    let depth = u16::from_be_bytes(buf2);

    // Read color table ID (2 bytes, signed)
    cursor
        .read_exact(&mut buf2)
        .context("read color table ID")?;
    let color_table_id = i16::from_be_bytes(buf2);

    // Read remaining extension data
    let mut extensions = Vec::new();
    cursor
        .read_to_end(&mut extensions)
        .context("read extensions")?;

    Ok(VideoSampleEntry {
        version,
        revision_level,
        vendor,
        temporal_quality,
        spatial_quality,
        width,
        height,
        horizresolution,
        vertresolution,
        entry_data_size,
        frame_count,
        compressor_name,
        depth,
        color_table_id,
        extensions,
    })
}

fn parse_audio_sample_entry(data: &[u8]) -> Result<AudioSampleEntry, anyhow::Error> {
    let mut cursor = Cursor::new(data);

    // Read version (2 bytes)
    let mut buf2 = [0u8; 2];
    cursor.read_exact(&mut buf2).context("read version")?;
    let version = u16::from_be_bytes(buf2);

    // Read revision level (2 bytes)
    cursor
        .read_exact(&mut buf2)
        .context("read revision level")?;
    let revision_level = u16::from_be_bytes(buf2);

    // Read vendor (4 bytes)
    let mut vendor = [0u8; 4];
    cursor.read_exact(&mut vendor).context("read vendor")?;

    // Read channel count (2 bytes)
    cursor.read_exact(&mut buf2).context("read channel count")?;
    let channel_count = u16::from_be_bytes(buf2);

    // Read sample size (2 bytes)
    cursor.read_exact(&mut buf2).context("read sample size")?;
    let sample_size = u16::from_be_bytes(buf2);

    // Read compression ID (2 bytes)
    cursor
        .read_exact(&mut buf2)
        .context("read compression ID")?;
    let compression_id = u16::from_be_bytes(buf2);

    // Read packet size (2 bytes)
    cursor.read_exact(&mut buf2).context("read packet size")?;
    let packet_size = u16::from_be_bytes(buf2);

    // Read sample rate (4 bytes, 16.16 fixed point)
    let mut buf4 = [0u8; 4];
    cursor.read_exact(&mut buf4).context("read sample rate")?;
    let sample_rate = u32::from_be_bytes(buf4) as f32 / 65536.0;

    // Read remaining extension data
    let mut extensions = Vec::new();
    cursor
        .read_to_end(&mut extensions)
        .context("read extensions")?;

    Ok(AudioSampleEntry {
        version,
        revision_level,
        vendor,
        channel_count,
        sample_size,
        compression_id,
        packet_size,
        sample_rate,
        extensions,
    })
}

fn parse_pascal_string(bytes: &[u8; 32]) -> String {
    if bytes[0] == 0 {
        return String::new();
    }

    let length = bytes[0] as usize;
    if length >= 32 {
        return String::new();
    }

    std::str::from_utf8(&bytes[1..=length])
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_entry_type_detection() {
        assert_eq!(
            SampleEntryType::from_bytes(SAMPLE_ENTRY_AVC1),
            SampleEntryType::Avc1
        );
        assert_eq!(
            SampleEntryType::from_bytes(SAMPLE_ENTRY_MP4A),
            SampleEntryType::Mp4a
        );
        assert_eq!(
            SampleEntryType::from_bytes(SAMPLE_ENTRY_TX3G),
            SampleEntryType::Tx3g
        );
        assert_eq!(
            SampleEntryType::from_bytes(SAMPLE_ENTRY_AAVD),
            SampleEntryType::Aavd
        );

        assert!(SampleEntryType::Avc1.is_video());
        assert!(SampleEntryType::Mp4a.is_audio());
        assert!(SampleEntryType::Tx3g.is_text());
        assert!(SampleEntryType::Aavd.is_video());

        assert_eq!(SampleEntryType::Avc1.as_bytes(), SAMPLE_ENTRY_AVC1);
        assert_eq!(SampleEntryType::Mp4a.as_bytes(), SAMPLE_ENTRY_MP4A);
        assert_eq!(SampleEntryType::Aavd.as_bytes(), SAMPLE_ENTRY_AAVD);
    }

    #[test]
    fn test_video_sample_entry_methods() {
        let video_entry = VideoSampleEntry {
            version: 0,
            revision_level: 0,
            vendor: [0; 4],
            temporal_quality: 0,
            spatial_quality: 0,
            width: 1920,
            height: 1080,
            horizresolution: 72.0,
            vertresolution: 72.0,
            entry_data_size: 0,
            frame_count: 1,
            compressor_name: "H.264".to_string(),
            depth: 24,
            color_table_id: -1,
            extensions: vec![],
        };

        assert_eq!(video_entry.dimensions(), (1920, 1080));
        assert!((video_entry.aspect_ratio() - 16.0 / 9.0).abs() < 0.01);
        assert!(video_entry.is_hd());
        assert!(!video_entry.is_4k());
    }

    #[test]
    fn test_audio_sample_entry_methods() {
        let audio_entry = AudioSampleEntry {
            version: 0,
            revision_level: 0,
            vendor: [0; 4],
            channel_count: 2,
            sample_size: 16,
            compression_id: 0,
            packet_size: 0,
            sample_rate: 44100.0,
            extensions: vec![],
        };

        assert!(audio_entry.is_stereo());
        assert!(!audio_entry.is_mono());
        assert_eq!(audio_entry.sample_rate_hz(), 44100);
    }

    #[test]
    fn test_pascal_string_parsing() {
        let mut bytes = [0u8; 32];
        bytes[0] = 5; // length
        bytes[1..6].copy_from_slice(b"H.264");

        let result = parse_pascal_string(&bytes);
        assert_eq!(result, "H.264");

        // Test empty string
        let empty_bytes = [0u8; 32];
        let empty_result = parse_pascal_string(&empty_bytes);
        assert_eq!(empty_result, "");
    }
}
