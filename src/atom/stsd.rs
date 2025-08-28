use anyhow::{anyhow, Context};
use derive_more::Display;
use futures_io::AsyncRead;
use std::io::{Cursor, Read};

pub use crate::atom::stsd::extension::{
    BtrtExtension, DecoderSpecificInfo, EsdsExtension, StsdExtension,
};
use crate::{
    atom::{
        stsd::extension::parse_stsd_extensions,
        util::{async_to_sync_read, FourCC},
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

mod extension;

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
pub const SAMPLE_ENTRY_TEXT: &[u8; 4] = b"text"; // Plain text

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
    /// Plain text
    Text,
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
            SAMPLE_ENTRY_TEXT => SampleEntryType::Text,
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
            SampleEntryType::Text => SAMPLE_ENTRY_TEXT,
            SampleEntryType::Unknown(bytes) => &bytes.0,
        }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(self.as_bytes()).unwrap_or("????")
    }
}

#[derive(Debug, Clone)]
pub enum SampleEntryData {
    Video(VideoSampleEntry),
    Audio(AudioSampleEntry),
    Text(TextSampleEntry),
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
    pub extensions: Vec<StsdExtension>,
}

#[derive(Default, Debug, Clone)]
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
    pub extensions: Vec<StsdExtension>,
}

impl AudioSampleEntry {
    pub fn find_or_create_extension<P, D>(&mut self, pred: P, default_fn: D) -> &mut StsdExtension
    where
        P: Fn(&StsdExtension) -> bool,
        D: FnOnce() -> StsdExtension,
    {
        if let Some(index) = self.extensions.iter().position(pred) {
            return &mut self.extensions[index];
        }
        self.extensions.push(default_fn());
        self.extensions.last_mut().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct TextSampleEntry {
    /// Version (usually 0)
    pub version: u16,
    /// Revision level (usually 0)
    pub revision_level: u16,
    /// Vendor identifier
    pub vendor: [u8; 4],
    /// Display flags
    pub display_flags: u32,
    /// Text justification (0 = left, 1 = center, -1 = right)
    pub text_justification: i8,
    /// Background color (RGB)
    pub background_color: [u16; 3],
    /// Default text box (top, left, bottom, right)
    pub default_text_box: [u16; 4],
    /// Default style record
    pub default_style: Option<Vec<u8>>,
    /// Font table
    pub font_table: Option<Vec<u8>>,
    /// Extension data (codec-specific atoms)
    pub extensions: Vec<StsdExtension>,
}

#[derive(Default, Debug, Clone)]
pub struct SampleDescriptionTableAtom {
    /// Version of the stsd atom format (0)
    pub version: u8,
    /// Flags for the stsd atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of sample entries
    pub entries: Vec<SampleEntry>,
}

impl From<Vec<SampleEntry>> for SampleDescriptionTableAtom {
    fn from(entries: Vec<SampleEntry>) -> Self {
        SampleDescriptionTableAtom {
            version: 0,
            flags: [0u8; 3],
            entries,
        }
    }
}

impl SampleDescriptionTableAtom {
    pub fn find_or_create_entry<P, D>(&mut self, pred: P, default_fn: D) -> &mut SampleEntry
    where
        P: Fn(&SampleEntry) -> bool,
        D: FnOnce() -> SampleEntry,
    {
        if let Some(index) = self.entries.iter().position(pred) {
            return &mut self.entries[index];
        }
        self.entries.push(default_fn());
        self.entries.last_mut().unwrap()
    }
}

impl ParseAtom for SampleDescriptionTableAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STSD {
            return Err(ParseError::new_unexpected_atom(atom_type, STSD));
        }
        parse_stsd_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
}

impl SerializeAtom for SampleDescriptionTableAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STSD)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version (1 byte)
        data.push(self.version);

        // Flags (3 bytes)
        data.extend_from_slice(&self.flags);

        // Entry count (4 bytes, big-endian)
        data.extend_from_slice(&(self.entries.len() as u32).to_be_bytes());

        // Sample entries
        for entry in self.entries {
            let mut entry_data = Vec::new();

            // Reserved (6 bytes) - must be zero
            entry_data.extend_from_slice(&[0u8; 6]);

            // Data reference index (2 bytes, big-endian)
            entry_data.extend_from_slice(&entry.data_reference_index.to_be_bytes());

            // Entry-specific data based on type
            match entry.data {
                SampleEntryData::Video(video) => {
                    // Video sample entry structure
                    entry_data.extend_from_slice(&video.version.to_be_bytes());
                    entry_data.extend_from_slice(&video.revision_level.to_be_bytes());
                    entry_data.extend_from_slice(&video.vendor);
                    entry_data.extend_from_slice(&video.temporal_quality.to_be_bytes());
                    entry_data.extend_from_slice(&video.spatial_quality.to_be_bytes());
                    entry_data.extend_from_slice(&video.width.to_be_bytes());
                    entry_data.extend_from_slice(&video.height.to_be_bytes());
                    entry_data.extend_from_slice(
                        &((video.horizresolution * 65536.0) as u32).to_be_bytes(),
                    );
                    entry_data.extend_from_slice(
                        &((video.vertresolution * 65536.0) as u32).to_be_bytes(),
                    );
                    entry_data.extend_from_slice(&video.entry_data_size.to_be_bytes());
                    entry_data.extend_from_slice(&video.frame_count.to_be_bytes());

                    // Compressor name (32 bytes, Pascal string)
                    let mut compressor_bytes = [0u8; 32];
                    let name_bytes = video.compressor_name.as_bytes();
                    let len = name_bytes.len().min(31);
                    compressor_bytes[0] = len as u8;
                    compressor_bytes[1..=len].copy_from_slice(&name_bytes[..len]);
                    entry_data.extend_from_slice(&compressor_bytes);

                    entry_data.extend_from_slice(&video.depth.to_be_bytes());
                    entry_data.extend_from_slice(&video.color_table_id.to_be_bytes());
                    video.extensions.into_iter().for_each(|ext| {
                        let ext_data: Vec<u8> = ext.to_bytes();
                        entry_data.extend_from_slice(&ext_data);
                    });
                }
                SampleEntryData::Audio(audio) => {
                    // Audio sample entry structure
                    entry_data.extend_from_slice(&audio.version.to_be_bytes());
                    entry_data.extend_from_slice(&audio.revision_level.to_be_bytes());
                    entry_data.extend_from_slice(&audio.vendor);
                    entry_data.extend_from_slice(&audio.channel_count.to_be_bytes());
                    entry_data.extend_from_slice(&audio.sample_size.to_be_bytes());
                    entry_data.extend_from_slice(&audio.compression_id.to_be_bytes());
                    entry_data.extend_from_slice(&audio.packet_size.to_be_bytes());
                    entry_data
                        .extend_from_slice(&((audio.sample_rate * 65536.0) as u32).to_be_bytes());
                    audio.extensions.into_iter().for_each(|ext| {
                        let ext_data = ext.to_bytes();
                        entry_data.extend_from_slice(&ext_data);
                    });
                }
                SampleEntryData::Text(text) => {
                    // Text sample entry structure
                    entry_data.extend_from_slice(&text.version.to_be_bytes());
                    entry_data.extend_from_slice(&text.revision_level.to_be_bytes());
                    entry_data.extend_from_slice(&text.vendor);
                    entry_data.extend_from_slice(&text.display_flags.to_be_bytes());
                    entry_data.push(text.text_justification as u8);

                    // Background color (3 * 2 bytes)
                    for &color in &text.background_color {
                        entry_data.extend_from_slice(&color.to_be_bytes());
                    }

                    // Default text box (4 * 2 bytes)
                    for &coord in &text.default_text_box {
                        entry_data.extend_from_slice(&coord.to_be_bytes());
                    }

                    // Add default style if present
                    if let Some(ref style_data) = text.default_style {
                        entry_data.extend_from_slice(style_data);
                    }

                    // Add font table if present
                    if let Some(ref font_data) = text.font_table {
                        entry_data.extend_from_slice(font_data);
                    }

                    // Add extensions
                    text.extensions.into_iter().for_each(|ext| {
                        let ext_data: Vec<u8> = ext.to_bytes();
                        entry_data.extend_from_slice(&ext_data);
                    });
                }
                SampleEntryData::Other(other_data) => {
                    // Other sample entry data
                    entry_data.extend_from_slice(&other_data);
                }
            }

            // Calculate total entry size (4 + 4 + entry_data.len())
            let entry_size = 8 + entry_data.len();

            // Write entry size (4 bytes, big-endian)
            data.extend_from_slice(&(entry_size as u32).to_be_bytes());

            // Write entry type (4 bytes)
            data.extend_from_slice(entry.entry_type.as_bytes());

            // Write entry data
            data.extend_from_slice(&entry_data);
        }

        data
    }
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
        let entry = parse_sample_entry(&mut reader).context(format!("parse sample entry {i}"))?;
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
        SampleEntryType::Avc1 | SampleEntryType::Hvc1 | SampleEntryType::Mp4v => {
            SampleEntryData::Video(parse_video_sample_entry(&data)?)
        }
        SampleEntryType::Mp4a | SampleEntryType::Aavd => {
            SampleEntryData::Audio(parse_audio_sample_entry(&data)?)
        }
        SampleEntryType::Tx3g
        | SampleEntryType::Wvtt
        | SampleEntryType::Stpp
        | SampleEntryType::Text => SampleEntryData::Text(parse_text_sample_entry(&data)?),
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
    let mut extension_data = Vec::new();
    cursor
        .read_to_end(&mut extension_data)
        .context("read extensions")?;

    let extensions = parse_stsd_extensions(&extension_data)
        .context("parse extensions")?
        .extensions;

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
    let mut extension_data = Vec::new();
    cursor
        .read_to_end(&mut extension_data)
        .context("read extensions")?;

    let extensions = parse_stsd_extensions(&extension_data)
        .context("parse extensions")?
        .extensions;

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

fn parse_text_sample_entry(data: &[u8]) -> Result<TextSampleEntry, anyhow::Error> {
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

    // Read display flags (4 bytes)
    let mut buf4 = [0u8; 4];
    cursor.read_exact(&mut buf4).context("read display flags")?;
    let display_flags = u32::from_be_bytes(buf4);

    // Read text justification (1 byte)
    let mut buf1 = [0u8; 1];
    cursor
        .read_exact(&mut buf1)
        .context("read text justification")?;
    let text_justification = buf1[0] as i8;

    // Read background color (3 * 2 bytes = 6 bytes)
    let mut background_color = [0u16; 3];
    for item in &mut background_color {
        cursor
            .read_exact(&mut buf2)
            .context("read background color")?;
        *item = u16::from_be_bytes(buf2);
    }

    // Read default text box (4 * 2 bytes = 8 bytes)
    let mut default_text_box = [0u16; 4];
    for item in &mut default_text_box {
        cursor
            .read_exact(&mut buf2)
            .context("read default text box")?;
        *item = u16::from_be_bytes(buf2);
    }

    // For simplicity, we'll store any remaining data as raw bytes
    // In a full implementation, you'd parse the default style and font table
    let mut remaining_data = Vec::new();
    cursor
        .read_to_end(&mut remaining_data)
        .context("read remaining text entry data")?;

    // Try to parse extensions from remaining data, but handle gracefully if it fails
    let (default_style, font_table, extensions) = if remaining_data.is_empty() {
        (None, None, Vec::new())
    } else {
        // Try to parse as extensions first
        match parse_stsd_extensions(&remaining_data) {
            Ok(parsed) => (None, None, parsed.extensions),
            Err(_) => {
                // If extension parsing fails, treat as raw style/font data
                (Some(remaining_data), None, Vec::new())
            }
        }
    };

    Ok(TextSampleEntry {
        version,
        revision_level,
        vendor,
        display_flags,
        text_justification,
        background_color,
        default_text_box,
        default_style,
        font_table,
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
    use crate::atom::stsd::extension::parse_stsd_extensions;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    use super::*;

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

    #[test]
    fn test_parse_extensions_round_trip() {
        let extension_data: Vec<u8> = vec![
            0, 0, 0, 51, 101, 115, 100, 115, 0, 0, 0, 0, 3, 128, 128, 128, 34, 0, 1, 0, 4, 128,
            128, 128, 20, 64, 21, 0, 0, 0, 0, 0, 245, 74, 0, 0, 245, 74, 5, 128, 128, 128, 2, 19,
            144, 6, 128, 128, 128, 1, 2, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 245, 74,
            0, 0, 245, 74,
        ];
        let result = parse_stsd_extensions(&extension_data);
        assert!(result.is_ok(), "extensions should parse");
        let result = result.unwrap().extensions;

        let round_trip_data: Vec<u8> = result
            .clone()
            .into_iter()
            .flat_map(|ext| ext.to_bytes())
            .collect();

        assert_eq!(
            round_trip_data, extension_data,
            "expected round trip data to equal input data ({result:?})"
        );
    }

    /// Test round-trip for all available stsd test data files
    #[test]
    fn test_stsd_roundtrip() {
        test_atom_roundtrip_sync::<SampleDescriptionTableAtom>(STSD);
    }

    #[test]
    fn test_text_sample_entry_type_recognition() {
        // Test that "text" sample entry type is recognized correctly
        let text_bytes = b"text";
        let entry_type = SampleEntryType::from_bytes(text_bytes);
        assert_eq!(entry_type, SampleEntryType::Text);

        // Test round-trip
        let bytes_back = entry_type.as_bytes();
        assert_eq!(bytes_back, text_bytes);

        // Test string representation
        assert_eq!(entry_type.as_str(), "text");
    }

    #[test]
    fn test_text_sample_entry_parsing() {
        // Test that a "text" sample entry gets parsed as Text variant, not Other
        // This simulates the user's example data
        let sample_data = vec![
            0, 0, 0, 1, // version + revision_level
            0, 0, 0, 0, // vendor
            0, 0, 0, 0, // display_flags
            0, // text_justification
            0, 0, 0, 0, 0, 0, // background_color (3 * u16)
            0, 0, 0, 1, // default_text_box (4 * u16) - partial
            0, 0, 0, 0, // default_text_box continued
            0, 13, // some extension data
            102, 116, 97, 98, 0, 1, 0, 1, 0, // remaining data
        ];

        let result = parse_text_sample_entry(&sample_data);
        assert!(
            result.is_ok(),
            "text sample entry should parse successfully: {:?}",
            result.err()
        );

        let text_entry = result.unwrap();
        assert_eq!(text_entry.version, 0);
        assert_eq!(text_entry.revision_level, 1);
        assert_eq!(text_entry.display_flags, 0);
        assert_eq!(text_entry.text_justification, 0);
    }
}
