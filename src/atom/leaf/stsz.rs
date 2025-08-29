use anyhow::{anyhow, Context};
use bon::bon;
use derive_more::{Deref, DerefMut};
use either::Either;
use futures_io::AsyncRead;
use std::{
    fmt::{self},
    io::Read,
    ops::Range,
};

use crate::{
    atom::{
        util::{async_to_sync_read, DebugEllipsis},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const STSZ: &[u8; 4] = b"stsz";

#[derive(Clone, Default, Deref, DerefMut)]
pub struct SampleEntrySizes(Vec<u32>);

impl SampleEntrySizes {
    pub fn inner(&self) -> &[u32] {
        &self.0
    }
}

impl From<Vec<u32>> for SampleEntrySizes {
    fn from(value: Vec<u32>) -> Self {
        SampleEntrySizes(value)
    }
}

impl fmt::Debug for SampleEntrySizes {
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

impl SampleEntrySizes {
    /// Create a new SampleEntrySizes from a vector of sample sizes
    pub fn new(sizes: Vec<u32>) -> Self {
        Self(sizes)
    }

    /// Create a new SampleEntrySizes from a vector of sample sizes
    pub fn from_vec(sizes: Vec<u32>) -> Self {
        Self(sizes)
    }

    /// Convert to the inner Vec<u32>
    pub fn to_vec(&self) -> Vec<u32> {
        self.0.clone()
    }
}

/// Sample Size Atom (stsz) - ISO/IEC 14496-12
/// This atom contains the sample count and a table giving the size in bytes of each sample.
/// Samples within the media may have different sizes, up to the limit of a 32-bit integer.
#[derive(Default, Debug, Clone)]
pub struct SampleSizeAtom {
    /// Version of this atom (0 or 1)
    pub version: u8,
    /// Flags (24 bits)
    pub flags: u32,
    /// If this field is set to some value other than 0, then it gives the (constant) size
    /// of every sample in the track. If this field is set to 0, then the samples have
    /// different sizes, and those sizes are stored in the sample size table.
    pub sample_size: u32,
    /// Number of samples in the track
    pub sample_count: u32,
    /// If `sample_size` is 0, this contains the size of each sample, indexed by sample number.
    /// If `sample_size` is non-zero, this table is empty.
    pub entry_sizes: SampleEntrySizes,
}

impl SampleSizeAtom {
    pub(crate) fn remove_sample_indices(&mut self, indices_to_remove: &[Range<usize>]) {
        let num_samples_removed = indices_to_remove
            .iter()
            .map(|r| r.end - r.start)
            .sum::<usize>() as u32;

        if !self.entry_sizes.is_empty() && !indices_to_remove.is_empty() {
            for range in indices_to_remove {
                self.entry_sizes.drain(range.clone());
            }
        }

        self.sample_count = self.sample_count.saturating_sub(num_samples_removed);
    }

    /// Returns `sample_count` if it's set, otherwise `entry_sizes.len()`
    pub fn sample_count(&self) -> usize {
        if self.sample_count > 0 {
            self.sample_count as usize
        } else {
            self.entry_sizes.len()
        }
    }
}

#[bon]
impl SampleSizeAtom {
    #[builder]
    pub fn new(
        #[builder(setters(vis = "", name = "sample_size_internal"))] sample_size: u32,
        #[builder(default = 0)] sample_count: u32,
        /// either set `sample_size` and `sample_count` or `entry_sizes`
        #[builder(with = FromIterator::from_iter, setters(vis = "", name = "entry_sizes_internal"))]
        entry_sizes: Vec<u32>,
    ) -> Self {
        let entry_sizes: SampleEntrySizes = entry_sizes.into();
        let sample_count = if sample_count == 0 {
            u32::try_from(entry_sizes.len()).expect("entry_sizes.len() should fit in a u32")
        } else {
            sample_count
        };
        Self {
            version: 0,
            flags: 0,
            sample_size,
            sample_count,
            entry_sizes,
        }
    }

    /// Returns an iterator over _all_ sample sizes.
    ///
    /// If `sample_size != 0` this will repeat that value
    /// `sample_count` times; otherwise it will yield
    /// the values from `entry_sizes`.
    pub fn sample_sizes(&self) -> impl Iterator<Item = &u32> + '_ {
        if self.sample_size != 0 {
            Either::Left(std::iter::repeat_n(
                &self.sample_size,
                self.sample_count as usize,
            ))
        } else {
            Either::Right(self.entry_sizes.iter())
        }
    }
}

#[bon]
impl<S: sample_size_atom_builder::State> SampleSizeAtomBuilder<S> {
    pub fn sample_size(
        self,
        sample_size: u32,
    ) -> SampleSizeAtomBuilder<
        sample_size_atom_builder::SetSampleSize<sample_size_atom_builder::SetEntrySizes<S>>,
    >
    where
        S::EntrySizes: sample_size_atom_builder::IsUnset,
        S::SampleSize: sample_size_atom_builder::IsUnset,
    {
        self.entry_sizes_internal(vec![])
            .sample_size_internal(sample_size)
    }

    #[builder(finish_fn(name = "build"))]
    pub fn entry_sizes(
        self,
        #[builder(start_fn)] entry_sizes: impl IntoIterator<Item = u32>,
    ) -> SampleSizeAtom
    where
        S::EntrySizes: sample_size_atom_builder::IsUnset,
        S::SampleSize: sample_size_atom_builder::IsUnset,
        S::SampleCount: sample_size_atom_builder::IsUnset,
    {
        self.entry_sizes_internal(entry_sizes)
            .sample_size_internal(0)
            .sample_count(0)
            .build()
    }
}

impl ParseAtom for SampleSizeAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != STSZ {
            return Err(ParseError::new_unexpected_atom(atom_type, STSZ));
        }
        parse_stsz_data(async_to_sync_read(reader).await?).map_err(ParseError::new_atom_parse)
    }
}

fn parse_stsz_data<R: Read>(mut reader: R) -> Result<SampleSizeAtom, anyhow::Error> {
    // Read all data into buffer for easier parsing
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).context("reading stsz data")?;

    // Minimum size check: version/flags (4) + sample_size (4) + sample_count (4) = 12 bytes
    if buf.len() < 12 {
        return Err(anyhow!("stsz atom too small: {} bytes", buf.len()));
    }

    // Version and flags (4 bytes total)
    let version_flags = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let version = (version_flags >> 24) as u8;
    let flags = version_flags & 0x00FFFFFF;

    // Validate version
    if version != 0 {
        return Err(anyhow!("Unsupported stsz version: {}", version));
    }

    // Sample size (4 bytes)
    let sample_size = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);

    // Sample count (4 bytes)
    let sample_count = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);

    let mut entry_sizes = Vec::new();

    if sample_size == 0 {
        // Variable sample sizes - read the table
        let remaining_bytes = &buf[12..];

        if remaining_bytes.len() % 4 > 0 {
            return Err(anyhow!(
                "Invalid stsz atom: {} is not aligned to 4 bytes",
                remaining_bytes.len()
            ));
        }

        entry_sizes.reserve(sample_count as usize);
        for chunk in remaining_bytes.chunks_exact(4) {
            let size = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            entry_sizes.push(size);
        }
    } else {
        // Constant sample size - no table needed
        if buf.len() != 12 {
            return Err(anyhow!(
                "Invalid stsz atom: constant sample size specified but extra data present"
            ));
        }
    }

    Ok(SampleSizeAtom {
        version,
        flags,
        sample_size,
        sample_count,
        entry_sizes: SampleEntrySizes(entry_sizes),
    })
}

impl fmt::Display for SampleSizeAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SampleSize(count: {}, ", self.sample_count)?;

        if self.sample_size != 0 {
            write!(f, "constant_size: {})", self.sample_size)
        } else {
            write!(f, "variable_sizes: {} entries)", self.entry_sizes.len())
        }
    }
}

impl SerializeAtom for SampleSizeAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*STSZ)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        let mut data = Vec::new();

        // Version and flags (4 bytes total)
        let version_flags = u32::from(self.version) << 24 | (self.flags & 0x00FFFFFF);
        data.extend_from_slice(&version_flags.to_be_bytes());

        // Sample size (4 bytes)
        data.extend_from_slice(&self.sample_size.to_be_bytes());

        // Sample count (4 bytes)
        data.extend_from_slice(&self.sample_count.to_be_bytes());

        // If sample_size is 0, write the sample size table
        if self.sample_size == 0 {
            for size in self.entry_sizes.iter() {
                data.extend_from_slice(&size.to_be_bytes());
            }
        }

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available stsz test data files
    #[test]
    fn test_stsz_roundtrip() {
        test_atom_roundtrip_sync::<SampleSizeAtom>(STSZ);
    }
}
