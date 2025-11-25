use bon::bon;
use derive_more::{Deref, DerefMut};
use either::Either;
use futures_io::AsyncRead;
use std::{
    fmt::{self},
    ops::Range,
};

use crate::{
    atom::{
        util::{read_to_end, DebugList},
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
        fmt::Debug::fmt(&DebugList::new(self.0.iter(), 10), f)
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

    /// Convert to the inner `Vec<u32>`
    pub fn to_vec(&self) -> Vec<u32> {
        self.0.clone()
    }
}

/// Sample Size Atom (stsz) - ISO/IEC 14496-12
/// This atom contains the sample count and a table giving the size in bytes of each sample.
/// Samples within the media may have different sizes, up to the limit of a 32-bit integer.
#[derive(Default, Debug, Clone)]
pub struct SampleSizeAtom {
    pub version: u8,
    pub flags: [u8; 3],
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

        fn adjust_range(n_removed: usize, range: &Range<usize>) -> Range<usize> {
            let start = range.start - n_removed;
            let end = range.end - n_removed;
            start..end
        }

        if !self.entry_sizes.is_empty() && !indices_to_remove.is_empty() {
            let mut n_removed = 0;
            for range in indices_to_remove {
                let range = adjust_range(n_removed, range);
                n_removed += range.len();
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
            flags: [0u8; 3],
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
        let data = read_to_end(reader).await?;
        parser::parse_stsz_data(&data)
    }
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
        serializer::serialize_stsz_data(self)
    }
}

mod serializer {
    use super::SampleSizeAtom;

    pub fn serialize_stsz_data(stsz: SampleSizeAtom) -> Vec<u8> {
        let mut data = Vec::new();

        data.push(stsz.version);
        data.extend(stsz.flags);
        data.extend(stsz.sample_size.to_be_bytes());
        data.extend(stsz.sample_count.to_be_bytes());

        // If sample_size is 0, write the sample size table
        if stsz.sample_size == 0 {
            for size in stsz.entry_sizes.0.into_iter() {
                data.extend(size.to_be_bytes());
            }
        }

        data
    }
}

mod parser {
    use winnow::{
        binary::be_u32,
        combinator::{repeat, seq, trace},
        error::StrContext,
        ModalResult, Parser,
    };

    use super::{SampleEntrySizes, SampleSizeAtom};
    use crate::atom::util::parser::{flags3, stream, version, Stream};

    pub fn parse_stsz_data(input: &[u8]) -> Result<SampleSizeAtom, crate::ParseError> {
        parse_stsz_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_stsz_data_inner(input: &mut Stream<'_>) -> ModalResult<SampleSizeAtom> {
        trace(
            "stsz",
            seq!(SampleSizeAtom {
                version: version,
                flags: flags3,
                sample_size: be_u32.context(StrContext::Label("sample_size")),
                sample_count: be_u32.context(StrContext::Label("sample_count")),
                entry_sizes: repeat(0.., be_u32.context(StrContext::Label("entry_size")))
                    .map(SampleEntrySizes)
                    .context(StrContext::Label("entry_sizes")),
            })
            .context(StrContext::Label("stsz")),
        )
        .parse_next(input)
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
