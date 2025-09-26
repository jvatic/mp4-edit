use std::{
    fmt::{self, Debug},
    ops::RangeBounds,
    time::Duration,
};

use crate::{
    atom::{
        atom_ref::{AtomRef, AtomRefMut},
        stsd::{
            BtrtExtension, DecoderSpecificInfo, EsdsExtension, Mp4aEntryData, SampleEntry,
            SampleEntryData, StsdExtension,
        },
        tkhd::TKHD,
        tref::TREF,
        util::{scaled_duration_range, unscaled_duration},
        EdtsAtomRefMut, MdiaAtomRef, MdiaAtomRefMut, TrackHeaderAtom, TrackReferenceAtom, EDTS,
        MDIA,
    },
    unwrap_atom_data, Atom, AtomData,
};

pub const TRAK: &[u8; 4] = b"trak";

#[derive(Clone, Copy)]
pub struct TrakAtomRef<'a>(AtomRef<'a>);

impl fmt::Debug for TrakAtomRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrakAtomRef")
            .field("track_id", &self.header().unwrap().track_id)
            .finish()
    }
}

impl<'a> TrakAtomRef<'a> {
    pub(crate) fn new(atom: &'a Atom) -> Self {
        Self(AtomRef(Some(atom)))
    }

    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    /// Finds the TKHD atom
    pub fn header(&self) -> Option<&'a TrackHeaderAtom> {
        let atom = self.0.find_child(TKHD)?;
        match atom.data.as_ref()? {
            AtomData::TrackHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the MDIA atom
    pub fn media(&self) -> MdiaAtomRef<'a> {
        MdiaAtomRef(AtomRef(self.0.find_child(MDIA)))
    }

    pub fn track_id(&self) -> Option<u32> {
        let tkhd = self.header()?;
        Some(tkhd.track_id)
    }

    /// Returns the sum of all sample sizes
    pub fn size(&self) -> usize {
        self.media()
            .media_information()
            .sample_table()
            .sample_size()
            .map_or(0, |s| {
                if s.entry_sizes.is_empty() {
                    s.sample_size * s.sample_count
                } else {
                    s.entry_sizes.iter().sum::<u32>()
                }
            }) as usize
    }

    /// Calculates the track's bitrate
    ///
    /// Returns None if either stsz or mdhd atoms can't be found
    pub fn bitrate(&self) -> Option<u32> {
        let duration_secds = self
            .media()
            .header()
            .map(|mdhd| (mdhd.duration as f64) / f64::from(mdhd.timescale))?;

        self.media()
            .media_information()
            .sample_table()
            .sample_size()
            .map(|s| {
                let num_bits = s
                    .entry_sizes
                    .iter()
                    .map(|s| *s as usize)
                    .sum::<usize>()
                    .saturating_mul(8);

                let bitrate = (num_bits as f64) / duration_secds;
                bitrate.round() as u32
            })
    }
}

#[derive(Debug)]
pub struct TrakAtomRefMut<'a>(pub(crate) AtomRefMut<'a>);

impl<'a> TrakAtomRefMut<'a> {
    pub(crate) fn new(atom: &'a mut Atom) -> Self {
        Self(AtomRefMut(atom))
    }

    pub fn as_ref(&self) -> TrakAtomRef<'_> {
        TrakAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> TrakAtomRef<'a> {
        TrakAtomRef(self.0.into_ref())
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children()
    }

    /// Finds or inserts the TKHD atom
    pub fn header(&mut self) -> &mut TrackHeaderAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(TKHD)
                .insert_data(AtomData::TrackHeader(TrackHeaderAtom::default()))
                .call(),
            AtomData::TrackHeader,
        )
    }

    /// Finds or creates the MDIA atom
    pub fn media(&mut self) -> MdiaAtomRefMut<'_> {
        MdiaAtomRefMut(
            self.0
                .find_or_insert_child(MDIA)
                .insert_after(vec![TREF, EDTS, TKHD])
                .call(),
        )
    }

    /// Finds the MDIA atom
    pub fn into_media(self) -> Option<MdiaAtomRefMut<'a>> {
        let atom = self.0.into_child(MDIA)?;
        Some(MdiaAtomRefMut(AtomRefMut(atom)))
    }

    /// Finds or creates the EDTS atom
    pub fn edit_list_container(&mut self) -> EdtsAtomRefMut<'_> {
        EdtsAtomRefMut(
            self.0
                .find_or_insert_child(EDTS)
                .insert_after(vec![TREF, TKHD])
                .call(),
        )
    }

    /// Finds or inserts the TREF atom
    pub fn track_reference(&mut self) -> &mut TrackReferenceAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(TREF)
                .insert_after(vec![TKHD])
                .insert_data(AtomData::TrackReference(TrackReferenceAtom::default()))
                .call(),
            AtomData::TrackReference,
        )
    }

    /// Updates track metadata with the new audio bitrate
    ///
    /// Creates any missing atoms needed to do so
    pub fn update_audio_bitrate(&mut self, bitrate: u32) {
        let mut mdia = self.media();
        let mut minf = mdia.media_information();
        let mut stbl = minf.sample_table();
        let stsd = stbl.sample_description();

        let entry = stsd.find_or_create_audio_entry(
            |entry| matches!(entry.data, SampleEntryData::Mp4a(_)),
            || SampleEntry {
                data_reference_index: 0,
                data: SampleEntryData::Mp4a(Mp4aEntryData::default()),
            },
        );

        if let SampleEntryData::Mp4a(mp4a) = &mut entry.data {
            let mut sample_frequency = None;
            mp4a.extensions
                .retain(|ext| matches!(ext, StsdExtension::Esds(_)));
            let esds = mp4a.find_or_create_extension(
                |ext| matches!(ext, StsdExtension::Esds(_)),
                || StsdExtension::Esds(EsdsExtension::default()),
            );
            if let StsdExtension::Esds(esds) = esds {
                let cfg = esds
                    .es_descriptor
                    .decoder_config_descriptor
                    .get_or_insert_default();
                cfg.avg_bitrate = bitrate;
                cfg.max_bitrate = bitrate;
                if let Some(DecoderSpecificInfo::Audio(a)) = cfg.decoder_specific_info.as_ref() {
                    sample_frequency = a.sampling_frequency.as_hz();
                }
            }
            mp4a.extensions.push(StsdExtension::Btrt(BtrtExtension {
                buffer_size_db: 0,
                avg_bitrate: bitrate,
                max_bitrate: bitrate,
            }));

            if let Some(hz) = sample_frequency {
                mp4a.sample_rate = hz as f32;
            }
        } else {
            // this indicates a programming error since we won't get here with parsed data
            unreachable!("STSD constructed with invalid data")
        }
    }

    /// trims given duration range, excluding partially matched samples, and returns the actual duration trimmed
    pub(crate) fn trim_duration<R>(&mut self, movie_timescale: u64, trim_ranges: &[R]) -> Duration
    where
        R: RangeBounds<Duration> + Clone + Debug,
    {
        let mut mdia = self.media();
        let media_timescale = u64::from(mdia.header().timescale);
        let mut minf = mdia.media_information();
        let mut stbl = minf.sample_table();

        let scaled_ranges = trim_ranges
            .iter()
            .cloned()
            .map(|range| scaled_duration_range(range, media_timescale))
            .collect::<Vec<_>>();

        // Step 1: Determine which samples to remove based on time
        let (trimmed_duration, sample_indices_to_remove) =
            stbl.time_to_sample().trim_duration(&scaled_ranges);

        let trimmed_duration = unscaled_duration(trimmed_duration, media_timescale);

        // Step 2: Update sample sizes
        stbl.sample_size()
            .remove_sample_indices(&sample_indices_to_remove);

        // Step 3: Calculate and remove chunks based on samples
        let total_chunks = stbl.chunk_offset().chunk_count();
        let chunk_indices_to_remove = stbl
            .sample_to_chunk()
            .remove_sample_indices(&sample_indices_to_remove, total_chunks);

        // Step 4: Remove chunk offsets
        stbl.chunk_offset()
            .remove_chunk_indices(&chunk_indices_to_remove);

        // Step 5: Update headers
        mdia.header().update_duration(|d| d - trimmed_duration);
        self.header()
            .update_duration(movie_timescale, |d| d - trimmed_duration);

        trimmed_duration
    }
}
