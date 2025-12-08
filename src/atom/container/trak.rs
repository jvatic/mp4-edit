use std::{
    fmt::{self, Debug},
    ops::{Range, RangeBounds},
    time::Duration,
};

use crate::{
    atom::{
        atom_ref::{AtomRef, AtomRefMut},
        elst::EditEntry,
        stsd::{
            AudioSampleEntry, BtrtExtension, DecoderSpecificInfo, EsdsExtension, SampleEntry,
            SampleEntryData, SampleEntryType, StsdExtension,
        },
        tkhd::TKHD,
        tref::TREF,
        util::{scaled_duration_range, unscaled_duration},
        EdtsAtomRef, EdtsAtomRefMut, MdiaAtomRef, MdiaAtomRefMut, TrackHeaderAtom,
        TrackReferenceAtom, EDTS, MDIA,
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

    pub fn edit_list_container(&self) -> EdtsAtomRef<'a> {
        EdtsAtomRef(AtomRef(self.0.find_child(EDTS)))
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

        let entry = stsd.find_or_create_entry(
            |entry| matches!(entry.data, SampleEntryData::Audio(_)),
            || SampleEntry {
                entry_type: SampleEntryType::Mp4a,
                data_reference_index: 0,
                data: SampleEntryData::Audio(AudioSampleEntry::default()),
            },
        );

        entry.entry_type = SampleEntryType::Mp4a;

        if let SampleEntryData::Audio(audio) = &mut entry.data {
            let mut sample_frequency = None;
            audio
                .extensions
                .retain(|ext| matches!(ext, StsdExtension::Esds(_)));
            let esds = audio.find_or_create_extension(
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
                    sample_frequency = Some(a.sampling_frequency.as_hz());
                }
            }
            audio.extensions.push(StsdExtension::Btrt(BtrtExtension {
                buffer_size_db: 0,
                avg_bitrate: bitrate,
                max_bitrate: bitrate,
            }));

            if let Some(hz) = sample_frequency {
                audio.sample_rate = hz as f32;
            }
        } else {
            // this indicates a programming error since we won't get here with parsed data
            unreachable!("STSD constructed with invalid data")
        }
    }

    /// trims given duration range, excluding partially matched samples, and returns the remaining duration
    pub(crate) fn trim_duration<R>(&mut self, movie_timescale: u64, trim_ranges: &[R]) -> Duration
    where
        R: RangeBounds<Duration> + Clone + Debug,
    {
        let mut mdia = self.media();
        let media_timescale = u64::from(mdia.header().timescale);
        let media_duration = mdia.header().duration;
        let mut minf = mdia.media_information();
        let mut stbl = minf.sample_table();

        // Step 1: Scale and convert trim ranges
        let scaled_ranges = trim_ranges
            .iter()
            .cloned()
            .map(|range| {
                convert_range(
                    media_duration,
                    scaled_duration_range(range, media_timescale),
                )
            })
            .collect::<Vec<_>>();

        // Step 2: Determine which samples to remove based on time
        let (remaining_duration, sample_indices_to_remove) =
            stbl.time_to_sample().trim_duration(&scaled_ranges);

        let remaining_duration = unscaled_duration(remaining_duration, media_timescale);

        // Step 3: Update sample sizes
        let removed_sample_sizes = stbl
            .sample_size()
            .remove_sample_indices(&sample_indices_to_remove);

        // Step 4: Calculate and remove chunks based on samples
        let total_chunks = stbl.chunk_offset().chunk_count();
        let chunk_offset_ops = stbl
            .sample_to_chunk()
            .remove_sample_indices(&sample_indices_to_remove, total_chunks);

        // Step 5: Resolve chunk offset ops that depend on sample sizes
        let chunk_offsets = &stbl.chunk_offset().chunk_offsets;
        let chunk_offset_ops = chunk_offset_ops
            .into_iter()
            .map(|op| op.resolve(chunk_offsets, &removed_sample_sizes))
            .collect::<anyhow::Result<Vec<_>>>()
            .expect("chunk offset ops should only involve removed sample indices and valid chunk indices");

        // Step 6: Remove chunk offsets
        stbl.chunk_offset().apply_operations(chunk_offset_ops);

        // Step 7: Update headers
        mdia.header().update_duration(|_| remaining_duration);
        self.header()
            .update_duration(movie_timescale, |_| remaining_duration);

        // Step 8: Replace any edit list entries with a no-op one
        if matches!(self.as_ref().edit_list_container().edit_list(), Some(_)) {
            self.edit_list_container()
                .edit_list()
                .replace_entries(vec![EditEntry::builder()
                    .movie_timescale(movie_timescale)
                    .segment_duration(remaining_duration)
                    .build()]);
        }

        remaining_duration
    }
}

fn convert_range(media_time: u64, range: impl RangeBounds<u64>) -> Range<u64> {
    use std::ops::Bound;
    let start = match range.start_bound() {
        Bound::Included(start) => *start,
        Bound::Excluded(start) => *start + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(end) => *end + 1,
        Bound::Excluded(end) => *end,
        Bound::Unbounded => media_time,
    };
    start..end
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{ops::Bound, time::Duration};

    use bon::Builder;

    use crate::atom::{
        container::{DINF, MDIA, MINF, STBL, TRAK},
        dref::{DataReferenceAtom, DataReferenceEntry, DREF},
        gmin::GMIN,
        hdlr::{HandlerReferenceAtom, HandlerType, HDLR},
        mdhd::{MediaHeaderAtom, MDHD},
        smhd::{SoundMediaHeaderAtom, SMHD},
        stco_co64::{ChunkOffsetAtom, STCO},
        stsc::{SampleToChunkAtom, SampleToChunkEntry, STSC},
        stsd::{SampleDescriptionTableAtom, STSD},
        stsz::{SampleSizeAtom, STSZ},
        stts::{TimeToSampleAtom, TimeToSampleEntry, STTS},
        text::TEXT,
        tkhd::{TrackHeaderAtom, TKHD},
        util::scaled_duration,
        Atom, AtomHeader, BaseMediaInfoAtom, TextMediaInfoAtom, TrakAtomRef, TrakAtomRefMut, GMHD,
    };

    #[bon::builder(finish_fn(name = "build"), state_mod(vis = "pub(crate)"))]
    pub fn create_test_track(
        #[builder(getter)] movie_timescale: u32,
        #[builder(getter)] media_timescale: u32,
        #[builder(getter)] duration: Duration,
        handler_reference: Option<HandlerReferenceAtom>,
        minf_header: Option<Atom>,
        stsc_entries: Option<Vec<SampleToChunkEntry>>,
        sample_sizes: Option<Vec<u32>>,
    ) -> Atom {
        Atom::builder()
            .header(AtomHeader::new(*TRAK))
            .children(vec![
                // Track header (tkhd)
                Atom::builder()
                    .header(AtomHeader::new(*TKHD))
                    .data(
                        TrackHeaderAtom::builder()
                            .track_id(1)
                            .duration(scaled_duration(duration, movie_timescale as u64))
                            .build(),
                    )
                    .build(),
                // Media (mdia)
                create_test_media(media_timescale, duration)
                    .maybe_handler_reference(handler_reference)
                    .maybe_minf_header(minf_header)
                    .maybe_stsc_entries(stsc_entries)
                    .maybe_sample_sizes(sample_sizes)
                    .build(),
            ])
            .build()
    }

    #[bon::builder(finish_fn(name = "build"))]
    fn create_test_media(
        #[builder(start_fn)] media_timescale: u32,
        #[builder(start_fn)] duration: Duration,
        handler_reference: Option<HandlerReferenceAtom>,
        minf_header: Option<Atom>,
        stsc_entries: Option<Vec<SampleToChunkEntry>>,
        sample_sizes: Option<Vec<u32>>,
    ) -> Atom {
        let handler_reference = handler_reference.unwrap_or_else(|| {
            HandlerReferenceAtom::builder()
                .handler_type(HandlerType::Audio)
                .name("SoundHandler".to_string())
                .build()
        });

        let minf_header = minf_header.unwrap_or_else(|| {
            match &handler_reference.handler_type {
                HandlerType::Audio => {
                    // Sound media information header (smhd)
                    Atom::builder()
                        .header(AtomHeader::new(*SMHD))
                        .data(SoundMediaHeaderAtom::default())
                        .build()
                }
                HandlerType::Text => {
                    // Generic media information header (gmhd)
                    Atom::builder()
                        .header(AtomHeader::new(*GMHD))
                        .children(vec![
                            Atom::builder()
                                .header(AtomHeader::new(*GMIN))
                                .data(BaseMediaInfoAtom::default())
                                .build(),
                            Atom::builder()
                                .header(AtomHeader::new(*TEXT))
                                .data(TextMediaInfoAtom::default())
                                .build(),
                        ])
                        .build()
                }
                _ => {
                    todo!(
                        "no default minf header for {:?}",
                        &handler_reference.handler_type
                    )
                }
            }
        });

        Atom::builder()
            .header(AtomHeader::new(*MDIA))
            .children(vec![
                // Media header (mdhd)
                Atom::builder()
                    .header(AtomHeader::new(*MDHD))
                    .data(
                        MediaHeaderAtom::builder()
                            .timescale(media_timescale)
                            .duration(scaled_duration(duration, media_timescale as u64))
                            .build(),
                    )
                    .build(),
                // Handler reference (hdlr)
                Atom::builder()
                    .header(AtomHeader::new(*HDLR))
                    .data(handler_reference)
                    .build(),
                // Media information (minf)
                create_test_media_info()
                    .media_timescale(media_timescale)
                    .duration(duration)
                    .header(minf_header)
                    .maybe_stsc_entries(stsc_entries)
                    .maybe_sample_sizes(sample_sizes)
                    .build(),
            ])
            .build()
    }

    #[bon::builder(finish_fn(name = "build"))]
    fn create_test_media_info(
        media_timescale: u32,
        duration: Duration,
        header: Atom,
        stsc_entries: Option<Vec<SampleToChunkEntry>>,
        sample_sizes: Option<Vec<u32>>,
    ) -> Atom {
        let stsc_entries = stsc_entries.unwrap_or_else(|| {
            vec![SampleToChunkEntry::builder()
                .first_chunk(1)
                .samples_per_chunk(2)
                .sample_description_index(1)
                .build()]
        });

        let sample_sizes = sample_sizes.unwrap_or_else(|| {
            // one sample per second
            let total_samples = duration.as_secs() as usize;
            let sample_size = 256;
            vec![sample_size; total_samples]
        });

        Atom::builder()
            .header(AtomHeader::new(*MINF))
            .children(vec![
                header,
                // Data information (dinf)
                Atom::builder()
                    .header(AtomHeader::new(*DINF))
                    .children(vec![
                        // Data reference (dref)
                        Atom::builder()
                            .header(AtomHeader::new(*DREF))
                            .data(
                                DataReferenceAtom::builder()
                                    .entry(DataReferenceEntry::builder().url("").build())
                                    .build(),
                            )
                            .build(),
                    ])
                    .build(),
                // Sample table (stbl)
                create_test_sample_table()
                    .media_timescale(media_timescale)
                    .stsc_entries(stsc_entries)
                    .sample_sizes(sample_sizes)
                    .build(),
            ])
            .build()
    }

    #[bon::builder(finish_fn(name = "build"))]
    fn create_test_sample_table(
        media_timescale: u32,
        stsc_entries: Vec<SampleToChunkEntry>,
        sample_sizes: Vec<u32>,
        #[builder(default = 1000)] mdat_content_offset: u64,
    ) -> Atom {
        let total_samples = sample_sizes.len() as u32;

        // Calculate chunk offsets
        let chunk_offsets = {
            let mut chunk_offsets = Vec::new();
            let mut current_offset = mdat_content_offset;
            let mut sample_size_index = 0;
            let mut remaining_samples = total_samples;
            let mut stsc_iter = stsc_entries.iter().peekable();
            while let Some(entry) = stsc_iter.next() {
                let n_chunks = match stsc_iter.peek() {
                    Some(next) => next.first_chunk - entry.first_chunk,
                    None => remaining_samples / entry.samples_per_chunk,
                };

                let n_samples = entry.samples_per_chunk * n_chunks;
                remaining_samples = remaining_samples.saturating_sub(n_samples);

                for _ in 0..n_chunks {
                    chunk_offsets.push(current_offset);
                    current_offset += sample_sizes
                        .iter()
                        .skip(sample_size_index)
                        .take(entry.samples_per_chunk as usize)
                        .map(|s| *s as u64)
                        .sum::<u64>();
                    sample_size_index += entry.samples_per_chunk as usize;
                }
            }

            chunk_offsets
        };

        Atom::builder()
            .header(AtomHeader::new(*STBL))
            .children(vec![
                // Sample Description (stsd)
                Atom::builder()
                    .header(AtomHeader::new(*STSD))
                    .data(SampleDescriptionTableAtom::default())
                    .build(),
                // Time to Sample (stts)
                Atom::builder()
                    .header(AtomHeader::new(*STTS))
                    .data(
                        TimeToSampleAtom::builder()
                            .entry(
                                TimeToSampleEntry::builder()
                                    .sample_count(total_samples)
                                    .sample_duration(media_timescale)
                                    .build(),
                            )
                            .build(),
                    )
                    .build(),
                // Sample to Chunk (stsc)
                Atom::builder()
                    .header(AtomHeader::new(*STSC))
                    .data(SampleToChunkAtom::from(stsc_entries))
                    .build(),
                // Sample Size (stsz)
                Atom::builder()
                    .header(AtomHeader::new(*STSZ))
                    .data(SampleSizeAtom::builder().entry_sizes(sample_sizes).build())
                    .build(),
                // Chunk Offset (stco)
                Atom::builder()
                    .header(AtomHeader::new(*STCO))
                    .data(
                        ChunkOffsetAtom::builder()
                            .chunk_offsets(chunk_offsets)
                            .build(),
                    )
                    .build(),
            ])
            .build()
    }

    #[derive(Debug, Builder)]
    struct TrimDurationRange {
        start_bound: Bound<Duration>,
        end_bound: Bound<Duration>,
    }

    #[derive(Builder)]
    struct TrimDurationTestCase<ECO> {
        #[builder(field)]
        ranges: Vec<TrimDurationRange>,
        #[builder(default = 1_000)]
        movie_timescale: u32,
        #[builder(default = 10_000)]
        media_timescale: u32,
        expected_duration: Duration,
        expected_chunk_offsets: ECO,
    }

    impl<ECO, S> TrimDurationTestCaseBuilder<ECO, S>
    where
        S: trim_duration_test_case_builder::State,
    {
        fn range(mut self, range: TrimDurationRange) -> Self {
            self.ranges.push(range);
            self
        }
    }

    fn get_chunk_offsets(track: TrakAtomRef) -> Vec<u64> {
        track
            .media()
            .media_information()
            .sample_table()
            .chunk_offset()
            .unwrap()
            .chunk_offsets
            .clone()
            .into_inner()
    }

    fn test_trim_duration<ECO>(mut track: Atom, test_case: TrimDurationTestCase<ECO>)
    where
        ECO: FnOnce(Vec<u64>) -> Vec<u64>,
    {
        let mut track = TrakAtomRefMut::new(&mut track);
        let starting_chunk_offsets = get_chunk_offsets(track.as_ref());

        let trim_ranges = test_case
            .ranges
            .into_iter()
            .map(|r| (r.start_bound, r.end_bound))
            .collect::<Vec<_>>();
        let res = track.trim_duration(test_case.movie_timescale as u64, &trim_ranges);
        assert_eq!(res, test_case.expected_duration);

        let trimmed_chunk_offsets = get_chunk_offsets(track.as_ref());
        let expected_chunk_offsets = (test_case.expected_chunk_offsets)(starting_chunk_offsets);
        assert_eq!(
            trimmed_chunk_offsets, expected_chunk_offsets,
            "trimmed chunk offsets don't match what's expected"
        );
    }

    macro_rules! test_trim_duration {
        ($(
            $name:ident {
                @track(
                    $( $track_field:ident: $track_value:expr ),+,
                ),
                $( $field:ident: $value:expr ),+,
            }
        )*) => {
            $(
                #[test]
                fn $name() {
                    test_trim_duration!(
                        @inner $($field: $value),+,
                        @track $($track_field: $track_value),+,
                    );
                }
            )*
        };

        (
            @inner $( $field:ident: $value:expr ),+,
            @track $( $track_field:ident: $track_value:expr ),+,
        ) => {
            let test_case = TrimDurationTestCase::builder()
                .$( $field($value) ).+
                .build();
            let track = create_test_track()
                .movie_timescale(test_case.movie_timescale)
                .media_timescale(test_case.media_timescale)
                .$( $track_field($track_value) ).+
                .build();
            test_trim_duration(track, test_case);
        };
    }

    mod test_trim_duration {
        use super::*;

        test_trim_duration!(
            trim_start_11_seconds {
                @track(
                    duration: Duration::from_secs(100),
                ),
                range: TrimDurationRange::builder()
                    .start_bound(Bound::Included(Duration::from_secs(0)))
                    .end_bound(Bound::Excluded(Duration::from_secs(11))).build(),
                expected_duration: Duration::from_secs(89),
                expected_chunk_offsets: |mut orig_offsets: Vec<u64>| {
                    // 1 second per sample = 11 samples trimmed
                    // 2 samples per chunk = 5 chunks trimmed + the first sample of the 6th chunk
                    orig_offsets.drain(..5);
                    let removed_sample_size = 256; // the default
                    orig_offsets[0] += removed_sample_size; // the 6th chunk offset should be moved forward
                    orig_offsets
                },
            }
        );
    }
}
