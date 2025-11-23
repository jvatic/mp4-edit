use std::{
    fmt::Debug,
    ops::{Range, RangeBounds},
    time::Duration,
};

use bon::bon;

use crate::{
    atom::{
        atom_ref, hdlr::HandlerType, mdhd::MDHD, mvhd::MVHD, AtomHeader, MovieHeaderAtom,
        TrakAtomRef, TrakAtomRefMut, UserDataAtomRefMut, TRAK, UDTA,
    },
    unwrap_atom_data, Atom, AtomData,
};

pub const MOOV: &[u8; 4] = b"moov";

#[derive(Debug, Clone, Copy)]
pub struct MoovAtomRef<'a>(pub(crate) atom_ref::AtomRef<'a>);

impl<'a> MoovAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    pub fn header(&self) -> Option<&'a MovieHeaderAtom> {
        let atom = self.children().find(|a| a.header.atom_type == MVHD)?;
        match atom.data.as_ref()? {
            AtomData::MovieHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Iterate through TRAK atoms
    pub fn into_tracks_iter(self) -> impl Iterator<Item = TrakAtomRef<'a>> {
        self.children()
            .filter(|a| a.header.atom_type == TRAK)
            .map(TrakAtomRef::new)
    }

    /// Iterate through TRAK atoms with handler type Audio
    pub fn into_audio_track_iter(self) -> impl Iterator<Item = TrakAtomRef<'a>> {
        self.into_tracks_iter().filter(|trak| {
            matches!(
                trak.media()
                    .handler_reference()
                    .map(|hdlr| &hdlr.handler_type),
                Some(HandlerType::Audio)
            )
        })
    }
}

#[derive(Debug)]
pub struct MoovAtomRefMut<'a>(pub(crate) atom_ref::AtomRefMut<'a>);

impl<'a> MoovAtomRefMut<'a> {
    pub fn as_ref(&self) -> MoovAtomRef<'_> {
        MoovAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> MoovAtomRef<'a> {
        MoovAtomRef(self.0.into_ref())
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children()
    }

    /// Finds or inserts MVHD atom
    pub fn header(&mut self) -> &'_ mut MovieHeaderAtom {
        unwrap_atom_data!(
            self.0.find_or_insert_child(MVHD).call(),
            AtomData::MovieHeader,
        )
    }

    /// Finds or inserts UDTA atom
    pub fn user_data(&mut self) -> UserDataAtomRefMut<'_> {
        UserDataAtomRefMut(
            self.0
                .find_or_insert_child(UDTA)
                .insert_after(vec![TRAK, MVHD])
                .call(),
        )
    }

    pub fn tracks(&mut self) -> impl Iterator<Item = TrakAtomRefMut<'_>> {
        self.0
            .children()
            .filter(|a| a.header.atom_type == TRAK)
            .map(TrakAtomRefMut::new)
    }

    /// Iterate through TRAK atoms with handler type Audio
    pub fn audio_tracks(&mut self) -> impl Iterator<Item = TrakAtomRefMut<'_>> {
        self.tracks().filter(|trak| {
            matches!(
                trak.as_ref()
                    .media()
                    .handler_reference()
                    .map(|hdlr| &hdlr.handler_type),
                Some(HandlerType::Audio)
            )
        })
    }

    /// Retains only the TRAK atoms specified by the predicate
    pub fn tracks_retain<P>(&mut self, mut pred: P) -> &mut Self
    where
        P: FnMut(TrakAtomRef) -> bool,
    {
        self.0
             .0
            .children
            .retain(|a| a.header.atom_type != TRAK || pred(TrakAtomRef::new(a)));
        self
    }
}

#[bon]
impl<'a> MoovAtomRefMut<'a> {
    /// Trim duration from tracks.
    ///
    /// See also [`Self::retain_duration`].
    #[builder(finish_fn(name = "trim"), builder_type = TrimDuration)]
    pub fn trim_duration(
        &mut self,
        from_start: Option<Duration>,
        from_end: Option<Duration>,
    ) -> &mut Self {
        use std::ops::Bound;
        let start_duration = from_start.map(|d| (Bound::Unbounded, Bound::Excluded(d)));
        let end_duration = from_end.map(|d| {
            let d = self.header().duration().saturating_sub(d);
            (Bound::Included(d), Bound::Unbounded)
        });
        let trim_ranges = vec![start_duration, end_duration]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        self.trim_duration_ranges(&trim_ranges)
    }

    /// Retains given duration range, trimming everything before and after.
    ///
    /// See also [`Self::trim_duration`].
    pub fn retain_duration(&mut self, range: Range<Duration>) -> &mut Self {
        use std::ops::Bound;
        let trim_ranges = vec![
            (Bound::Unbounded, Bound::Included(range.start)),
            (Bound::Excluded(range.end), Bound::Unbounded),
        ];
        self.trim_duration_ranges(&trim_ranges)
    }

    fn trim_duration_ranges<R>(&mut self, trim_ranges: &[R]) -> &mut Self
    where
        R: RangeBounds<Duration> + Clone + Debug,
    {
        let movie_timescale = u64::from(self.header().timescale);
        let trimmed_duration = self
            .tracks()
            .map(|mut trak| trak.trim_duration(movie_timescale, trim_ranges))
            .min();
        if let Some(trimmed_duration) = trimmed_duration {
            self.header().update_duration(|d| d - trimmed_duration);
        }
        self
    }
}

#[bon]
impl<'a, 'b, S: trim_duration::State> TrimDuration<'a, 'b, S> {
    #[builder(finish_fn(name = "trim"), builder_type = TrimDurationRanges)]
    pub fn ranges<R>(
        self,
        #[builder(start_fn)] ranges: impl IntoIterator<Item = R>,
    ) -> &'b mut MoovAtomRefMut<'a>
    where
        R: RangeBounds<Duration> + Clone + Debug,
        S::FromEnd: trim_duration::IsUnset,
        S::FromStart: trim_duration::IsUnset,
    {
        self.self_receiver
            .trim_duration_ranges(&ranges.into_iter().collect::<Vec<_>>())
    }
}

#[bon]
impl<'a> MoovAtomRefMut<'a> {
    /// Adds trak atom to moov
    #[builder]
    pub fn add_track(
        &mut self,
        #[builder(default = Vec::new())] children: Vec<Atom>,
    ) -> TrakAtomRefMut<'_> {
        let trak = Atom::builder()
            .header(AtomHeader::new(*TRAK))
            .children(children)
            .build();
        let index = self.0.get_insert_position().after(vec![TRAK, MDHD]).call();
        TrakAtomRefMut(self.0.insert_child(index, trak))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Bound;

    use crate::{
        atom::{
            container::{DINF, MDIA, MINF, MOOV, STBL, TRAK},
            dref::{DataReferenceAtom, DataReferenceEntry, DREF},
            ftyp::{FileTypeAtom, FTYP},
            hdlr::{HandlerReferenceAtom, HandlerType, HDLR},
            mdhd::{MediaHeaderAtom, MDHD},
            mvhd::{MovieHeaderAtom, MVHD},
            smhd::{SoundMediaHeaderAtom, SMHD},
            stco_co64::{ChunkOffsetAtom, STCO},
            stsc::{SampleToChunkAtom, SampleToChunkEntry, STSC},
            stsd::{SampleDescriptionTableAtom, STSD},
            stsz::{SampleSizeAtom, STSZ},
            stts::{TimeToSampleAtom, TimeToSampleEntry, STTS},
            tkhd::{TrackHeaderAtom, TKHD},
            util::scaled_duration,
            Atom, AtomHeader,
        },
        parser::Metadata,
        FourCC,
    };
    use std::time::Duration;

    #[bon::builder(finish_fn(name = "build"))]
    fn create_test_metadata(
        movie_timescale: u32,
        media_timescale: u32,
        duration: Duration,
    ) -> Metadata {
        let atoms = vec![
            // Create ftyp atom
            Atom::builder()
                .header(AtomHeader::new(*FTYP))
                .data(
                    FileTypeAtom::builder()
                        .major_brand(*b"isom")
                        .minor_version(512)
                        .compatible_brands(
                            vec![*b"isom", *b"iso2", *b"mp41"]
                                .into_iter()
                                .map(FourCC::from)
                                .collect::<Vec<_>>(),
                        )
                        .build(),
                )
                .build(),
            // Create moov atom with a single track with complex sample data
            Atom::builder()
                .header(AtomHeader::new(*MOOV))
                .children(vec![
                    // Movie header (mvhd)
                    Atom::builder()
                        .header(AtomHeader::new(*MVHD))
                        .data(
                            MovieHeaderAtom::builder()
                                .timescale(movie_timescale)
                                .duration(scaled_duration(duration, movie_timescale as u64))
                                .next_track_id(2)
                                .build(),
                        )
                        .build(),
                    // Track (trak) with complex sample data
                    create_test_track(movie_timescale, media_timescale, duration),
                ])
                .build(),
        ];

        Metadata::new(atoms.into())
    }

    fn create_test_track(movie_timescale: u32, media_timescale: u32, duration: Duration) -> Atom {
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
                // Media (mdia) with complex sample data
                create_test_media(media_timescale, duration),
            ])
            .build()
    }

    fn create_test_media(media_timescale: u32, duration: Duration) -> Atom {
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
                    .data(
                        HandlerReferenceAtom::builder()
                            .handler_type(HandlerType::Audio)
                            .name("SoundHandler".to_string())
                            .build(),
                    )
                    .build(),
                // Media information (minf)
                create_test_media_info(duration, media_timescale),
            ])
            .build()
    }

    fn create_test_media_info(duration: Duration, media_timescale: u32) -> Atom {
        Atom::builder()
            .header(AtomHeader::new(*MINF))
            .children(vec![
                // Sound media information header (smhd)
                Atom::builder()
                    .header(AtomHeader::new(*SMHD))
                    .data(SoundMediaHeaderAtom::default())
                    .build(),
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
                // Sample table (stbl) with complex data
                create_test_sample_table(duration, media_timescale),
            ])
            .build()
    }

    fn create_test_sample_table(duration: Duration, media_timescale: u32) -> Atom {
        // Create one sample per second with 2 samples per chunk
        let duration_secs = duration.as_secs() as u32;
        let total_samples = duration_secs; // One sample per second
        let samples_per_chunk = 2u32;

        // Calculate number of chunks needed
        let total_chunks = (total_samples + samples_per_chunk - 1) / samples_per_chunk;

        // bytes per sample
        const SAMPLE_SIZE: usize = 256;

        // Create chunk offsets
        let mut chunk_offsets = Vec::new();
        let mut current_offset = 1000u64;
        for _ in 0..total_chunks {
            chunk_offsets.push(current_offset);
            current_offset += samples_per_chunk as u64 * SAMPLE_SIZE as u64;
        }

        // Create sample sizes (256 bytes per sample)
        let sample_sizes: Vec<u32> = vec![SAMPLE_SIZE as u32; total_samples as usize];

        // Create single sample-to-chunk entry (all chunks have 2 samples)
        let stsc_entries = vec![SampleToChunkEntry::builder()
            .first_chunk(1)
            .samples_per_chunk(samples_per_chunk)
            .sample_description_index(1)
            .build()];

        Atom::builder()
            .header(AtomHeader::new(*STBL))
            .children(vec![
                // Sample Description (stsd)
                Atom::builder()
                    .header(AtomHeader::new(*STSD))
                    .data(SampleDescriptionTableAtom::default())
                    .build(),
                // Time to Sample (stts) - each sample represents 1 second
                Atom::builder()
                    .header(AtomHeader::new(*STTS))
                    .data(
                        TimeToSampleAtom::builder()
                            .entry(
                                TimeToSampleEntry::builder()
                                    .sample_count(total_samples)
                                    .sample_duration(media_timescale) // 1 second per sample
                                    .build(),
                            )
                            .build(),
                    )
                    .build(),
                // Sample to Chunk (stsc) - 2 samples per chunk
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

    fn test_moov_trim_duration<F>(test_case: F)
    where
        F: FnOnce() -> TrimDurationTestCase,
    {
        let test_case = test_case();

        let movie_timescale = 1_000;
        let media_timescale = 10_000;

        // Create fresh metadata for each test case
        let mut metadata = create_test_metadata()
            .movie_timescale(movie_timescale)
            .media_timescale(media_timescale)
            .duration(test_case.original_duration)
            .build();

        // Perform the trim operation with the range bounds
        let range = (test_case.start_bound, test_case.end_bound);
        metadata
            .moov_mut()
            .trim_duration()
            .ranges(vec![range])
            .trim();

        // Verify movie header duration was updated
        let new_movie_duration = metadata.moov().header().map(|h| h.duration).unwrap_or(0);
        let expected_movie_duration = scaled_duration(
            test_case.expected_remaining_duration,
            movie_timescale as u64,
        );
        assert_eq!(
            new_movie_duration, expected_movie_duration,
            "Movie duration should match expected",
        );

        // Verify track header duration was updated
        let new_track_duration = metadata
            .moov()
            .into_tracks_iter()
            .next()
            .and_then(|t| t.header().map(|h| h.duration))
            .unwrap_or(0);
        let expected_track_duration = scaled_duration(
            test_case.expected_remaining_duration,
            movie_timescale as u64,
        );
        assert_eq!(
            new_track_duration, expected_track_duration,
            "Track duration should match expected",
        );

        // Verify media header duration was updated
        let new_media_duration = metadata
            .moov()
            .into_tracks_iter()
            .next()
            .map(|t| t.media().header().map(|h| h.duration).unwrap_or(0))
            .unwrap_or(0);
        let expected_media_duration = scaled_duration(
            test_case.expected_remaining_duration,
            media_timescale as u64,
        );
        assert_eq!(
            new_media_duration, expected_media_duration,
            "Media duration should match expected",
        );

        // Verify sample table structure is still valid
        let track = metadata.moov().into_tracks_iter().next().unwrap();
        let stbl = track.media().media_information().sample_table();

        // Validate that all required sample table atoms exist
        let stts = stbl
            .time_to_sample()
            .expect("Time-to-sample atom should exist");
        let stsc = stbl
            .sample_to_chunk()
            .expect("Sample-to-chunk atom should exist");
        let stsz = stbl.sample_size().expect("Sample-size atom should exist");
        let stco = stbl.chunk_offset().expect("Chunk-offset atom should exist");

        // Calculate total samples from sample sizes
        let total_samples = stsz.sample_count() as u32;
        if test_case.expected_remaining_duration != Duration::ZERO {
            assert!(total_samples > 0, "Sample table should have samples",);
        }

        // Validate time-to-sample consistency
        let stts_total_samples: u32 = stts.entries.iter().map(|entry| entry.sample_count).sum();
        assert_eq!(
            stts_total_samples, total_samples,
            "Time-to-sample total samples should match sample size count",
        );

        // Validate sample-to-chunk references
        let chunk_count = stco.chunk_count() as u32;
        assert!(chunk_count > 0, "Should have at least one chunk",);

        // Verify all chunk references in stsc are valid
        for entry in stsc.entries.iter() {
            assert!(
                entry.first_chunk >= 1 && entry.first_chunk <= chunk_count,
                "Sample-to-chunk first_chunk {} should be between 1 and {}",
                entry.first_chunk,
                chunk_count,
            );
            assert!(
                entry.samples_per_chunk > 0,
                "Sample-to-chunk samples_per_chunk should be > 0",
            );
        }

        // Verify expected duration consistency with time-to-sample
        let total_duration: u64 = stts
            .entries
            .iter()
            .map(|entry| entry.sample_count as u64 * entry.sample_duration as u64)
            .sum();
        let expected_duration_scaled = scaled_duration(
            test_case.expected_remaining_duration,
            media_timescale as u64,
        );

        assert_eq!(
            total_duration, expected_duration_scaled,
            "Sample table total duration should match the expected duration",
        );
    }

    struct TrimDurationTestCase {
        original_duration: Duration,
        start_bound: Bound<Duration>,
        end_bound: Bound<Duration>,
        expected_remaining_duration: Duration,
    }

    macro_rules! test_moov_trim_duration {
        ($($name:ident => $test_case:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    test_moov_trim_duration($test_case);
                }
            )*
        };
    }

    test_moov_trim_duration!(
        trim_start_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::ZERO),
            end_bound: Bound::Included(Duration::from_secs(2)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_end_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(8)),
            end_bound: Bound::Included(Duration::from_secs(10)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_middle_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(4)),
            end_bound: Bound::Included(Duration::from_secs(6)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_middle_included_start_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(2)),
            end_bound: Bound::Included(Duration::from_secs(4)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_middle_excluded_start_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_millis(10_000),
            start_bound: Bound::Excluded(Duration::from_millis(1_999)),
            end_bound: Bound::Included(Duration::from_millis(4_000)),
            expected_remaining_duration: Duration::from_millis(8_000),
        },
        trim_middle_excluded_end_2_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Included(Duration::from_secs(1)),
            end_bound: Bound::Excluded(Duration::from_secs(3)),
            expected_remaining_duration: Duration::from_secs(8),
        },
        trim_start_unbounded_5_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(10),
            start_bound: Bound::Unbounded,
            end_bound: Bound::Included(Duration::from_secs(5)),
            expected_remaining_duration: Duration::from_secs(5),
        },
        trim_end_unbounded_6_seconds => || TrimDurationTestCase {
            original_duration: Duration::from_secs(100),
            start_bound: Bound::Included(Duration::from_secs(94)),
            end_bound: Bound::Unbounded,
            expected_remaining_duration: Duration::from_secs(94),
        },
        trim_unbounded => || TrimDurationTestCase {
            original_duration: Duration::from_secs(100),
            start_bound: Bound::Unbounded,
            end_bound: Bound::Unbounded,
            expected_remaining_duration: Duration::ZERO,
        },
    );
}
