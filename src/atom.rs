pub mod chpl;
pub mod dref;
pub mod elst;
pub mod free;
pub mod ftyp;
pub mod gmhd;
pub mod hdlr;
pub mod ilst;
pub mod mdhd;
pub mod meta;
pub mod mvhd;
pub mod sbgp;
pub mod sgpd;
pub mod smhd;
pub mod stco_co64;
pub mod stsc;
pub mod stsd;
pub mod stsz;
pub mod stts;
pub mod tkhd;
pub mod tref;
mod util;

pub use self::{
    chpl::ChapterListAtom, dref::DataReferenceAtom, elst::EditListAtom, free::FreeAtom,
    ftyp::FileTypeAtom, gmhd::GenericMediaHeaderAtom, hdlr::HandlerReferenceAtom,
    ilst::ItemListAtom, mdhd::MediaHeaderAtom, meta::MetadataAtom, mvhd::MovieHeaderAtom,
    sbgp::SampleToGroupAtom, sgpd::SampleGroupDescriptionAtom, smhd::SoundMediaHeaderAtom,
    stco_co64::ChunkOffsetAtom, stsc::SampleToChunkAtom, stsd::SampleDescriptionTableAtom,
    stsz::SampleSizeAtom, stts::TimeToSampleAtom, tkhd::TrackHeaderAtom,
    tref::TrackReferenceAtom, util::FourCC,
};

#[derive(Clone)]
pub struct RawData(pub Vec<u8>);

impl std::fmt::Debug for RawData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[u8; {}]", self.0.len())
    }
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub atom_type: FourCC,
    pub size: u64,
    pub offset: u64,
    // Optionally, store child atoms if any
    pub children: Vec<Atom>,
    pub data: Option<AtomData>,
}

#[derive(Debug, Clone)]
pub enum AtomData {
    FileType(FileTypeAtom),
    MovieHeader(MovieHeaderAtom),
    TrackHeader(TrackHeaderAtom),
    EditList(EditListAtom),
    MediaHeader(MediaHeaderAtom),
    HandlerReference(HandlerReferenceAtom),
    GenericMediaHeader(GenericMediaHeaderAtom),
    ItemList(ItemListAtom),
    Metadata(MetadataAtom),
    SoundMediaHeader(SoundMediaHeaderAtom),
    SampleDescriptionTable(SampleDescriptionTableAtom),
    TrackReference(TrackReferenceAtom),
    DataReference(DataReferenceAtom),
    SampleSize(SampleSizeAtom),
    ChunkOffset(ChunkOffsetAtom),
    TimeToSample(TimeToSampleAtom),
    SampleToChunk(SampleToChunkAtom),
    SampleToGroup(SampleToGroupAtom),
    SampleGroupDescription(SampleGroupDescriptionAtom),
    ChapterList(ChapterListAtom),
    Free(FreeAtom),
    RawData(RawData),
}

// Implement From traits for all atom types
impl From<FileTypeAtom> for AtomData {
    fn from(atom: FileTypeAtom) -> Self {
        AtomData::FileType(atom)
    }
}

impl From<MovieHeaderAtom> for AtomData {
    fn from(atom: MovieHeaderAtom) -> Self {
        AtomData::MovieHeader(atom)
    }
}

impl From<TrackHeaderAtom> for AtomData {
    fn from(atom: TrackHeaderAtom) -> Self {
        AtomData::TrackHeader(atom)
    }
}

impl From<EditListAtom> for AtomData {
    fn from(atom: EditListAtom) -> Self {
        AtomData::EditList(atom)
    }
}

impl From<MediaHeaderAtom> for AtomData {
    fn from(atom: MediaHeaderAtom) -> Self {
        AtomData::MediaHeader(atom)
    }
}

impl From<HandlerReferenceAtom> for AtomData {
    fn from(atom: HandlerReferenceAtom) -> Self {
        AtomData::HandlerReference(atom)
    }
}

impl From<GenericMediaHeaderAtom> for AtomData {
    fn from(atom: GenericMediaHeaderAtom) -> Self {
        AtomData::GenericMediaHeader(atom)
    }
}

impl From<ItemListAtom> for AtomData {
    fn from(atom: ItemListAtom) -> Self {
        AtomData::ItemList(atom)
    }
}

impl From<MetadataAtom> for AtomData {
    fn from(atom: MetadataAtom) -> Self {
        AtomData::Metadata(atom)
    }
}

impl From<SoundMediaHeaderAtom> for AtomData {
    fn from(atom: SoundMediaHeaderAtom) -> Self {
        AtomData::SoundMediaHeader(atom)
    }
}

impl From<SampleDescriptionTableAtom> for AtomData {
    fn from(atom: SampleDescriptionTableAtom) -> Self {
        AtomData::SampleDescriptionTable(atom)
    }
}

impl From<TrackReferenceAtom> for AtomData {
    fn from(atom: TrackReferenceAtom) -> Self {
        AtomData::TrackReference(atom)
    }
}

impl From<DataReferenceAtom> for AtomData {
    fn from(atom: DataReferenceAtom) -> Self {
        AtomData::DataReference(atom)
    }
}

impl From<SampleSizeAtom> for AtomData {
    fn from(atom: SampleSizeAtom) -> Self {
        AtomData::SampleSize(atom)
    }
}

impl From<ChunkOffsetAtom> for AtomData {
    fn from(atom: ChunkOffsetAtom) -> Self {
        AtomData::ChunkOffset(atom)
    }
}

impl From<TimeToSampleAtom> for AtomData {
    fn from(atom: TimeToSampleAtom) -> Self {
        AtomData::TimeToSample(atom)
    }
}

impl From<SampleToChunkAtom> for AtomData {
    fn from(atom: SampleToChunkAtom) -> Self {
        AtomData::SampleToChunk(atom)
    }
}

impl From<SampleToGroupAtom> for AtomData {
    fn from(atom: SampleToGroupAtom) -> Self {
        AtomData::SampleToGroup(atom)
    }
}

impl From<ChapterListAtom> for AtomData {
    fn from(atom: ChapterListAtom) -> Self {
        AtomData::ChapterList(atom)
    }
}

impl From<SampleGroupDescriptionAtom> for AtomData {
    fn from(atom: SampleGroupDescriptionAtom) -> Self {
        AtomData::SampleGroupDescription(atom)
    }
}

impl From<FreeAtom> for AtomData {
    fn from(atom: FreeAtom) -> Self {
        AtomData::Free(atom)
    }
}

impl From<RawData> for AtomData {
    fn from(data: RawData) -> Self {
        AtomData::RawData(data)
    }
}
