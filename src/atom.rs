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
pub(crate) mod util;

pub mod containers {
    pub const MOOV: &[u8; 4] = b"moov";
    pub const MFRA: &[u8; 4] = b"mfra";
    pub const UDTA: &[u8; 4] = b"udta";
    pub const TRAK: &[u8; 4] = b"trak";
    pub const EDTS: &[u8; 4] = b"edts";
    pub const MDIA: &[u8; 4] = b"mdia";
    pub const MINF: &[u8; 4] = b"minf";
    pub const DINF: &[u8; 4] = b"dinf";
    pub const STBL: &[u8; 4] = b"stbl";
    pub const MOOF: &[u8; 4] = b"moof";
    pub const TRAF: &[u8; 4] = b"traf";
    pub const SINF: &[u8; 4] = b"sinf";
    pub const SCHI: &[u8; 4] = b"schi";

    pub const META: &[u8; 4] = b"meta";
    pub const META_VERSION_FLAGS_SIZE: usize = 4;
}

pub use self::{
    chpl::ChapterListAtom, dref::DataReferenceAtom, elst::EditListAtom, free::FreeAtom,
    ftyp::FileTypeAtom, gmhd::GenericMediaHeaderAtom, hdlr::HandlerReferenceAtom,
    ilst::ItemListAtom, mdhd::MediaHeaderAtom, mvhd::MovieHeaderAtom, sbgp::SampleToGroupAtom,
    sgpd::SampleGroupDescriptionAtom, smhd::SoundMediaHeaderAtom, stco_co64::ChunkOffsetAtom,
    stsc::SampleToChunkAtom, stsd::SampleDescriptionTableAtom, stsz::SampleSizeAtom,
    stts::TimeToSampleAtom, tkhd::TrackHeaderAtom, tref::TrackReferenceAtom, util::FourCC,
};

#[derive(Clone)]
pub struct RawData(pub Vec<u8>);

impl std::fmt::Debug for RawData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[u8; {}]", self.0.len())
    }
}

#[derive(Debug, Clone)]
pub struct AtomHeader {
    pub atom_type: FourCC,
    pub offset: usize,
    pub header_size: usize,
    pub data_size: usize,
}

impl AtomHeader {
    pub fn location(&self) -> (usize, usize) {
        (self.offset, self.header_size + self.data_size)
    }

    pub fn atom_size(&self) -> usize {
        self.header_size + self.data_size
    }
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub header: AtomHeader,
    pub data: Option<AtomData>,
    pub children: Vec<Atom>,
}

impl Atom {
    /// Recursively retains only the atoms that satisfy the predicate,
    /// one level of depth at a time (least to most nested).
    pub fn children_flat_retain_mut<P>(&mut self, mut pred: P)
    where
        P: FnMut(&mut Atom) -> bool,
    {
        let mut current_level = vec![self];

        while !current_level.is_empty() {
            let mut next_level = Vec::new();

            for atom in current_level {
                // Apply retain to this atom's children
                atom.children.retain_mut(|child| pred(child));

                // Collect remaining children for next level processing
                for child in &mut atom.children {
                    next_level.push(child);
                }
            }

            current_level = next_level;
        }
    }
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

impl From<RawData> for Vec<u8> {
    fn from(data: RawData) -> Self {
        data.0
    }
}

impl From<AtomData> for Vec<u8> {
    fn from(data: AtomData) -> Self {
        use AtomData::*;
        match data {
            FileType(atom) => atom.into(),
            MovieHeader(atom) => atom.into(),
            TrackHeader(atom) => atom.into(),
            EditList(atom) => atom.into(),
            MediaHeader(atom) => atom.into(),
            HandlerReference(atom) => atom.into(),
            GenericMediaHeader(atom) => atom.into(),
            ItemList(atom) => atom.into(),
            SoundMediaHeader(atom) => atom.into(),
            SampleDescriptionTable(atom) => atom.into(),
            TrackReference(atom) => atom.into(),
            DataReference(atom) => atom.into(),
            SampleSize(atom) => atom.into(),
            ChunkOffset(atom) => atom.into(),
            TimeToSample(atom) => atom.into(),
            SampleToChunk(atom) => atom.into(),
            SampleToGroup(atom) => atom.into(),
            SampleGroupDescription(atom) => atom.into(),
            ChapterList(atom) => atom.into(),
            Free(atom) => atom.into(),
            RawData(data) => data.into(),
        }
    }
}
