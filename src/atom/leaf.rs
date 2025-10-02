/*!
 * Atoms without children.
 */

pub mod chpl;
pub mod dref;
pub mod elst;
pub mod free;
pub mod ftyp;
pub mod gmin;
pub mod hdlr;
pub mod ilst;
pub mod mdhd;
pub mod mvhd;
pub mod sbgp;
pub mod sgpd;
pub mod smhd;
pub mod stco_co64;
pub mod stsc;
pub mod stsd;
pub mod stsz;
pub mod stts;
pub mod text;
pub mod tkhd;
pub mod tref;

pub use self::{
    chpl::ChapterListAtom, dref::DataReferenceAtom, elst::EditListAtom, free::FreeAtom,
    ftyp::FileTypeAtom, gmin::BaseMediaInfoAtom, hdlr::HandlerReferenceAtom, ilst::ItemListAtom,
    mdhd::MediaHeaderAtom, mvhd::MovieHeaderAtom, smhd::SoundMediaHeaderAtom,
    stco_co64::ChunkOffsetAtom, stsc::SampleToChunkAtom, stsd::SampleDescriptionTableAtom,
    stsz::SampleSizeAtom, stts::TimeToSampleAtom, text::TextMediaInfoAtom, tkhd::TrackHeaderAtom,
    tref::TrackReferenceAtom,
};
