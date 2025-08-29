use std::fmt::Debug;

use crate::atom::atom_ref;
use crate::unwrap_atom_data;
use crate::{
    atom::{
        stco_co64::{ChunkOffsetAtom, STCO},
        stsc::{SampleToChunkAtom, STSC},
        stsd::{SampleDescriptionTableAtom, STSD},
        stsz::{SampleSizeAtom, STSZ},
        stts::{TimeToSampleAtom, STTS},
    },
    Atom, AtomData,
};

pub const STBL: &[u8; 4] = b"stbl";

#[derive(Debug)]
pub struct StblAtomRef<'a>(pub(crate) atom_ref::AtomRef<'a>);

impl<'a> StblAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    /// Finds the STSD atom
    pub fn sample_description(&self) -> Option<&'a SampleDescriptionTableAtom> {
        let atom = self.0.find_child(STSD)?;
        match atom.data.as_ref()? {
            AtomData::SampleDescriptionTable(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STTS atom
    pub fn time_to_sample(&self) -> Option<&'a TimeToSampleAtom> {
        let atom = self.0.find_child(STTS)?;
        match atom.data.as_ref()? {
            AtomData::TimeToSample(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSC atom
    pub fn sample_to_chunk(&self) -> Option<&'a SampleToChunkAtom> {
        let atom = self.0.find_child(STSC)?;
        match atom.data.as_ref()? {
            AtomData::SampleToChunk(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STSZ atom
    pub fn sample_size(&self) -> Option<&'a SampleSizeAtom> {
        let atom = self.0.find_child(STSZ)?;
        match atom.data.as_ref()? {
            AtomData::SampleSize(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the STCO atom
    pub fn chunk_offset(&self) -> Option<&'a ChunkOffsetAtom> {
        let atom = self.0.find_child(STCO)?;
        match atom.data.as_ref()? {
            AtomData::ChunkOffset(data) => Some(data),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct StblAtomRefMut<'a>(pub(crate) atom_ref::AtomRefMut<'a>);

impl<'a> StblAtomRefMut<'a> {
    pub fn as_ref(&self) -> StblAtomRef<'_> {
        StblAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> StblAtomRef<'a> {
        StblAtomRef(self.0.into_ref())
    }

    pub fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.into_children()
    }

    /// Finds or inserts the STSD atom
    pub fn sample_description(&mut self) -> &'_ mut SampleDescriptionTableAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STSD)
                .insert_data(AtomData::SampleDescriptionTable(
                    SampleDescriptionTableAtom::default(),
                ))
                .call(),
            AtomData::SampleDescriptionTable,
        )
    }

    /// Finds or inserts the STTS atom
    pub fn time_to_sample(&mut self) -> &mut TimeToSampleAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STTS)
                .insert_after(vec![STTS, STSD])
                .insert_data(AtomData::TimeToSample(TimeToSampleAtom::default()))
                .call(),
            AtomData::TimeToSample,
        )
    }

    /// Finds or inserts the STSC atom
    pub fn sample_to_chunk(&mut self) -> &mut SampleToChunkAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STSC)
                .insert_after(vec![STTS, STSD])
                .insert_data(AtomData::SampleToChunk(SampleToChunkAtom::default()))
                .call(),
            AtomData::SampleToChunk,
        )
    }

    /// Finds or inserts the STSZ atom
    pub fn sample_size(&mut self) -> &mut SampleSizeAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STSZ)
                .insert_after(vec![STSC, STSD])
                .insert_data(AtomData::SampleSize(SampleSizeAtom::default()))
                .call(),
            AtomData::SampleSize,
        )
    }

    /// Finds or inserts the STCO atom
    pub fn chunk_offset(&mut self) -> &mut ChunkOffsetAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(STCO)
                .insert_after(vec![STSZ, STSD])
                .insert_data(AtomData::ChunkOffset(ChunkOffsetAtom::default()))
                .call(),
            AtomData::ChunkOffset,
        )
    }
}
