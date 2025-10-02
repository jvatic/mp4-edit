use crate::{
    atom::{
        atom_ref::{AtomRef, AtomRefMut},
        smhd::SMHD,
        GmhdAtomRef, GmhdAtomRefMut, StblAtomRef, StblAtomRefMut, DINF, GMHD, STBL,
    },
    Atom,
};

pub const MINF: &[u8; 4] = b"minf";

#[derive(Debug, Clone, Copy)]
pub struct MinfAtomRef<'a>(pub(crate) AtomRef<'a>);

impl<'a> MinfAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    pub fn header(&self) -> GmhdAtomRef<'a> {
        let atom = self.0.find_child(GMHD);
        GmhdAtomRef(AtomRef(atom))
    }

    /// Finds the STBL atom
    pub fn sample_table(&self) -> StblAtomRef<'a> {
        let atom = self.0.find_child(STBL);
        StblAtomRef(AtomRef(atom))
    }
}

#[derive(Debug)]
pub struct MinfAtomRefMut<'a>(pub(crate) AtomRefMut<'a>);

impl<'a> MinfAtomRefMut<'a> {
    pub fn as_ref(&self) -> MinfAtomRef<'_> {
        MinfAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> MinfAtomRef<'a> {
        MinfAtomRef(self.0.into_ref())
    }

    pub fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.into_children()
    }

    /// Finds or inserts the GMHD atom
    pub fn header(&mut self) -> GmhdAtomRefMut<'_> {
        GmhdAtomRefMut(self.0.find_or_insert_child(GMHD).insert_index(0).call())
    }

    /// Finds or inserts the STBL atom
    pub fn sample_table(&mut self) -> StblAtomRefMut<'_> {
        StblAtomRefMut(
            self.0
                .find_or_insert_child(STBL)
                .insert_after(vec![DINF, SMHD])
                .call(),
        )
    }

    /// Finds the STBL atom
    pub fn into_sample_table(self) -> Option<StblAtomRefMut<'a>> {
        let atom = self.0.into_child(STBL)?;
        Some(StblAtomRefMut(AtomRefMut(atom)))
    }
}
