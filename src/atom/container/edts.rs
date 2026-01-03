use std::fmt;

use crate::{
    atom::{
        atom_ref::{unwrap_atom_data, AtomRef, AtomRefMut},
        elst::ELST,
        EditListAtom,
    },
    Atom, AtomData, FourCC,
};

pub const EDTS: FourCC = FourCC::new(b"edts");

#[derive(Clone, Copy)]
pub struct EdtsAtomRef<'a>(pub(crate) AtomRef<'a>);

impl fmt::Debug for EdtsAtomRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EdtsAtomRef").finish()
    }
}

impl<'a> EdtsAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    pub fn edit_list(&self) -> Option<&'a EditListAtom> {
        let atom = self.0.find_child(ELST)?;
        match atom.data.as_ref()? {
            AtomData::EditList(data) => Some(data),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct EdtsAtomRefMut<'a>(pub(crate) AtomRefMut<'a>);

impl EdtsAtomRefMut<'_> {
    /// Finds or creates the ELST atom
    pub fn edit_list(&mut self) -> &mut EditListAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(ELST)
                .insert_data(AtomData::EditList(EditListAtom::default()))
                .call(),
            AtomData::EditList,
        )
    }
}
