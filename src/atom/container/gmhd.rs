use crate::{
    atom::atom_ref::{AtomRef, AtomRefMut},
    Atom,
};

pub const GMHD: &[u8; 4] = b"gmhd";

#[derive(Debug, Clone, Copy)]
pub struct GmhdAtomRef<'a>(pub(crate) AtomRef<'a>);

impl<'a> GmhdAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    // TODO: gmin
    // TODO: text
}

#[derive(Debug)]
pub struct GmhdAtomRefMut<'a>(pub(crate) AtomRefMut<'a>);

impl<'a> GmhdAtomRefMut<'a> {
    pub fn as_ref(&self) -> GmhdAtomRef<'_> {
        GmhdAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> GmhdAtomRef<'a> {
        GmhdAtomRef(self.0.into_ref())
    }

    pub fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.into_children()
    }

    // TODO: gmin
    // TODO: text
}
