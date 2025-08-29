use crate::{
    atom::{
        atom_ref::{AtomRef, AtomRefMut},
        hdlr::HDLR,
        mdhd::MDHD,
        HandlerReferenceAtom, MediaHeaderAtom, MinfAtomRef, MinfAtomRefMut, MINF,
    },
    unwrap_atom_data, Atom, AtomData,
};

pub const MDIA: &[u8; 4] = b"mdia";

#[derive(Debug, Clone, Copy)]
pub struct MdiaAtomRef<'a>(pub(crate) AtomRef<'a>);

impl<'a> MdiaAtomRef<'a> {
    pub fn children(&self) -> impl Iterator<Item = &'a Atom> {
        self.0.children()
    }

    /// Finds the MDHD atom
    pub fn header(&self) -> Option<&'a MediaHeaderAtom> {
        let atom = self.0.find_child(MDHD)?;
        match atom.data.as_ref()? {
            AtomData::MediaHeader(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the HDLR atom
    pub fn handler_reference(&self) -> Option<&'a HandlerReferenceAtom> {
        let atom = self.0.find_child(HDLR)?;
        match atom.data.as_ref()? {
            AtomData::HandlerReference(data) => Some(data),
            _ => None,
        }
    }

    /// Finds the MINF atom
    pub fn media_information(&self) -> MinfAtomRef<'a> {
        let atom = self.0.find_child(MINF);
        MinfAtomRef(AtomRef(atom))
    }
}

#[derive(Debug)]
pub struct MdiaAtomRefMut<'a>(pub(crate) AtomRefMut<'a>);

impl<'a> MdiaAtomRefMut<'a> {
    pub fn as_ref(&self) -> MdiaAtomRef<'_> {
        MdiaAtomRef(self.0.as_ref())
    }

    pub fn into_ref(self) -> MdiaAtomRef<'a> {
        MdiaAtomRef(self.0.into_ref())
    }

    pub fn children(&mut self) -> impl Iterator<Item = &'_ mut Atom> {
        self.0.children()
    }

    pub fn into_children(self) -> impl Iterator<Item = &'a mut Atom> {
        self.0.into_children()
    }

    /// Finds or inserts the MDHD atom
    pub fn header(&mut self) -> &mut MediaHeaderAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(MDHD)
                .insert_data(AtomData::MediaHeader(MediaHeaderAtom::default()))
                .call(),
            AtomData::MediaHeader,
        )
    }

    /// Finds or inserts the HDLR atom
    pub fn handler_reference(&mut self) -> &mut HandlerReferenceAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(HDLR)
                .insert_data(AtomData::HandlerReference(HandlerReferenceAtom::default()))
                .call(),
            AtomData::HandlerReference,
        )
    }

    /// Finds or inserts the MINF atom
    pub fn media_information(&mut self) -> MinfAtomRefMut<'_> {
        MinfAtomRefMut(
            self.0
                .find_or_insert_child(MINF)
                .insert_after(vec![HDLR, MDHD])
                .call(),
        )
    }

    /// Finds the MINF atom
    pub fn into_media_information(self) -> Option<MinfAtomRefMut<'a>> {
        let atom = self.0.into_child(MINF)?;
        Some(MinfAtomRefMut(AtomRefMut(atom)))
    }
}
