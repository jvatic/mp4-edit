use crate::{
    atom::{atom_ref::AtomRefMut, elst::ELST, EditListAtom},
    unwrap_atom_data, AtomData,
};

pub const EDTS: &[u8; 4] = b"edts";

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
