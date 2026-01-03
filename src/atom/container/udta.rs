use crate::{
    atom::{
        atom_ref::{unwrap_atom_data, AtomRefMut},
        chpl::CHPL,
        container::META,
        ChapterListAtom,
    },
    AtomData, FourCC,
};

pub const UDTA: FourCC = FourCC::new(b"udta");

pub struct UserDataAtomRefMut<'a>(pub(crate) AtomRefMut<'a>);

impl UserDataAtomRefMut<'_> {
    /// Finds or inserts CHPL atom
    pub fn chapter_list(&mut self) -> &'_ mut ChapterListAtom {
        unwrap_atom_data!(
            self.0
                .find_or_insert_child(CHPL)
                .insert_after(vec![META])
                .insert_data(AtomData::ChapterList(ChapterListAtom::default()))
                .call(),
            AtomData::ChapterList,
        )
    }
}
