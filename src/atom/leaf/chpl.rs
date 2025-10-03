use bon::bon;
use derive_more::Deref;
use futures_io::AsyncRead;
use std::{fmt, time::Duration};

use crate::{
    atom::{
        util::{read_to_end, DebugList},
        FourCC,
    },
    parser::ParseAtom,
    writer::SerializeAtom,
    ParseError,
};

pub const CHPL: &[u8; 4] = b"chpl";

#[derive(Default, Clone, Deref)]
pub struct ChapterEntries(Vec<ChapterEntry>);

impl ChapterEntries {
    pub fn into_vec(self) -> Vec<ChapterEntry> {
        self.0
    }
}

impl FromIterator<ChapterEntry> for ChapterEntries {
    fn from_iter<T: IntoIterator<Item = ChapterEntry>>(iter: T) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl From<Vec<ChapterEntry>> for ChapterEntries {
    fn from(entries: Vec<ChapterEntry>) -> Self {
        ChapterEntries(entries)
    }
}

impl fmt::Debug for ChapterEntries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&DebugList::new(self.0.iter(), 10), f)
    }
}

/// Chapter entry containing start time and title
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterEntry {
    /// Start time of the chapter in 100-nanosecond units
    pub start_time: u64,
    /// Chapter title as UTF-8 string
    pub title: String,
}

#[bon]
impl ChapterEntry {
    #[builder]
    pub fn new(#[builder(into, finish_fn)] title: String, start_time: Duration) -> Self {
        // convert to 100-nanosecond units
        let start_time = (start_time.as_nanos() / 100).min(u128::from(u64::MAX)) as u64;
        ChapterEntry { start_time, title }
    }
}

/// Chapter List Atom - contains chapter information for media
#[derive(Debug, Clone)]
pub struct ChapterListAtom {
    /// Version of the chpl atom format (1)
    pub version: u8,
    pub flags: [u8; 3],
    pub reserved: [u8; 4],
    /// List of chapter entries
    pub chapters: ChapterEntries,
}

impl Default for ChapterListAtom {
    fn default() -> Self {
        Self {
            version: 1,
            flags: [0u8; 3],
            reserved: [0u8; 4],
            chapters: Default::default(),
        }
    }
}

impl ChapterListAtom {
    pub fn new(chapters: impl Into<ChapterEntries>) -> Self {
        Self {
            version: 1,
            flags: [0u8; 3],
            reserved: [0u8; 4],
            chapters: chapters.into(),
        }
    }

    pub fn replace_chapters(&mut self, chapters: impl Into<ChapterEntries>) {
        self.chapters = chapters.into();
    }
}

impl ParseAtom for ChapterListAtom {
    async fn parse<R: AsyncRead + Unpin + Send>(
        atom_type: FourCC,
        reader: R,
    ) -> Result<Self, ParseError> {
        if atom_type != CHPL {
            return Err(ParseError::new_unexpected_atom(atom_type, CHPL));
        }
        let data = read_to_end(reader).await?;
        parser::parse_chpl_data(&data)
    }
}

impl SerializeAtom for ChapterListAtom {
    fn atom_type(&self) -> FourCC {
        FourCC(*CHPL)
    }

    fn into_body_bytes(self) -> Vec<u8> {
        serializer::serialize_chpl_data(self)
    }
}

mod serializer {
    use crate::atom::chpl::{ChapterEntries, ChapterEntry};

    use super::ChapterListAtom;

    pub fn serialize_chpl_data(atom: ChapterListAtom) -> Vec<u8> {
        vec![
            version(atom.version),
            flags(atom.flags),
            reserved(atom.reserved),
            chapters(atom.chapters),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn version(version: u8) -> Vec<u8> {
        vec![version]
    }

    fn flags(flags: [u8; 3]) -> Vec<u8> {
        flags.to_vec()
    }

    fn reserved(reserved: [u8; 4]) -> Vec<u8> {
        reserved.to_vec()
    }

    fn chapters(chapters: ChapterEntries) -> Vec<u8> {
        vec![
            vec![u8::try_from(chapters.len())
                .expect("there must be no more than {u8::MAX} chapter entries")],
            chapters.0.into_iter().flat_map(chapter).collect(),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    fn chapter(chapter: ChapterEntry) -> Vec<u8> {
        vec![start_time(chapter.start_time), title(chapter.title)]
            .into_iter()
            .flatten()
            .collect()
    }

    fn start_time(start_time: u64) -> Vec<u8> {
        start_time.to_be_bytes().to_vec()
    }

    fn title(title: String) -> Vec<u8> {
        let title_bytes = title.into_bytes();
        vec![
            vec![u8::try_from(title_bytes.len()).expect("title length must not exceed {u8::MAX}")],
            title_bytes,
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

mod parser {
    use winnow::{
        binary::{be_u64, length_and_then, u8},
        combinator::{repeat, seq, trace},
        error::StrContext,
        token::rest,
        ModalResult, Parser,
    };

    use super::ChapterListAtom;
    use crate::atom::{
        chpl::{ChapterEntries, ChapterEntry},
        util::parser::{byte_array, stream, version, Stream},
    };

    pub fn parse_chpl_data(input: &[u8]) -> Result<ChapterListAtom, crate::ParseError> {
        parse_chpl_data_inner
            .parse(stream(input))
            .map_err(crate::ParseError::from_winnow)
    }

    fn parse_chpl_data_inner(input: &mut Stream<'_>) -> ModalResult<ChapterListAtom> {
        trace(
            "chpl",
            seq!(ChapterListAtom {
                version: version.verify(|v| *v == 1),
                flags: byte_array.context(StrContext::Label("flags")),
                reserved: byte_array.context(StrContext::Label("reserved")),
                chapters: chapters.context(StrContext::Label("chapters")),
            })
            .context(StrContext::Label("chpl")),
        )
        .parse_next(input)
    }

    fn chapters(input: &mut Stream<'_>) -> ModalResult<ChapterEntries> {
        trace("chapters", move |input: &mut Stream<'_>| {
            let chapter_count = u8
                .context(StrContext::Label("chapter_count"))
                .parse_next(input)?;
            repeat(chapter_count as usize, chapter)
                .map(ChapterEntries)
                .parse_next(input)
        })
        .parse_next(input)
    }

    fn chapter(input: &mut Stream<'_>) -> ModalResult<ChapterEntry> {
        trace(
            "chapter",
            seq!(ChapterEntry {
                start_time: be_u64.context(StrContext::Label("start_time")),
                title: length_and_then(
                    u8,
                    rest.try_map(|buf: &[u8]| String::from_utf8(buf.to_vec()))
                )
                .context(StrContext::Label("title")),
                // _: null_bytes, // discard trailing null bytes
            })
            .context(StrContext::Label("chapter")),
        )
        .parse_next(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::test_utils::test_atom_roundtrip_sync;

    /// Test round-trip for all available chpl test data files
    #[test]
    fn test_chpl_roundtrip() {
        test_atom_roundtrip_sync::<ChapterListAtom>(CHPL);
    }
}
