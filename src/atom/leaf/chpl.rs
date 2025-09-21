use bon::bon;
use derive_more::Deref;
use futures_io::AsyncRead;
use std::{fmt, time::Duration};

use crate::{
    atom::{
        util::{read_to_end, DebugEllipsis},
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
        if self.0.len() <= 10 {
            return f.debug_list().entries(self.0.iter()).finish();
        }
        f.debug_list()
            .entries(self.0.iter().take(10))
            .entry(&DebugEllipsis(Some(self.0.len() - 10)))
            .finish()
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
    /// Flags for the chpl atom (usually all zeros)
    pub flags: [u8; 3],
    /// List of chapter entries
    pub chapters: ChapterEntries,
}

impl Default for ChapterListAtom {
    fn default() -> Self {
        Self {
            version: 1,
            flags: [0u8; 3],
            chapters: Default::default(),
        }
    }
}

impl ChapterListAtom {
    pub fn new(chapters: impl Into<ChapterEntries>) -> Self {
        Self {
            version: 1,
            flags: [0u8; 3],
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

    fn chapters(chapters: ChapterEntries) -> Vec<u8> {
        chapters.0.into_iter().flat_map(chapter).collect()
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
        vec![title.into_bytes(), vec![0x00]]
            .into_iter()
            .flatten()
            .collect()
    }
}

mod parser {
    use winnow::{
        binary::{be_u64, u8},
        combinator::{repeat, seq, trace},
        error::StrContext,
        token::{literal, take_until},
        ModalResult, Parser,
    };

    use super::ChapterListAtom;
    use crate::atom::{
        chpl::{ChapterEntries, ChapterEntry},
        util::parser::{flags3, stream, version, Stream},
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
                flags: flags3,
                _: reserved,
                chapters: chapters,
            })
            .context(StrContext::Label("chpl")),
        )
        .parse_next(input)
    }

    fn reserved(input: &mut Stream<'_>) -> ModalResult<()> {
        trace(
            "reserved",
            repeat(8, u8).context(StrContext::Label("reserved")),
        )
        .parse_next(input)
    }

    fn chapters(input: &mut Stream<'_>) -> ModalResult<ChapterEntries> {
        trace(
            "chapters",
            repeat(1.., chapter)
                .map(ChapterEntries)
                .context(StrContext::Label("chapters")),
        )
        .parse_next(input)
    }

    fn chapter(input: &mut Stream<'_>) -> ModalResult<ChapterEntry> {
        trace(
            "chapter",
            seq!(ChapterEntry {
                start_time: be_u64.context(StrContext::Label("start_time")),
                title: take_until(1.., 0x00)
                    .try_map(|buf: &[u8]| String::from_utf8(buf.to_vec()))
                    .context(StrContext::Label("title")),
                _: literal(0x00), // discard the literal, TODO: clean this up so above expr consumes this
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
