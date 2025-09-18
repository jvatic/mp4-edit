use winnow::{
    binary::{be_u32, u8},
    combinator::trace,
    error::{StrContext, StrContextValue},
    Bytes, LocatingSlice, Parser,
};

use crate::FourCC;

pub type Stream<'i> = LocatingSlice<&'i Bytes>;

pub fn stream(b: &[u8]) -> Stream<'_> {
    LocatingSlice::new(Bytes::new(b))
}

pub fn fourcc(input: &mut Stream<'_>) -> winnow::ModalResult<FourCC> {
    trace(
        "fourcc",
        (
            u8.context(StrContext::Label("[0]")),
            u8.context(StrContext::Label("[1]")),
            u8.context(StrContext::Label("[2]")),
            u8.context(StrContext::Label("[3]")),
        )
            .map(|(a, b, c, d)| FourCC([a, b, c, d]))
            .context(StrContext::Label("fourcc")),
    )
    .parse_next(input)
}

pub fn version1(input: &mut Stream<'_>) -> winnow::ModalResult<u8> {
    trace("version", u8)
        .context(StrContext::Label("version"))
        .parse_next(input)
}

pub fn usize_be_u32(input: &mut Stream<'_>) -> winnow::ModalResult<usize> {
    trace(
        "usize_be_u32",
        be_u32
            .map(|s| s as usize)
            .context(StrContext::Expected(StrContextValue::Description("be u32"))),
    )
    .parse_next(input)
}

pub fn flags3(input: &mut Stream<'_>) -> winnow::ModalResult<[u8; 3]> {
    trace(
        "flags",
        (
            u8.context(StrContext::Label("[0]")),
            u8.context(StrContext::Label("[1]")),
            u8.context(StrContext::Label("[2]")),
        ),
    )
    .map(|(a, b, c)| [a, b, c])
    .context(StrContext::Label("flags"))
    .parse_next(input)
}

pub mod combinators {
    use winnow::error::ParserError;
    use winnow::stream::{Location, Stream};
    use winnow::Parser;

    pub fn with_len<I, O, E, ParseNext>(parser: ParseNext) -> impls::WithLen<ParseNext, I, O, E>
    where
        I: Stream + Location,
        E: ParserError<I>,
        ParseNext: Parser<I, O, E>,
    {
        impls::WithLen {
            parser,
            i: Default::default(),
            o: Default::default(),
            e: Default::default(),
        }
    }

    mod impls {
        use winnow::stream::{Location, Stream};
        use winnow::*;

        pub struct WithLen<F, I, O, E>
        where
            F: Parser<I, O, E>,
            I: Stream + Location,
        {
            pub(crate) parser: F,
            pub(crate) i: core::marker::PhantomData<I>,
            pub(crate) o: core::marker::PhantomData<O>,
            pub(crate) e: core::marker::PhantomData<E>,
        }

        impl<F, I, O, E> Parser<I, (O, usize), E> for WithLen<F, I, O, E>
        where
            F: Parser<I, O, E>,
            I: Stream + Location,
        {
            #[inline]
            fn parse_next(&mut self, input: &mut I) -> Result<(O, usize), E> {
                let start = input.current_token_start();
                self.parser.parse_next(input).map(move |output| {
                    let end = input.previous_token_end();
                    (output, end - start)
                })
            }
        }
    }
}
