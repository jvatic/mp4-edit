use std::fmt::Debug;

use winnow::{
    binary::{be_i32, be_u16, be_u32, be_u64, length_and_then, u8},
    combinator::trace,
    error::{ParserError, StrContext, StrContextValue},
    token::rest,
    Bytes, LocatingSlice, ModalResult, Parser,
};

use crate::FourCC;

pub type Stream<'i> = LocatingSlice<&'i Bytes>;

pub fn stream(b: &[u8]) -> Stream<'_> {
    LocatingSlice::new(Bytes::new(b))
}

pub fn fourcc(input: &mut Stream<'_>) -> winnow::ModalResult<FourCC> {
    trace(
        "fourcc",
        (byte_array)
            .map(|buf| FourCC(buf))
            .context(StrContext::Label("fourcc")),
    )
    .parse_next(input)
}

pub fn version(input: &mut Stream<'_>) -> winnow::ModalResult<u8> {
    trace("version", u8)
        .context(StrContext::Label("version"))
        .parse_next(input)
}

pub fn version_0_or_1(input: &mut Stream<'_>) -> ModalResult<u8> {
    trace(
        "version_0_or_1",
        version
            .verify(|version| *version <= 1)
            .context(StrContext::Expected(StrContextValue::Description(
                "expected version 0 or 1",
            ))),
    )
    .parse_next(input)
}

pub fn be_u32_as_usize(input: &mut Stream<'_>) -> winnow::ModalResult<usize> {
    trace(
        "usize_be_u32",
        be_u32
            .map(|s| s as usize)
            .context(StrContext::Expected(StrContextValue::Description("be u32"))),
    )
    .parse_next(input)
}

pub fn be_u32_as_u64(input: &mut Stream<'_>) -> ModalResult<u64> {
    trace(
        "be_u32_as_u64",
        be_u32
            .map(|s| s as u64)
            .context(StrContext::Expected(StrContextValue::Description("be u32"))),
    )
    .parse_next(input)
}

pub fn be_u32_as<'i, T, E>(input: &mut Stream<'i>) -> ModalResult<T>
where
    T: TryFrom<u32, Error = E> + 'i,
    E: std::error::Error + Send + Sync + 'static,
{
    trace(
        "be_u32_as",
        be_u32
            .try_map(|s| T::try_from(s))
            .context(StrContext::Expected(StrContextValue::Description("be u32"))),
    )
    .parse_next(input)
}

pub fn be_i32_as<'i, T, E>(input: &mut Stream<'i>) -> ModalResult<T>
where
    T: TryFrom<i32, Error = E> + 'i,
    E: std::error::Error + Send + Sync + 'static,
{
    trace(
        "be_i32_as",
        be_i32
            .try_map(|s| T::try_from(s))
            .context(StrContext::Expected(StrContextValue::Description("be i32"))),
    )
    .parse_next(input)
}

pub fn be_u24(input: &mut Stream<'_>) -> ModalResult<u32> {
    trace(
        "be_u24",
        byte_array::<3>.map(|buf| u32::from_be_bytes([0, buf[0], buf[1], buf[2]])),
    )
    .parse_next(input)
}

pub fn atom_size(input: &mut Stream<'_>) -> ModalResult<usize> {
    trace("atom_size", move |input: &mut Stream| {
        let mut size = be_u32_as_u64.parse_next(input)?;
        if size == 1 {
            size = be_u64.parse_next(input)?;
        }
        Ok(size as usize)
    })
    .parse_next(input)
}

pub fn flags3(input: &mut Stream<'_>) -> winnow::ModalResult<[u8; 3]> {
    trace("flags", byte_array)
        .context(StrContext::Label("flags"))
        .parse_next(input)
}

/// Parses a u8 len, and then a UTF8 string with that len
pub fn pascal_string(input: &mut Stream<'_>) -> ModalResult<String> {
    trace("pascal_string", length_and_then(u8, utf8_string)).parse_next(input)
}

/// Parses a UTF8 string from the remainder of the buffer
pub fn utf8_string(input: &mut Stream<'_>) -> ModalResult<String> {
    trace(
        "utf8_string",
        rest.try_map(|data: &[u8]| String::from_utf8(data.to_vec()))
            .context(StrContext::Expected(StrContextValue::Description(
                "UTF8 string",
            ))),
    )
    .parse_next(input)
}

pub fn byte_array<const N: usize>(input: &mut Stream<'_>) -> winnow::ModalResult<[u8; N]> {
    trace("byte_array", fixed_array(u8)).parse_next(input)
}

pub fn rest_vec<'i>(input: &mut Stream<'i>) -> ModalResult<Vec<u8>> {
    trace("rest_vec", move |input: &mut Stream<'i>| {
        let data = rest.parse_next(input)?;
        Ok(data.to_vec())
    })
    .parse_next(input)
}

pub fn rest_vec1<'i>(input: &mut Stream<'i>) -> ModalResult<Vec<u8>> {
    trace("rest_vec1", move |input: &mut Stream<'i>| {
        let data = rest
            .verify(|data: &[u8]| !data.is_empty())
            .parse_next(input)?;
        Ok(data.to_vec())
    })
    .parse_next(input)
}

pub fn fixed_array<'i, const N: usize, Input, Output, Error, ParseNext>(
    mut parser: ParseNext,
) -> impl Parser<Input, [Output; N], Error> + 'i
where
    Input: winnow::stream::Stream + 'i,
    ParseNext: Parser<Input, Output, Error> + 'i,
    Error: ParserError<Input> + 'i,
    Output: Debug + 'i,
{
    trace("fixed_array", move |input: &mut Input| {
        let mut list: Vec<Output> = Vec::with_capacity(N);
        for _ in 0..N {
            list.push(parser.by_ref().complete_err().parse_next(input)?);
        }
        let out: [Output; N] = list.try_into().unwrap();
        Ok(out)
    })
}

pub const FIXED_POINT_16X16_SCALE: f32 = 65536.0;

pub fn fixed_point_16x16(input: &mut Stream<'_>) -> ModalResult<f32> {
    trace(
        "fixed_point_16_x_16",
        be_u32.map(|v| (v as f32) / FIXED_POINT_16X16_SCALE),
    )
    .parse_next(input)
}

pub const FIXED_POINT_8X8_SCALE: f32 = 256.0;

pub fn fixed_point_8x8(input: &mut Stream<'_>) -> ModalResult<f32> {
    trace(
        "fixed_point_8x8",
        be_u16.map(|v| (v as f32) / FIXED_POINT_8X8_SCALE),
    )
    .parse_next(input)
}

/// Read be_u32 from between 1 and 4 bytes using VLQ
pub fn variable_length_be_u32(input: &mut Stream<'_>) -> ModalResult<u32> {
    variable_length_quantity::<_, 4>(input)
}

fn variable_length_quantity<T, const N: usize>(input: &mut Stream<'_>) -> ModalResult<T>
where
    T: From<u8> + std::ops::Shl<u8, Output = T> + std::ops::BitOr<T, Output = T>,
{
    let mut length = T::from(0);
    for _ in 0..N {
        let byte = u8.parse_next(input)?;
        length = (length << 7) | T::from(byte & 0b0111_1111);
        if (byte & 0b1000_0000) == 0 {
            break;
        }
    }
    Ok(length)
}

pub mod combinators {
    use winnow::combinator::trace;
    use winnow::error::ParserError;
    use winnow::stream::{Location, Stream, StreamIsPartial, ToUsize, UpdateSlice};
    use winnow::token::take;
    use winnow::Parser;

    pub fn inclusive_length_and_then<Input, Output, Count, Error, CountParser, ParseNext>(
        mut count: CountParser,
        mut parser: ParseNext,
    ) -> impl Parser<Input, Output, Error>
    where
        Input: StreamIsPartial + Stream + Location + UpdateSlice + Clone,
        Count: ToUsize,
        CountParser: Parser<Input, Count, Error>,
        ParseNext: Parser<Input, Output, Error>,
        Error: ParserError<Input>,
    {
        trace("inclusive_length_and_then", move |i: &mut Input| {
            let size = with_len(count.by_ref().map(|c| c.to_usize()))
                .map(|(a, b)| a.saturating_sub(b))
                .complete_err()
                .parse_next(i)?;
            let data = take(size).parse_next(i)?;
            let mut data = Input::update_slice(i.clone(), data);
            let _ = data.complete();
            let o = parser.by_ref().complete_err().parse_next(&mut data)?;
            Ok(o)
        })
    }

    fn with_len<I, O, E, ParseNext>(mut parser: ParseNext) -> impl Parser<I, (O, usize), E>
    where
        I: Stream + Location,
        E: ParserError<I>,
        ParseNext: Parser<I, O, E>,
    {
        trace("with_len", move |input: &mut I| {
            let start = input.current_token_start();
            parser.by_ref().parse_next(input).map(move |output| {
                let end = input.previous_token_end();
                (output, end - start)
            })
        })
    }
}
