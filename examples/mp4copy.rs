/*!
 * This example demonstrates copying and/or trimming an MP4 file. When using a seekable input, it moves metadata to the start of the file for fast-start (streaming) opimization. Using a non-seekable input requires the file to already be fast-start.
 */

use anyhow::{anyhow, Context};
use clap::Parser as ClapParser;
use futures_util::io::{BufReader, BufWriter};
use indicatif::ProgressBar;
use std::time::Duration;
use tokio::{
    fs,
    io::{self},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use mp4_edit::{
    atom::FourCC,
    parser::{ReadCapability, MDAT},
    Mp4Writer,
};

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the input mp4, use `-` for stdin
    input_mp4: String,

    /// Path to the output mp4, use `-` for stdout
    output_mp4: String,

    #[command(subcommand)]
    command: Option<SubCommand>,
}

#[derive(clap::Subcommand, Debug)]
enum SubCommand {
    #[cfg(feature = "experimental-trim")]
    Trim(TrimArgs),
    #[cfg(feature = "experimental-trim")]
    Retain(RetainArgs),
}

/// Trim the start and/or end of the mp4
#[derive(clap::Args, Debug)]
#[group(required = true, multiple = true)]
struct TrimArgs {
    /// Duration to trim from the start (e.g. 10s)
    #[arg(short, long, value_parser = humantime::parse_duration)]
    start: Option<Duration>,

    /// Duration to trim from the end (e.g. 1m20s)
    #[arg(short, long, value_parser = humantime::parse_duration)]
    end: Option<Duration>,
}

/// Retain a clip of the mp4
#[derive(clap::Args, Debug)]
#[group(required = true, multiple = true)]
struct RetainArgs {
    /// Position clip starts at (e.g. 1h10m32s)
    #[arg(short = 'o', long, value_parser = humantime::parse_duration)]
    from_offset: Option<Duration>,

    /// Duration of clip to retain (e.g. 30m)
    #[arg(short, long, value_parser = humantime::parse_duration)]
    duration: Duration,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let input_name = &args.input_mp4;
    let output_name = &args.output_mp4;
    let sub_command = args.command;

    eprintln!("Copying {} into {}", input_name, output_name);

    if input_name == "-" {
        eprintln!("parsing stdin as readonly");

        let input = io::stdin().compat();
        let input_reader = BufReader::new(input);
        let parser = mp4_edit::parser::Parser::new(input_reader);
        let metadata = parser
            .parse_metadata()
            .await
            .context("failed to parse metadata from stdin")?;

        process_mp4_copy(metadata, output_name, sub_command).await?;
    } else {
        eprintln!("parsing file as seekable");

        let file = fs::File::open(input_name).await?;
        let input_reader = file.compat();
        let parser = mp4_edit::parser::Parser::new_seekable(input_reader);
        let metadata = parser
            .parse_metadata()
            .await
            .context("failed to parse metadata from input file")?;

        process_mp4_copy(metadata, output_name, sub_command).await?;
    }

    Ok(())
}

async fn process_mp4_copy<R, C>(
    metadata: mp4_edit::parser::MdatParser<R, C>,
    output_name: &str,
    sub_command: Option<SubCommand>,
) -> anyhow::Result<()>
where
    C: ReadCapability,
    R: futures_util::io::AsyncRead + Unpin + Send,
{
    if output_name == "-" {
        process_mp4_copy_to_stdout(metadata, sub_command).await?;
    } else {
        process_mp4_copy_to_file(metadata, output_name, sub_command).await?;
    }
    Ok(())
}

async fn process_mp4_copy_to_file<R, C>(
    metadata: mp4_edit::parser::MdatParser<R, C>,
    output_name: &str,
    sub_command: Option<SubCommand>,
) -> anyhow::Result<()>
where
    C: ReadCapability,
    R: futures_util::io::AsyncRead + Unpin + Send,
{
    eprintln!("writing to file {output_name:#?}");
    let output = fs::File::create(output_name)
        .await
        .context("failed to create output file")?
        .compat_write();
    let output = BufWriter::new(output);
    let writer = Mp4Writer::new(output);
    process_mp4_copy_inner(metadata, writer, sub_command).await?;
    Ok(())
}

async fn process_mp4_copy_to_stdout<R, C>(
    metadata: mp4_edit::parser::MdatParser<R, C>,
    sub_command: Option<SubCommand>,
) -> anyhow::Result<()>
where
    C: ReadCapability,
    R: futures_util::io::AsyncRead + Unpin + Send,
{
    eprintln!("writing to stdout");
    let output = io::stdout().compat_write();
    let output = BufWriter::new(output);
    let writer = Mp4Writer::new(output);
    process_mp4_copy_inner(metadata, writer, sub_command).await?;
    Ok(())
}

async fn process_mp4_copy_inner<R, C, W>(
    metadata: mp4_edit::parser::MdatParser<R, C>,
    mut mp4_writer: Mp4Writer<W>,
    sub_command: Option<SubCommand>,
) -> anyhow::Result<()>
where
    C: ReadCapability,
    R: futures_util::io::AsyncRead + Unpin + Send,
    W: futures_util::io::AsyncWrite + Unpin + Send,
{
    let mut input_metadata = metadata;

    if let Some(sub_command) = sub_command {
        match sub_command {
            #[cfg(feature = "experimental-trim")]
            SubCommand::Trim(args) => {
                if let Some(start) = args.start {
                    eprintln!("trimming {} from start", humantime::format_duration(start));
                }
                if let Some(end) = args.end {
                    eprintln!("trimming {} from end", humantime::format_duration(end));
                }
                trim_duration(&mut input_metadata, args)?;
            }
            #[cfg(feature = "experimental-trim")]
            SubCommand::Retain(args) => {
                eprintln!(
                    "retaining {} to {}",
                    humantime::format_duration(args.from_offset.unwrap_or_default()),
                    humantime::format_duration(
                        args.from_offset.unwrap_or_default() + args.duration
                    )
                );
                retain_duration(&mut input_metadata, args)?;
            }
        }
    }

    let mut metadata = input_metadata.clone();

    let mdat_size = metadata
        .update_chunk_offsets()
        .context("error updating chunk offsets")?
        .total_size as usize;

    let new_metadata_size = metadata.metadata_size();
    let mdat_content_offset = new_metadata_size + 8;

    let progress_bar = ProgressBar::new((mdat_content_offset + mdat_size) as u64);

    // Write metadata atoms (all neccesary changes have been made already)
    for (i, atom) in metadata.atoms_iter().enumerate() {
        mp4_writer.write_atom(atom.clone()).await.with_context(|| {
            format!("failed to write atom {} ({})", i + 1, atom.header.atom_type)
        })?;
        progress_bar.set_position(mp4_writer.current_offset() as u64);
    }

    mp4_writer.flush().await.context("metadata flush")?;

    // Write MDAT header
    mp4_writer
        .write_atom_header(FourCC::from(*MDAT), mdat_size as usize)
        .await
        .context("error writing mdat placeholder header")?;

    if input_metadata.mdat_header().is_none() {
        return Err(anyhow!("mdat atom not found"));
    }

    assert_eq!(
        mp4_writer.current_offset(),
        mdat_content_offset,
        "incorrect mdat_content_offset"
    );

    // Copy and write sample data
    let mut chunk_idx = 0;
    let mut chunk_parser = input_metadata.chunks()?;
    while let Some(chunk) = chunk_parser.read_next_chunk().await? {
        for (i, sample) in chunk.samples().enumerate() {
            let data = sample.data.to_vec();

            mp4_writer.write_raw(&data).await.context(format!(
                "error writing sample {i:02} data in chunk {chunk_idx:02}"
            ))?;

            progress_bar.set_position(mp4_writer.current_offset() as u64);
        }

        chunk_idx += 1;
    }

    mp4_writer.flush().await.context("final flush")?;

    progress_bar.finish();

    assert_eq!(
        mp4_writer.current_offset(),
        mdat_content_offset + mdat_size as usize,
        "mdat header has incorrect size"
    );

    Ok(())
}

#[cfg(feature = "experimental-trim")]
fn trim_duration(metadata: &mut mp4_edit::parser::Metadata, args: TrimArgs) -> anyhow::Result<()> {
    metadata
        .moov_mut()
        .trim_duration()
        .maybe_from_start(args.start)
        .maybe_from_end(args.end)
        .trim();
    Ok(())
}

#[cfg(feature = "experimental-trim")]
fn retain_duration(
    metadata: &mut mp4_edit::parser::Metadata,
    args: RetainArgs,
) -> anyhow::Result<()> {
    metadata
        .moov_mut()
        .retain_duration()
        .maybe_from_offset(args.from_offset)
        .duration(args.duration)
        .retain();
    Ok(())
}
