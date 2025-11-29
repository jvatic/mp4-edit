/*!
 * This example demonstrates copying an MP4 file. When using a seekable input, it moves metadata to the start of the file for fast-start (streaming) opimization. Using a non-seekable input requires the file to already be fast-start.
 */

use anyhow::{anyhow, Context};
use futures_util::io::{BufReader, BufWriter};
use indicatif::ProgressBar;
use std::env;
use tokio::{
    fs,
    io::{self},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use mp4_edit::{
    atom::FourCC,
    parser::{ReadCapability, MDAT},
    Mp4Writer, Parser,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input_mp4> <output_mp4>", args[0]);
        eprintln!("  Use '-' for stdin/stdout");
        std::process::exit(1);
    }

    let input_name = &args[1];
    let output_name = &args[2];

    eprintln!("Copying {} into {}", input_name, output_name);

    if input_name == "-" {
        eprintln!("parsing stdin as readonly");

        let input = io::stdin().compat();
        let input_reader = BufReader::new(input);
        let parser = Parser::new(input_reader);
        let metadata = parser
            .parse_metadata()
            .await
            .context("failed to parse metadata from stdin")?;

        process_mp4_copy(metadata, output_name).await?;
    } else {
        eprintln!("parsing file as seekable");

        let file = fs::File::open(input_name).await?;
        let input_reader = file.compat();
        let parser = Parser::new_seekable(input_reader);
        let metadata = parser
            .parse_metadata()
            .await
            .context("failed to parse metadata from input file")?;

        process_mp4_copy(metadata, output_name).await?;
    }

    Ok(())
}

async fn process_mp4_copy<R, C>(
    metadata: mp4_edit::parser::MdatParser<R, C>,
    output_name: &str,
) -> anyhow::Result<()>
where
    C: ReadCapability,
    R: futures_util::io::AsyncRead + Unpin + Send,
{
    if output_name == "-" {
        process_mp4_copy_to_stdout(metadata).await?;
    } else {
        process_mp4_copy_to_file(metadata, output_name).await?;
    }
    Ok(())
}

async fn process_mp4_copy_to_file<R, C>(
    metadata: mp4_edit::parser::MdatParser<R, C>,
    output_name: &str,
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
    process_mp4_copy_inner(metadata, writer).await?;
    Ok(())
}

async fn process_mp4_copy_to_stdout<R, C>(
    metadata: mp4_edit::parser::MdatParser<R, C>,
) -> anyhow::Result<()>
where
    C: ReadCapability,
    R: futures_util::io::AsyncRead + Unpin + Send,
{
    eprintln!("writing to stdout");
    let output = io::stdout().compat_write();
    let output = BufWriter::new(output);
    let writer = Mp4Writer::new(output);
    process_mp4_copy_inner(metadata, writer).await?;
    Ok(())
}

async fn process_mp4_copy_inner<R, C, W>(
    metadata: mp4_edit::parser::MdatParser<R, C>,
    mut mp4_writer: Mp4Writer<W>,
) -> anyhow::Result<()>
where
    C: ReadCapability,
    R: futures_util::io::AsyncRead + Unpin + Send,
    W: futures_util::io::AsyncWrite + Unpin + Send,
{
    let mut input_metadata = metadata;

    let mut metadata = input_metadata.clone();

    let mdat_size = metadata
        .update_chunk_offsets()
        .context("error updating chunk offsets")?
        .total_size as usize;

    let new_metadata_size = metadata.metadata_size();
    let mdat_content_offset = new_metadata_size + 8;

    let progress_bar = ProgressBar::new((new_metadata_size + mdat_size) as u64);

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
