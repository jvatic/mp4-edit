use anyhow::{anyhow, Context};
use futures_util::io::{BufReader, BufWriter};
use progress_bar::pb::ProgressBar;
use std::env;
use tokio::{
    fs,
    io::{self},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use mp4_edit::{atom::FourCC, parser::MDAT, Mp4Writer, Parser};

async fn create_output_file(output_name: &str) -> anyhow::Result<fs::File> {
    if output_name == "-" {
        anyhow::bail!("Stdout output not supported in this version");
    } else {
        let file = fs::File::create(output_name).await?;
        Ok(file)
    }
}

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

    eprintln!("🎬 Copying {} into {}", input_name, output_name);

    if input_name == "-" {
        eprintln!("parsing stdin as readonly");
        let input = Box::new(io::stdin());
        let input_reader = BufReader::new(input.compat());
        let parser = Parser::new(input_reader);
        let input_metadata = parser
            .parse_metadata()
            .await
            .context("Failed to parse metadata from stdin")?;

        process_mp4_copy(input_metadata, output_name).await?;
    } else {
        eprintln!("parsing file as seekable");
        let file = fs::File::open(input_name).await?;
        let input_reader = file.compat();
        let parser = Parser::new(input_reader);
        let input_metadata = parser
            .parse_metadata_seek()
            .await
            .context("Failed to parse metadata from input file")?;

        process_mp4_copy(input_metadata, output_name).await?;
    }

    Ok(())
}

async fn process_mp4_copy<R>(
    metadata: mp4_edit::parser::MdatParser<R>,
    output_name: &str,
) -> anyhow::Result<()>
where
    R: futures_util::io::AsyncRead + Unpin + Send,
{
    let mut input_metadata = metadata;
    let mut metadata = input_metadata.clone();

    let mdat_size = metadata
        .tracks_iter()
        .map(|trak| trak.size())
        .sum::<usize>();

    let new_metadata_size = metadata.metadata_size();

    let mdat_content_offset = new_metadata_size + 8;

    let mut progress_bar = ProgressBar::new_with_eta(new_metadata_size + mdat_size);

    metadata
        .update_chunk_offsets()
        .context("error updating chunk offsets")?;

    // Open output file for writing
    let output_file = create_output_file(output_name).await?;
    let output_writer = output_file.compat_write();
    let output_writer = BufWriter::new(output_writer);

    let mut mp4_writer = Mp4Writer::new(output_writer);

    // Write metadata atoms (all neccesary changes have been made already)
    for (i, atom) in metadata.atoms_iter().enumerate() {
        mp4_writer.write_atom(atom.clone()).await.with_context(|| {
            format!("Failed to write atom {} ({})", i + 1, atom.header.atom_type)
        })?;
        progress_bar.set_progress(mp4_writer.current_offset());
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
        "incorrect mdat_content_offset!"
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

            progress_bar.set_progress(mp4_writer.current_offset());
        }

        chunk_idx += 1;
    }

    mp4_writer.flush().await.context("final flush")?;

    progress_bar.finalize();

    assert_eq!(
        mp4_writer.current_offset(),
        mdat_content_offset + mdat_size as usize,
        "mdat header has incorrect size"
    );

    Ok(())
}
