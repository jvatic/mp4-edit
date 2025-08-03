use anyhow::{anyhow, Context};
use futures_util::io::{BufReader, BufWriter};
use progress_bar::pb::ProgressBar;
use std::env;
use tokio::{
    fs,
    io::{self, AsyncRead},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use mp4_parser::{
    atom::{hdlr::HandlerType, stco_co64::ChunkOffsets, FourCC},
    chunk_offset_builder::ChunkOffsetBuilder,
    writer::SerializeAtom,
    Mp4Writer, Parser,
};

async fn open_input(input_name: &str) -> anyhow::Result<Box<dyn AsyncRead + Unpin + Send>> {
    if input_name == "-" {
        Ok(Box::new(io::stdin()))
    } else {
        let file = fs::File::open(input_name).await?;
        Ok(Box::new(file))
    }
}

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

    // Open input file and create parser
    let input_file = open_input(input_name).await?;
    let input_reader = BufReader::new(input_file.compat());
    let parser = Parser::new(input_reader);

    // Parse metadata atoms from input
    let metadata = parser
        .parse_metadata()
        .await
        .context("Failed to parse metadata from input file")?;

    // Filter out non-audio tracks
    let metadata = metadata.tracks_retain(|trak| {
        trak.media()
            .and_then(|mdia| mdia.handler_reference())
            .map(|hdlr| matches!(hdlr.handler_type, HandlerType::Audio))
            .unwrap_or_default()
    });

    let mut input_metadata = metadata;
    let mut metadata = input_metadata.clone();

    let (num_samples, mdat_size) = metadata.tracks_iter().fold((0, 0), |(n, size), trak| {
        trak.media()
            .and_then(|m| m.media_information())
            .and_then(|m| m.sample_table())
            .and_then(|st| st.sample_size())
            .map(|s| (n + s.sample_count, size + s.entry_sizes.iter().sum::<u32>()))
            .unwrap_or((0, 0))
    });

    let mut progress_bar = ProgressBar::new_with_eta(num_samples as usize);

    // serialize metadata to find the new size (should be fairly cheap)
    let new_metadata_size = metadata
        .atoms_iter()
        .cloned()
        .flat_map(SerializeAtom::into_bytes)
        .collect::<Vec<_>>()
        .len();

    let mdat_content_offset = new_metadata_size + 8;

    // Update chunk offsets to reflect new metadata size
    let mut chunk_offsets = metadata
        .tracks_iter()
        .fold(ChunkOffsetBuilder::new(), |mut builder, trak| {
            let stbl = trak
                .media()
                .and_then(|mdia| mdia.media_information())
                .and_then(|minf| minf.sample_table())
                .unwrap();
            let stsz = stbl.sample_size().unwrap();
            let stsc = stbl.sample_to_chunk().unwrap();
            builder.add_track(stsc, stsz);
            builder
        })
        .build_chunk_offsets(mdat_content_offset as u64);
    metadata
        .tracks_iter_mut()
        .enumerate()
        .for_each(|(track_idx, trak)| {
            let mut stbl = trak
                .media()
                .and_then(|mdia| mdia.media_information())
                .and_then(|minf| minf.sample_table())
                .unwrap();
            let stco = stbl.chunk_offset_mut().unwrap();
            let chunk_offsets = std::mem::take(&mut chunk_offsets[track_idx]);
            stco.chunk_offsets = ChunkOffsets::from(chunk_offsets);
        });

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
    }

    mp4_writer.flush().await.context("metadata flush")?;

    // Write MDAT header (it will have a size=0 which we'll update later)
    mp4_writer
        .write_atom_header(FourCC::from(*b"mdat"), mdat_size as usize - 8)
        .await
        .context("error writing mdat placeholder header")?;

    if input_metadata.mdat_header().is_none() {
        return Err(anyhow!("mdat atom not found"));
    }

    // Copy and write sample data
    let mut chunk_idx = 0;
    let mut sample_idx = 0;
    let mut chunk_parser = input_metadata.chunks()?;
    while let Some(chunk) = chunk_parser.read_next_chunk().await? {
        for (i, sample) in chunk.samples().enumerate() {
            let data = sample.data.to_vec();

            mp4_writer.write_raw(&data).await.context(format!(
                "error writing sample {i:02} data in chunk {chunk_idx:02}"
            ))?;

            sample_idx += 1;
            progress_bar.set_progress(sample_idx);
        }

        chunk_idx += 1;
    }

    mp4_writer.flush().await.context("final flush")?;

    progress_bar.finalize();

    Ok(())
}
