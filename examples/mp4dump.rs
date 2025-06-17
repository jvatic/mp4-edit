use anyhow::Context;
use std::env;
use tokio::{
    fs,
    io::{self, AsyncRead},
};
use tokio_util::compat::TokioAsyncReadCompatExt;

use mp4_parser::atom::{
    ChunkOffsetAtom, MovieHeaderAtom, SampleSizeAtom, SampleToChunkAtom, TimeToSampleAtom,
};
use mp4_parser::{Atom, AtomData, Parser};

/// Format file size in human-readable format
fn format_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = size as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", size as u64, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Get a summary of atom data
fn get_atom_summary(data: &Option<AtomData>) -> String {
    match data {
        Some(atom) => format!("{atom:?}"),
        None => "".to_string(),
    }
}

/// Print atom with proper formatting and indentation
fn print_atom(atom: &Atom, indent: usize) {
    let indent_str = if indent > 0 {
        format!("{:<width$}", "", width = indent)
    } else {
        String::new()
    };

    let atom_display = format!("{}{}", indent_str, atom.atom_type);
    let size_display = format_size(atom.size);
    let offset_display = format!(
        "0x{:08x}..=0x{:08x}",
        atom.offset,
        atom.offset + atom.size - 1
    );
    let summary = get_atom_summary(&atom.data);

    // Color coding based on atom type
    let atom_color = match atom.atom_type.to_string().as_str() {
        "ftyp" | "styp" => "\x1b[1;32m",          // Green for file type
        "moov" | "trak" | "mdia" => "\x1b[1;34m", // Blue for containers
        "mvhd" | "tkhd" | "mdhd" => "\x1b[1;35m", // Magenta for headers
        "stbl" | "stts" | "stsc" | "stsz" | "stco" | "co64" => "\x1b[1;31m", // Red for sample tables
        _ => "\x1b[0m",                                                      // Default
    };

    println!("\x1b[1;36m│\x1b[0m {}{:<20}\x1b[0m │ \x1b[2m{:<23}\x1b[0m │ \x1b[1m{:<10}\x1b[0m │ \x1b[2m{:<30}\x1b[0m",
        atom_color, atom_display, offset_display, size_display, summary);
}

/// Print table header
fn print_table_header() {
    println!("\n\x1b[1;36m╭─ MP4 Atom Structure ──────────────────────────────────────────────────────────────\x1b[0m");
    println!("\x1b[1;36m│\x1b[0m");
    println!(
        "\x1b[1;36m│\x1b[0m \x1b[1;33m{:<20} │ {:<23} │ {:<10} │ {:<30}\x1b[0m",
        "Atom Type", "Offset Range", "Size", "Summary"
    );
    println!(
        "\x1b[1;36m│\x1b[0m \x1b[2m{:─<20}─┼─{:─<23}─┼─{:─<10}─┼─{:─<30}\x1b[0m",
        "", "", "", ""
    );
}

/// Print table footer
fn print_table_footer() {
    println!("\x1b[1;36m│\x1b[0m");
    println!("\x1b[1;36m╰────────────────────────────────────────────────────────────────────────────────────\x1b[0m\n");
}

/// Process atom recursively, handling data extraction and printing
fn process_atom(
    atom: &Atom,
    indent: usize,
    atom_count: &mut usize,
    movie_header: &mut Option<MovieHeaderAtom>,
    track_id: &mut Option<u32>,
    sample_sizes: &mut Option<SampleSizeAtom>,
    sample_durations: &mut Option<TimeToSampleAtom>,
    chunk_offsets: &mut Option<ChunkOffsetAtom>,
    sample_to_chunk: &mut Option<SampleToChunkAtom>,
) {
    print_atom(atom, indent);
    *atom_count += 1;

    // Extract data from atom
    if let Some(data) = &atom.data {
        if let AtomData::TrackHeader(h) = data {
            *track_id = Some(h.track_id);
        }

        // take sample metadata for track 1
        if matches!(track_id, Some(1)) {
            use AtomData::*;
            match data {
                SampleSize(data) => *sample_sizes = Some(data.clone()),
                TimeToSample(data) => *sample_durations = Some(data.clone()),
                ChunkOffset(data) => *chunk_offsets = Some(data.clone()),
                SampleToChunk(data) => *sample_to_chunk = Some(data.clone()),
                _ => {}
            }
        } else if movie_header.is_none() {
            use AtomData::*;
            if let MovieHeader(data) = data {
                *movie_header = Some(data.clone());
            }
        }
    }

    // Process children recursively
    for child in &atom.children {
        process_atom(
            child,
            indent + 1,
            atom_count,
            movie_header,
            track_id,
            sample_sizes,
            sample_durations,
            chunk_offsets,
            sample_to_chunk,
        );
    }
}

/// Print atoms directly from hierarchical stream
async fn print_atoms_from_stream<R: futures_io::AsyncRead + Unpin + Send>(
    parser: Parser<R>,
) -> anyhow::Result<usize> {
    let mut atom_count = 0;
    let mut first_atom = true;

    let mut movie_header = None;
    let mut track_id = None;
    let mut sample_sizes = None;
    let mut sample_durations = None;
    let mut chunk_offsets = None;
    let mut sample_to_chunk = None;

    let metadata = parser.parse_metadata().await?.1;
    for atom in metadata.atoms_iter() {
        if first_atom {
            print_table_header();
            first_atom = false;
        }

        process_atom(
            atom,
            0,
            &mut atom_count,
            &mut movie_header,
            &mut track_id,
            &mut sample_sizes,
            &mut sample_durations,
            &mut chunk_offsets,
            &mut sample_to_chunk,
        );
    }

    // if movie_header.is_some()
    //     && sample_sizes.is_some()
    //     && sample_durations.is_some()
    //     && chunk_offsets.is_some()
    //     && sample_to_chunk.is_some()
    // {
    //     let movie_header = movie_header.unwrap();
    //     let stream = parser.stream_samples(
    //         sample_sizes.unwrap(),
    //         sample_durations.unwrap(),
    //         chunk_offsets.unwrap(),
    //         sample_to_chunk.unwrap(),
    //     );
    //     pin_mut!(stream);

    //     let timescale = movie_header.timescale;

    //     let mut i = -1;
    //     while let Some(sample) = stream
    //         .next()
    //         .await
    //         .transpose()
    //         .context("Failed to parse chunk stream item")?
    //     {
    //         i += 1;
    //         if i > 5 {
    //             break;
    //         }
    //         println!(
    //             "Sample: {:?} duration={}",
    //             sample.data.len(),
    //             (sample.duration as f64) / (timescale as f64),
    //         );
    //     }
    // }

    if !first_atom {
        print_table_footer();
    }

    Ok(atom_count)
}

async fn open_input(input_name: &str) -> anyhow::Result<Box<dyn AsyncRead + Unpin + Send>> {
    if input_name == "-" {
        Ok(Box::new(io::stdin()))
    } else {
        let file = fs::File::open(input_name).await?;
        Ok(Box::new(file))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mp4_filename>", args[0]);
        std::process::exit(1);
    }
    let file = open_input(args[1].as_str()).await?;

    println!("\x1b[1;32m🎬 Analyzing MP4 file: {}\x1b[0m", &args[1]);

    let parser = Parser::new(file.compat());
    let atom_count = print_atoms_from_stream(parser)
        .await
        .context("Failed to parse MP4 file")?;

    // Print summary statistics
    println!("\x1b[1;33m📊 Summary:\x1b[0m");
    println!("   Total atoms: \x1b[1m{}\x1b[0m", atom_count);
    if args[1].as_str() != "-" {
        let file_size = fs::metadata(&args[1]).await?.len();
        println!("   File size: \x1b[1m{}\x1b[0m", format_size(file_size));
    }

    Ok(())
}
