use std::env;
use tokio::{
    fs,
    io::{self},
};
use tokio_util::compat::TokioAsyncReadCompatExt;

use mp4_parser::{
    atom::{meta, AtomHeader},
    parser::Metadata,
    Atom, AtomData, Parser,
};

/// Format file size in human-readable format
fn format_size(size: usize) -> String {
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
fn get_atom_summary(atom: &Atom) -> String {
    match &atom.data {
        Some(AtomData::RawData(data)) if atom.header.atom_type == b"meta" => {
            meta::MetaHeader::from_bytes(data.as_slice())
                .map(|meta| format!("{meta:?}"))
                .unwrap_or_else(|_| format!("{data:?}"))
        }
        Some(data) => format!("{data:?}"),
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

    let atom_display = format!("{}{}", indent_str, atom.header.atom_type);
    let size_display = format_size(atom.header.atom_size());
    let offset_display = format!(
        "0x{:08x}..=0x{:08x}",
        atom.header.offset,
        atom.header.offset + atom.header.atom_size() - 1
    );
    let summary = get_atom_summary(atom);

    // Color coding based on atom type
    let atom_color = match atom.header.atom_type.to_string().as_str() {
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
fn process_atom(atom: &Atom, indent: usize, atom_count: &mut usize) {
    print_atom(atom, indent);
    *atom_count += 1;

    // Process children recursively
    for child in &atom.children {
        process_atom(child, indent + 1, atom_count);
    }
}

async fn print_atoms(metadata: Metadata, mdat_header: Option<AtomHeader>) -> anyhow::Result<usize> {
    let mut atom_count = 0;
    let mut first_atom = true;

    let mut track_bitrate = Vec::with_capacity(1);
    for trak in metadata.tracks_iter() {
        let num_bits = trak
            .media()
            .and_then(|m| m.media_information())
            .and_then(|m| m.sample_table())
            .and_then(|st| st.sample_size())
            .map(|s| s.entry_sizes.iter().sum::<u32>())
            .unwrap_or_default()
            * 8;

        let duration_secds = trak
            .media()
            .and_then(|m| m.header())
            .map(|mdhd| (mdhd.duration as f64) / (mdhd.timescale as f64))
            .unwrap_or_default();

        let bitrate = (num_bits as f64) / duration_secds;
        println!(
            "trak({track_id}) bitrate: {bitrate}",
            track_id = trak.header().map(|tkhd| tkhd.track_id).unwrap_or_default()
        );
        track_bitrate.push(bitrate.round() as u32);
    }

    for atom in metadata.atoms_iter() {
        if first_atom {
            print_table_header();
            first_atom = false;
        }

        process_atom(atom, 0, &mut atom_count);
    }

    if let Some(mdat_header) = mdat_header {
        let mdat_atom = Atom {
            header: mdat_header,
            data: None,
            children: Vec::new(),
        };
        process_atom(&mdat_atom, 0, &mut atom_count);
    }

    if !first_atom {
        print_table_footer();
    }

    Ok(atom_count)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mp4_filename>", args[0]);
        std::process::exit(1);
    }

    let input_name = args[1].as_str();
    println!("\x1b[1;32m🎬 Analyzing MP4 file: {}\x1b[0m", input_name);

    let atom_count = if input_name == "-" {
        eprintln!("parsing as readonly");
        let input = Box::new(io::stdin());
        let parser = Parser::new(input.compat());
        let metadata = parser.parse_metadata().await?;
        let mdat_header = metadata.mdat_header().cloned();
        print_atoms(metadata.into_metadata(), mdat_header).await?
    } else {
        eprintln!("parsing as seekable");
        let file = fs::File::open(input_name).await?;
        let parser = Parser::new(file.compat());
        let metadata = parser.parse_metadata_seek().await?;
        let mdat_header = metadata.mdat_header().cloned();
        print_atoms(metadata.into_metadata(), mdat_header).await?
    };

    // Print summary statistics
    println!("\x1b[1;33m📊 Summary:\x1b[0m");
    println!("   Total atoms: \x1b[1m{}\x1b[0m", atom_count);
    if args[1].as_str() != "-" {
        let file_size = fs::metadata(&args[1]).await?.len();
        println!(
            "   File size: \x1b[1m{}\x1b[0m",
            format_size(file_size as usize)
        );
    }

    Ok(())
}
