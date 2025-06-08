use anyhow::Context;
use std::{env, fs::File, ops::Deref};

use mp4_parser::{parse_mp4, Atom};

/// Recursively print [Atom]s with indentation to show nesting along with offset and size details.
fn print_atoms(atoms: &[Atom], indent: usize) {
    if indent == 0 {
        println!(
            "{:<21}| {:<12} | {:<11} | {:<11}",
            "Atom", "Offset", "Size", "Data"
        );
        println!("{:-<width$}", "", width = 49);
    }
    for b in atoms {
        // Convert the atom type to a printable string, handling non-UTF8 bytes as necessary.
        let atom_type_str = String::from_utf8_lossy(b.atom_type.deref());
        let atom_name = format!("{:>width$}{}", "", atom_type_str, width = indent);
        println!(
            "{} {:<width$}| {:<12} | {:<11} | {:?}",
            "",
            atom_name,
            b.offset,
            b.size,
            b.data,
            width = 20,
        );
        if !b.children.is_empty() {
            print_atoms(&b.children, indent + 2);
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mp4_filename>", args[0]);
        std::process::exit(1);
    }
    let input = File::open(&args[1])?;

    let atoms = parse_mp4(input).context("Failed to parse MP4 file")?;

    println!("MP4 Atom structure:");
    print_atoms(&atoms, 0);

    Ok(())
}
