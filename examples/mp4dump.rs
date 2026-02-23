/*!
 * This example demonstrates inspecting the atom structure of an mp4 file.
 */

use anyhow::Result;
use colored::{ColoredString, Colorize};
use std::env;
use terminal_size::terminal_size;
use tokio::fs;
use tokio_util::compat::TokioAsyncReadCompatExt;

use mp4_edit::{Atom, Parser};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mp4_filename>", args[0]);
        std::process::exit(1);
    }

    // Open <mp4_filename> and parse it's metadata
    let input_path = args[1].as_str();
    let file = fs::File::open(input_path).await?;
    let parser = Parser::new_seekable(file.compat());
    let metadata = parser.parse_metadata().await?;
    let mdat_header = metadata.mdat_header().cloned();

    // Get the terminal width
    let width = match terminal_size() {
        Some((terminal_size::Width(width), _)) => width as usize - 2,
        _ => 100,
    };

    // Print mp4 metadata
    let table = Mp4Table::new(width);
    table.print_header();
    table.print_atoms(metadata.atoms_iter(), 0);
    if let Some(mdat_header) = mdat_header {
        let atom = Atom::builder().header(mdat_header).build();
        table.print_atoms(std::iter::once(&atom), 0);
    };
    table.print_footer();

    Ok(())
}

struct Mp4Table {
    width: usize,
}

impl Mp4Table {
    pub fn new(width: usize) -> Self {
        Self { width }
    }

    pub fn print_header(&self) {
        println!(
            "{}",
            format!("{:─<width$}", "╭─ MP4 Atom Structure ", width = self.width).dimmed()
        );
        println!("{}", "│".dimmed());
        self.print_row("Atom Type", "Offset Range", "Size", "Data", 0);
        self.print_divider_row();
    }

    /// Recursively print each [Atom] and it's children.
    pub fn print_atoms<'a>(&self, atoms: impl Iterator<Item = &'a Atom>, indent: usize) {
        atoms.for_each(|atom| {
            self.print_atom(atom, indent);
            self.print_atoms(atom.children.iter(), indent + 1);
        });
    }

    pub fn print_footer(&self) {
        println!("{}", "│".dimmed());
        println!(
            "{}",
            format!("{:—<width$}", "╰─", width = self.width).dimmed()
        );
    }

    fn print_atom(&self, atom: &Atom, indent: usize) {
        let offset_range = format!(
            "0x{:08x}..=0x{:08x}",
            atom.header.offset,
            atom.header.offset + atom.header.atom_size() - 1
        );
        let size = format_size(atom.header.atom_size());
        let data = match &atom.data {
            Some(data) => format!("{data:?}"),
            None => "".to_string(),
        };
        self.print_row(
            self.atom_type_colored(atom),
            offset_range,
            size,
            data.dimmed(),
            indent,
        );
    }

    fn atom_type_colored(&self, atom: &Atom) -> ColoredString {
        let atom_type = atom.header.atom_type.as_str();
        if !atom.children.is_empty() {
            return atom_type.cyan();
        }
        if atom_type.ends_with("hd") {
            return atom_type.blue();
        }
        match atom_type {
            "ftyp" | "free" | "wide" | "ilst" | "dref" => atom_type.yellow(),
            _ => atom_type.purple(),
        }
    }

    /// Calculate column widths based on terminal size.
    fn col_widths(&self) -> [usize; 4] {
        let atom_type_width = (self.width / 5).max(20).min(30);
        let offset_range_width = 23;
        let size_width = (self.width / 15).max(10).min(15);
        let data_width = self
            .width
            .saturating_sub(atom_type_width + offset_range_width + size_width)
            .max(30);

        let n_pipes = 4;
        let padding = (n_pipes * 2) - 1;
        let data_width = data_width - n_pipes - padding;

        [atom_type_width, offset_range_width, size_width, data_width]
    }

    fn print_row(
        &self,
        atom_type: impl Into<ColoredString>,
        offset_range: impl Into<ColoredString>,
        size: impl Into<ColoredString>,
        data: impl Into<ColoredString>,
        indent: usize,
    ) {
        let [atom_type_width, offset_range_width, size_width, data_width] = self.col_widths();

        let mut atom_type = atom_type.into();
        let mut offset_range = offset_range.into();
        let mut size = size.into();
        let mut data = data.into();

        atom_type.input = match indent {
            0 => format!("{}", atom_type.input),
            indent => format!("{:indent$}{}", "", atom_type.input),
        };
        atom_type.input = format!("{: <atom_type_width$}", atom_type.input);
        offset_range.input = format!("{: <offset_range_width$}", offset_range.input);
        size.input = format!("{: <size_width$}", size.input);
        data.input = format!("{: <data_width$}", data.input);

        let pipe = "│".dimmed();
        println!("{pipe} {atom_type} {pipe} {offset_range} {pipe} {size} {pipe} {data}");
    }

    fn print_divider_row(&self) {
        let cols = self
            .col_widths()
            .into_iter()
            .map(|col_width| format!("{:—<col_width$}", ""))
            .collect::<Vec<_>>()
            .join(" │ ");

        println!("{}", format!("│ {cols}").dimmed());
    }
}

fn format_size(size: usize) -> String {
    const DIVISOR: f64 = 1024.0;
    let mut size = size as f64;
    let (size, unit) = std::iter::once(size)
        .chain(std::iter::from_fn(|| {
            if size < DIVISOR {
                return None;
            }
            size /= DIVISOR;
            Some(size)
        }))
        .zip(["B", "KB", "MB", "GB"].into_iter())
        .last()
        .unwrap();

    format!("{:.1} {}", size, unit)
}
