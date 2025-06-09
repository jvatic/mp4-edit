use anyhow::Context;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use mp4_parser::{parse_mp4, Atom, AtomData};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Padding, Paragraph, Wrap},
    Frame, Terminal,
};
use std::{env, io};
use tokio::fs;
use tokio_util::compat::TokioAsyncReadCompatExt;

#[derive(Debug, Clone)]
struct TreeNode {
    atom: Atom,
    children: Vec<TreeNode>,
    expanded: bool,
    depth: usize,
}

impl TreeNode {
    fn new(atom: Atom, depth: usize) -> Self {
        let children = atom
            .children
            .iter()
            .map(|child| TreeNode::new(child.clone(), depth + 1))
            .collect();
        
        Self {
            atom,
            children,
            expanded: false,
            depth,
        }
    }

    fn flatten(&self) -> Vec<&TreeNode> {
        let mut result = vec![self];
        if self.expanded {
            for child in &self.children {
                result.extend(child.flatten());
            }
        }
        result
    }

    fn toggle_expanded(&mut self) {
        if !self.children.is_empty() {
            self.expanded = !self.expanded;
        }
    }

    fn find_mut(&mut self, index: usize, current_index: &mut usize) -> Option<&mut TreeNode> {
        if *current_index == index {
            return Some(self);
        }
        *current_index += 1;
        
        if self.expanded {
            for child in &mut self.children {
                if let Some(node) = child.find_mut(index, current_index) {
                    return Some(node);
                }
            }
        }
        None
    }
}

struct App {
    tree_nodes: Vec<TreeNode>,
    list_state: ListState,
    show_details: bool,
}

impl App {
    fn new(atoms: Vec<Atom>) -> Self {
        let tree_nodes = atoms
            .into_iter()
            .map(|atom| TreeNode::new(atom, 0))
            .collect();
        
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        
        Self {
            tree_nodes,
            list_state,
            show_details: false,
        }
    }

    fn flatten_nodes(&self) -> Vec<&TreeNode> {
        self.tree_nodes
            .iter()
            .flat_map(|node| node.flatten())
            .collect()
    }

    fn next(&mut self) {
        let flattened = self.flatten_nodes();
        let i = match self.list_state.selected() {
            Some(i) => {
                if i >= flattened.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.list_state.select(Some(i));
    }

    fn previous(&mut self) {
        let flattened = self.flatten_nodes();
        let i = match self.list_state.selected() {
            Some(i) => {
                if i == 0 {
                    flattened.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.list_state.select(Some(i));
    }

    fn toggle_current(&mut self) {
        if let Some(selected) = self.list_state.selected() {
            let mut current_index = 0;
            for tree_node in &mut self.tree_nodes {
                if let Some(node) = tree_node.find_mut(selected, &mut current_index) {
                    node.toggle_expanded();
                    break;
                }
            }
        }
    }

    fn get_selected_atom(&self) -> Option<&Atom> {
        if let Some(selected) = self.list_state.selected() {
            let flattened = self.flatten_nodes();
            flattened.get(selected).map(|node| &node.atom)
        } else {
            None
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = if app.show_details {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(f.area())
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(100)])
            .split(f.area())
    };

    // Create list items
    let flattened = app.flatten_nodes();
    let items: Vec<ListItem> = flattened
        .iter()
        .map(|node| {
            let indent = "  ".repeat(node.depth);
            let expansion_indicator = if node.children.is_empty() {
                "  "
            } else if node.expanded {
                "▼ "
            } else {
                "▶ "
            };
            
            let atom_type = format!("{}", node.atom.atom_type);
            let size = format_size(node.atom.size);
            let offset = format!("0x{:08x}", node.atom.offset);
            
            let content = format!(
                "{}{}{:<8} │ {:<10} │ {}",
                indent, expansion_indicator, atom_type, size, offset
            );
            
            ListItem::new(content)
        })
        .collect();

    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("MP4 Atom Structure")
                .padding(Padding::uniform(1)),
        )
        .highlight_style(
            Style::default()
                .bg(Color::LightGreen)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("➤ ");

    f.render_stateful_widget(list, chunks[0], &mut app.list_state);

    // Render details panel if enabled
    if app.show_details {
        if let Some(atom) = app.get_selected_atom() {
            let details = create_details_text(atom);
            let paragraph = Paragraph::new(details)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Atom Details")
                        .padding(Padding::uniform(1)),
                )
                .wrap(Wrap { trim: true });
            
            f.render_widget(paragraph, chunks[1]);
        }
    }

    // Help text at the bottom
    let help_text = if app.show_details {
        "↑/↓: Navigate │ Space: Expand/Collapse │ Tab: Hide Details │ Q: Quit"
    } else {
        "↑/↓: Navigate │ Space: Expand/Collapse │ Tab: Show Details │ Q: Quit"
    };
    
    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::Gray))
        .block(Block::default().borders(Borders::TOP));
    
    let help_area = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(f.area())[1];
    
    f.render_widget(help, help_area);
}

fn create_details_text(atom: &Atom) -> Text {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Type: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("{}", atom.atom_type)),
        ]),
        Line::from(vec![
            Span::styled("Size: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format_size(atom.size)),
        ]),
        Line::from(vec![
            Span::styled("Offset: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(format!("0x{:08x} ({})", atom.offset, atom.offset)),
        ]),
        Line::from(vec![
            Span::styled("Children: ", Style::default().add_modifier(Modifier::BOLD)),
            Span::raw(atom.children.len().to_string()),
        ]),
    ];

    if let Some(data) = &atom.data {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Data:", Style::default().add_modifier(Modifier::BOLD)),
        ]));
        
        match data {
            AtomData::FileType(ftyp) => {
                lines.push(Line::from(format!("  Major Brand: {}", ftyp.major_brand)));
                lines.push(Line::from(format!("  Minor Version: {}", ftyp.minor_version)));
                lines.push(Line::from(format!("  Compatible Brands: {:?}", ftyp.compatible_brands)));
            }
            AtomData::MovieHeader(mvhd) => {
                lines.push(Line::from(format!("  Version: {}", mvhd.version)));
                lines.push(Line::from(format!("  Creation Time: {}", mvhd.creation_time)));
                lines.push(Line::from(format!("  Modification Time: {}", mvhd.modification_time)));
                lines.push(Line::from(format!("  Time Scale: {}", mvhd.timescale)));
                lines.push(Line::from(format!("  Duration: {}", mvhd.duration)));
                lines.push(Line::from(format!("  Rate: {}", mvhd.rate)));
                lines.push(Line::from(format!("  Volume: {}", mvhd.volume)));
            }
            AtomData::TrackHeader(tkhd) => {
                lines.push(Line::from(format!("  Version: {}", tkhd.version)));
                lines.push(Line::from(format!("  Flags: 0x{:02x}{:02x}{:02x}", tkhd.flags[0], tkhd.flags[1], tkhd.flags[2])));
                lines.push(Line::from(format!("  Track ID: {}", tkhd.track_id)));
                lines.push(Line::from(format!("  Duration: {}", tkhd.duration)));
                lines.push(Line::from(format!("  Width: {}", tkhd.width)));
                lines.push(Line::from(format!("  Height: {}", tkhd.height)));
            }
            AtomData::HandlerReference(hdlr) => {
                lines.push(Line::from(format!("  Handler Type: {}", hdlr.handler_type.as_str())));
                lines.push(Line::from(format!("  Name: {}", hdlr.name)));
            }
            _ => {
                lines.push(Line::from(format!("  {:?}", data)));
            }
        }
    }

    Text::from(lines)
}

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

fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => return Ok(()),
                    KeyCode::Down => app.next(),
                    KeyCode::Up => app.previous(),
                    KeyCode::Char(' ') => app.toggle_current(),
                    KeyCode::Tab => app.show_details = !app.show_details,
                    _ => {}
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <mp4_filename>", args[0]);
        std::process::exit(1);
    }

    let file = fs::File::open(&args[1]).await?;
    let atoms = parse_mp4(file.compat()).await.context("Failed to parse MP4 file")?;

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run it
    let app = App::new(atoms);
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}