use super::*;
use crossterm::{
    cursor, event, queue,
    terminal::{self, Clear, ClearType},
};
use std::io::{self, IsTerminal, Write};

/// Drawing is restricted to the lines allocated below the shell prompt.
pub(super) struct Screen {
    output: Box<dyn Write>,
    terminal: bool,
    lines: u16,
    active: bool,
}
impl Screen {
    pub fn terminal() -> Self {
        let stdout = io::stdout().is_terminal();
        let terminal = io::stdin().is_terminal() && (stdout || io::stderr().is_terminal());
        Self {
            output: if stdout {
                Box::new(io::stdout())
            } else {
                Box::new(io::stderr())
            },
            terminal,
            lines: 0,
            active: false,
        }
    }
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }
    pub fn size(&self) -> Result<(u16, u16)> {
        terminal::size().map_err(Failure::io)
    }
    pub fn enter(&mut self) -> Result<()> {
        terminal::enable_raw_mode().map_err(Failure::io)?;
        self.active = true;
        queue!(self.output, cursor::Hide, event::EnableBracketedPaste).map_err(Failure::io)?;
        self.output.flush().map_err(Failure::io)
    }
    fn clear(&mut self) -> io::Result<()> {
        if self.lines > 0 {
            queue!(
                self.output,
                cursor::MoveUp(self.lines),
                cursor::MoveToColumn(0)
            )?;
            for _ in 0..self.lines {
                queue!(self.output, Clear(ClearType::CurrentLine))?;
                write!(self.output, "\r\n")?;
            }
            queue!(
                self.output,
                cursor::MoveUp(self.lines),
                cursor::MoveToColumn(0)
            )?;
            self.lines = 0;
        }
        Ok(())
    }
    pub fn draw(&mut self, lines: &[String]) -> Result<()> {
        let (width, height) = self.size()?;
        // A resize can move the old viewport into scrollback. Start a new
        // viewport rather than moving above the current terminal height.
        self.lines = self.lines.min(height.saturating_sub(1));
        self.clear().map_err(Failure::io)?;
        let start = lines
            .len()
            .saturating_sub(height.saturating_sub(1) as usize);
        for line in &lines[start..] {
            queue!(
                self.output,
                cursor::MoveToColumn(0),
                Clear(ClearType::CurrentLine)
            )
            .map_err(Failure::io)?;
            write!(
                self.output,
                "{}\r\n",
                clipped(line, width.saturating_sub(1) as usize)
            )
            .map_err(Failure::io)?;
            self.lines += 1;
        }
        self.output.flush().map_err(Failure::io)
    }
    /// Returns the terminal to cooked input before a pager or action runs.
    pub fn leave(&mut self) -> Result<()> {
        if self.active {
            let cleared = self.clear();
            let controls =
                crossterm::execute!(self.output, event::DisableBracketedPaste, cursor::Show);
            let restored = terminal::disable_raw_mode();
            self.active = false;
            cleared.and(controls).and(restored).map_err(Failure::io)?;
        }
        Ok(())
    }
    pub fn print(&mut self, references: &[String]) -> Result<()> {
        self.clear().map_err(Failure::io)?;
        for reference in references {
            if self.active {
                write!(self.output, "{}\r\n", safe_line(reference)).map_err(Failure::io)?;
            } else {
                writeln!(self.output, "{}", safe_line(reference)).map_err(Failure::io)?;
            }
        }
        self.output.flush().map_err(Failure::io)
    }
}
impl Drop for Screen {
    fn drop(&mut self) {
        let _ = self.leave();
    }
}

pub(super) fn safe_line(text: &str) -> String {
    polylogue_render::terminal::escape(text)
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

// UTF-8 bytes bound the number of occupied cells, including wide characters.
// Combining characters may leave unused cells but cannot wrap the viewport.
pub(super) fn clipped(text: &str, width: usize) -> String {
    let mut end = text.len().min(width);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}
