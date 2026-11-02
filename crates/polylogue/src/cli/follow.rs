//! `follow`: prints one line per live-state change and per new message or
//! tool call until interrupted. Each daemon request runs until its
//! deadline and ends with marks, which the next request carries, so lines
//! continue without repeats across requests.
use super::{Archive, Request, fail};
use crate::protocol::follow::{Line, Marks};
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Duration;

/// How long one follow request runs before the client renews it.
const REQUEST_SPAN: Duration = Duration::from_secs(60);

pub fn run(
    _: &crate::commands::Values,
    request: &Request,
    startup: &crate::config::Startup,
    _: &command_surface::RunnableRequest,
    _: crate::commands::CallForm,
) -> ExitCode {
    let Request::Follow { query, marks } = request else {
        return fail(startup.output, true, "bad_request", "a follow request is required", 2);
    };
    let zone = startup
        .config
        .standalone_day()
        .map(|day| day.zone().clone())
        .unwrap_or_else(|_| jiff::tz::TimeZone::system());
    let archive = Archive::at(&startup.config.archive);
    let mut marks = marks.clone();
    let mut stdout = io::stdout().lock();
    loop {
        let request = Request::Follow {
            query: query.clone(),
            marks: marks.clone(),
        };
        let mut next: Option<Marks> = None;
        let result = crate::client::stream::run(&archive, &request, Some(REQUEST_SPAN), |frame| {
            if frame.event == "error" {
                return Err(io::Error::other(frame.data));
            }
            let line: Line = serde_json::from_str(&frame.data).map_err(io::Error::other)?;
            match line {
                Line::Marks { marks } => next = Some(marks),
                line => {
                    if let Some(text) = line.text(&zone) {
                        writeln!(stdout, "{}", polylogue_render::terminal::escape(&text))?;
                        stdout.flush()?;
                    }
                }
            }
            Ok(true)
        });
        match (result, next) {
            (Ok(()), Some(reached)) => marks = Some(reached),
            (Ok(()), None) => {
                return fail(
                    startup.output,
                    true,
                    "io",
                    "the daemon ended the follow without its marks",
                    1,
                );
            }
            (Err(error), _) => {
                return fail(startup.output, true, "io", &error.to_string(), 1);
            }
        }
    }
}
