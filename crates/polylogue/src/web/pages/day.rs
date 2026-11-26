use super::*;

/// The queries `/day/DATE` shows, in its order: the day's sessions, their
/// usage during the day, the month's messages by logical day, and the same
/// day in other years. Each section prints its own line.
pub fn day_queries(date: jiff::civil::Date) -> [String; 4] {
    let day = format!("since:{date} until:{date}");
    [
        format!("sessions where {day}"),
        format!("sessions where {day} | measure usage"),
        format!(
            "messages where since:{} until:{} active:* | group by day(at) | count",
            date.first_of_month(),
            date.last_of_month()
        ),
        format!(
            "sessions where month_day:{:02}-{:02}",
            date.month(),
            date.day()
        ),
    ]
}

/// One logical day: its sessions and usage, its month's calendar, and the
/// same day in other years. `replies` are the replies to [`day_queries`].
pub fn day(date: jiff::civil::Date, replies: &[Value; 4]) -> String {
    let queries = day_queries(date);
    let [sessions, usage, calendar, other_years] = replies;
    let line = |query: &str| {
        format!(
            "<p class=\"meta\"><code>polylogue {}</code></p>",
            esc(&shell_quote(query))
        )
    };
    let link = |day: Option<jiff::civil::Date>, label: &str| {
        day.map(|day| format!("<a href=\"/day/{day}\">{label}</a>"))
            .unwrap_or_default()
    };
    let mut body = format!(
        "<h1>{date}</h1><nav class=\"paging\" aria-label=\"Days\">{} {}</nav>",
        link(date.yesterday().ok(), "Previous day"),
        link(date.tomorrow().ok(), "Next day"),
    );
    let _ = write!(
        body,
        "<h2>Sessions</h2>{}{}",
        results_table(sessions),
        line(&queries[0])
    );
    let _ = write!(
        body,
        "<h2>Usage during the day</h2><pre>{}</pre>{}",
        esc(&polylogue_render::measures::usage(&usage["data"])),
        line(&queries[1])
    );
    let _ = write!(
        body,
        "<h2>Messages by day in {}</h2>{}{}",
        date.strftime("%Y-%m"),
        month(date, calendar),
        line(&queries[2])
    );
    let _ = write!(
        body,
        "<h2>This day in other years</h2>{}{}",
        results_table(other_years),
        line(&queries[3])
    );
    layout(
        &date.to_string(),
        &queries[0],
        &format!("polylogue {}", shell_quote(&queries[0])),
        &body,
    )
}

/// Every day of the month with its count, each linking to its day; a day
/// with no row counts none.
fn month(date: jiff::civil::Date, calendar: &Value) -> String {
    let rows = calendar["rows"].as_array().map_or(&[][..], Vec::as_slice);
    let mut table = String::from(
        "<table class=\"results\"><thead><tr><th scope=\"col\">day</th><th scope=\"col\">messages</th></tr></thead><tbody>",
    );
    let mut cell = date.first_of_month();
    loop {
        let label = cell.to_string();
        let count = rows
            .iter()
            .find(|row| row["day"].as_str() == Some(label.as_str()))
            .and_then(|row| row["count"].as_u64())
            .unwrap_or(0);
        let current = if cell == date {
            " aria-current=\"date\""
        } else {
            ""
        };
        let _ = write!(
            table,
            "<tr{current}><td><a href=\"/day/{label}\">{label}</a></td><td>{count}</td></tr>"
        );
        match cell.tomorrow() {
            Ok(next) if next.month() == date.month() => cell = next,
            _ => break,
        }
    }
    table.push_str("</tbody></table>");
    table
}
