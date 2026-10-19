//! Plain-text views of `measure` answers. Unknown prints as `unknown`,
//! never as 0; partial money says how many responses it could not price.

use serde_json::Value;

pub fn render(what: &str, data: &Value) -> String {
    match what {
        "quota" | "outlook" => outlook(data),
        "usage" | "cost" => usage(data, ""),
        _ => format!(
            "{}\n",
            serde_json::to_string_pretty(data).unwrap_or_default()
        ),
    }
}

fn time(ms: &Value) -> String {
    ms.as_i64()
        .and_then(|ms| jiff::Timestamp::from_millisecond(ms).ok())
        .map(|t| {
            t.to_zoned(jiff::tz::TimeZone::system())
                .strftime("%a %Y-%m-%d %H:%M")
                .to_string()
        })
        .unwrap_or_else(|| "unknown".into())
}

fn tokens(n: u64) -> String {
    match n {
        n if n >= 1_000_000_000 => format!("{:.2}B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.1}k", n as f64 / 1e3),
        n => n.to_string(),
    }
}

fn money(m: &Value) -> String {
    let unpriced = m["unpriced_messages"].as_u64().unwrap_or(0);
    match m["known_usd"].as_f64() {
        None if unpriced > 0 => format!("unknown ({unpriced} unpriced)"),
        None => "unknown".into(),
        Some(usd) if unpriced > 0 => format!("${usd:.2} + {unpriced} unpriced"),
        Some(usd) => format!("${usd:.2}"),
    }
}

fn lane(l: &Value) -> String {
    let n = l["tokens"].as_u64().unwrap_or(0);
    let unknown = l["unknown_messages"].as_u64().unwrap_or(0);
    if unknown > 0 && n == 0 {
        "unknown".into()
    } else if unknown > 0 {
        format!("{} (+{unknown} unreported)", tokens(n))
    } else {
        tokens(n)
    }
}

fn usage(u: &Value, indent: &str) -> String {
    let l = &u["lanes"];
    let mut out = format!(
        "{indent}{} responses in {} sessions · input {} · cached {} · cache write {} · output {} · reasoning {}\n\
         {indent}API-equivalent {} · tool-reported {}",
        u["messages"],
        u["sessions"],
        lane(&l["input"]),
        lane(&l["cached_input"]),
        lane(&l["cache_write"]),
        lane(&l["output"]),
        lane(&l["reasoning"]),
        money(&u["api_cost"]),
        money(&u["reported_cost"]),
    );
    if let Some(h) = u["hidden_sessions"].as_u64().filter(|&h| h > 0) {
        out.push_str(&format!(" · includes {h} hidden session(s)"));
    }
    out.push('\n');
    if indent.is_empty() {
        for m in u["by_model"].as_array().into_iter().flatten() {
            out.push_str(&format!(
                "  {:<34} {:>8} responses  out {:>8}  {}\n",
                m["model"].as_str().unwrap_or("(no model)"),
                m["messages"],
                lane(&m["lanes"]["output"]),
                money(&m["api_cost"])
            ));
        }
        out.push_str(&format!(
            "prices: {}\n",
            u["price_catalog"].as_str().unwrap_or("")
        ));
    }
    out
}

fn outlook(o: &Value) -> String {
    let mut out = String::new();
    for t in o["tools"].as_array().into_iter().flatten() {
        out.push_str(t["tool"].as_str().unwrap_or(""));
        if let Some(p) = t["plan"].as_str() {
            out.push_str(&format!(" · plan {p}"));
        }
        if let Some(c) = t["credits"].as_object() {
            out.push_str(&format!(
                " · credits {} (as of {})",
                c["balance"].as_str().unwrap_or("?"),
                time(&c["observed_at"])
            ));
        }
        out.push('\n');
        for w in t["windows"].as_array().into_iter().flatten() {
            let minutes = w["window_minutes"].as_i64().unwrap_or(0);
            let name = match minutes {
                300 => "5-hour".to_owned(),
                10_080 => "weekly".to_owned(),
                m => format!("{m}-minute"),
            };
            let used = match w["used_percent"].as_f64() {
                Some(p) => format!("{p:.0}% used, {:.0}% left", (100.0 - p).max(0.0)),
                None => "use unknown".into(),
            };
            out.push_str(&format!(
                "  {name} {} ({}): {used} · resets {}",
                w["limit"].as_str().unwrap_or(""),
                w["basis"].as_str().unwrap_or(""),
                time(&w["resets_at"]),
            ));
            if let Some(s) = w["status"].as_str() {
                out.push_str(&format!(" · {s}"));
            }
            if let Some(b) = w["burn_percent_per_hour"].as_f64() {
                out.push_str(&format!(" · burn {b:.1}%/h"));
            }
            if !w["exhausted_at"].is_null() {
                out.push_str(&format!(" · used up {}", time(&w["exhausted_at"])));
            } else if w["lasts_to_reset"].as_bool() == Some(true) {
                out.push_str(" · lasts to reset");
            }
            out.push('\n');
            if let Some(r) = w["tokens_per_hour"].as_f64() {
                out.push_str(&format!(
                    "    {} tokens/h, projected {} at reset",
                    tokens(r as u64),
                    tokens(w["projected_tokens"].as_u64().unwrap_or(0))
                ));
                if let Some(usd) = w["api_usd_per_hour"].as_f64() {
                    out.push_str(&format!(" · ${usd:.2}/h API-equivalent"));
                }
                out.push('\n');
            }
            out.push_str(&usage(&w["usage"], "    "));
        }
    }
    out
}
