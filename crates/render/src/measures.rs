//! Plain-text views of `measure` answers. Unknown prints as `unknown`,
//! never as 0; partial money says how many responses it could not price.

use serde_json::Value;

/// Any measurement as indented JSON, for those without a text view.
pub fn json(data: &Value) -> String {
    format!(
        "{}\n",
        serde_json::to_string_pretty(data).unwrap_or_default()
    )
}

/// `usage` and `cost`: tokens by kind and cost by model.
pub fn usage(data: &Value) -> String {
    usage_at(data, "")
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
    let unknown = m["response_total_unknown"].as_bool() == Some(true);
    if unknown {
        return match m["known_usd"].as_f64() {
            Some(usd) => format!("${usd:.2} known + unknown response usage"),
            None => "unknown (response grouping unknown)".into(),
        };
    }
    match m["known_usd"].as_f64() {
        None if unpriced > 0 => format!("unknown ({unpriced} unpriced)"),
        None => "unknown".into(),
        Some(usd) if unpriced > 0 => format!("${usd:.2} + {unpriced} unpriced"),
        Some(usd) => format!("${usd:.2}"),
    }
}

fn token_count(l: &Value) -> String {
    let n = l["tokens"].as_u64().unwrap_or(0);
    let unknown = l["unknown_messages"].as_u64().unwrap_or(0);
    if l["response_total_unknown"].as_bool() == Some(true) {
        return if n == 0 {
            "unknown".into()
        } else {
            format!("{} known + unknown response usage", tokens(n))
        };
    }
    if unknown > 0 && n == 0 {
        "unknown".into()
    } else if unknown > 0 {
        format!("{} (+{unknown} unreported)", tokens(n))
    } else {
        tokens(n)
    }
}

fn usage_at(u: &Value, indent: &str) -> String {
    let l = &u["tokens"];
    let mut out = format!(
        "{indent}{} responses in {} sessions · input {} · cached {} · cache write {} · output {} · reasoning {}\n\
         {indent}API-equivalent {} · tool-reported {}",
        u["messages"]
            .as_u64()
            .map_or_else(|| "unknown".into(), |n| n.to_string()),
        u["sessions"],
        token_count(&l["input"]),
        token_count(&l["cached_input"]),
        token_count(&l["cache_write"]),
        token_count(&l["output"]),
        token_count(&l["reasoning"]),
        money(&u["api_cost"]),
        money(&u["reported_cost"]),
    );
    if let Some(h) = u["hidden_sessions"].as_u64().filter(|&h| h > 0) {
        out.push_str(&format!(" · includes {h} hidden session(s)"));
    }
    out.push('\n');
    out.push_str(&recorded_usage(&u["recorded_usage"], indent));
    if indent.is_empty() {
        for m in u["by_model"].as_array().into_iter().flatten() {
            out.push_str(&format!(
                "  {:<34} {:>8} responses  out {:>8}  {}\n",
                m["model"].as_str().unwrap_or("(no model)"),
                m["messages"],
                token_count(&m["tokens"]["output"]),
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

fn recorded_usage(reading: &Value, indent: &str) -> String {
    let Some(n) = reading["observations"].as_u64().filter(|n| *n > 0) else {
        return String::new();
    };
    let known = &reading["known_usage"];
    let count = |value: &Value| {
        value
            .as_u64()
            .map(tokens)
            .unwrap_or_else(|| "unknown".into())
    };
    let mut text = format!(
        "{indent}{n} recorded usage observations ({} measured completion observations, {} unconfirmed); {} have unknown spend time\n{indent}recorded known tokens: input {}, cached {}, output {}, reasoning {}\n",
        reading["measured_completion_observations"],
        reading["unconfirmed_observations"],
        reading["untimed_observations"],
        count(&known["input"]),
        count(&known["cached_input"]),
        count(&known["output"]),
        count(&known["reasoning"])
    );
    if let Some(usd) = reading["known_api_cost_usd"].as_f64() {
        text.push_str(&format!(
            "{indent}recorded known API-equivalent ${usd:.2} ({} unpriced observations)\n",
            reading["unpriced_observations"]
        ));
    }
    text
}

pub fn quota(o: &Value) -> String {
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
                time(&c["measured_at"])
            ));
        }
        out.push('\n');
        out.push_str(&recorded_usage(&t["recorded_usage"], "  "));
        for reading in t["unplaced"].as_array().into_iter().flatten() {
            out.push_str(&format!(
                "  {}: {} recorded use; provider time and reset unknown\n",
                reading["limit"].as_str().unwrap_or(""),
                reading["used_percent"]
                    .as_f64()
                    .map_or_else(|| "unknown".into(), |p| format!("{p:.1}%"))
            ));
        }
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
                w["source"].as_str().unwrap_or(""),
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
            if w["usage"].is_object() {
                out.push_str(&usage_at(&w["usage"], "    "));
            } else {
                out.push_str(
                    "    window usage unknown (recorded usage has no provider spend time)\n",
                );
            }
        }
    }
    out
}

/// FailingCommand commands grouped by repository, most recent failure first.
pub fn failing(data: &Value) -> String {
    let items = data.as_array().map_or(&[][..], Vec::as_slice);
    if items.is_empty() {
        return "no command's latest run failed\n".into();
    }
    // Repositories in the order of their most recent failure.
    let mut groups: Vec<(&str, Vec<String>)> = Vec::new();
    for u in items {
        let repo = u["repo"]
            .as_str()
            .or(u["repository"].as_str())
            .unwrap_or("?");
        let exit = u["exit_code"]
            .as_i64()
            .map_or(String::new(), |c| format!(" (exit {c})"));
        let since = match u["last_ok_at"] {
            Value::Null => "never succeeded".to_owned(),
            ref at => format!("last ok {}", time(at)),
        };
        let line = format!(
            "  {}  failed {}{exit} in {} · {} failure(s) since ok · {since}\n",
            u["command"].as_str().unwrap_or("?"),
            time(&u["at"]),
            u["session"].as_str().unwrap_or("?"),
            u["failures_since_ok"],
        );
        match groups.iter_mut().find(|(r, _)| *r == repo) {
            Some((_, lines)) => lines.push(line),
            None => groups.push((repo, vec![line])),
        }
    }
    let mut out = String::new();
    for (repo, lines) in groups {
        out.push_str(&format!("{repo}\n"));
        out.extend(lines);
    }
    out
}

/// Drift by format: what each parser did not know, most frequent first.
pub fn drift(data: &Value) -> String {
    let items = data.as_array().map_or(&[][..], Vec::as_slice);
    if items.is_empty() {
        return "every record and field was recognized\n".into();
    }
    let mut out = String::new();
    for d in items {
        let note = if d["by_design"].as_bool() == Some(true) {
            " (kept as a count by design)"
        } else {
            ""
        };
        out.push_str(&format!(
            "{}  {}  {} in {} input(s){note}\n",
            d["format"].as_str().unwrap_or("?"),
            d["what"].as_str().unwrap_or("?"),
            d["count"],
            d["inputs"],
        ));
    }
    out
}

fn number(value: &Value) -> String {
    if value.is_null() {
        "unknown".into()
    } else {
        value.to_string()
    }
}

/// Billing projections keep units apart and label estimates and missing inputs.
pub fn outlook(data: &Value) -> String {
    if let Some(reason) = data["reason"].as_str() {
        let detail = match reason {
            "no_cycle_anchor" => "Set cycle_anchor_day for the plan in cost.plans.",
            "plan_name_required" => "Choose a configured plan with --metric plan_name:method.",
            _ => "Configure cost.plans and choose a plan with --metric plan_name:method.",
        };
        return format!("Cost outlook unavailable: {reason}. {detail}\n");
    }
    let billing_time = |value: &Value| {
        value
            .as_i64()
            .and_then(|ms| jiff::Timestamp::from_millisecond(ms).ok())
            .map(|t| {
                t.to_zoned(jiff::tz::TimeZone::UTC)
                    .strftime("%Y-%m-%d %H:%M UTC")
                    .to_string()
            })
            .unwrap_or_else(|| "unknown".into())
    };
    let mut out = format!(
        "{} (estimate), method {}\nUTC billing window {} to {}\n",
        data["plan_name"].as_str().unwrap_or(""),
        data["projection_method"].as_str().unwrap_or(""),
        billing_time(&data["window"]["start"]),
        billing_time(&data["window"]["end"])
    );
    if let Some(units) = data["units"].as_object() {
        for (unit, values) in units {
            let label = match unit.as_str() {
                "api_usd" => "USD (API-equivalent)",
                "reported_usd" => "USD (tool-reported)",
                other => other,
            };
            out.push_str(&format!(
                "{label}: used {}, burn {}/day, projected {} (estimate)\n",
                number(&values["used"]),
                number(&values["burn_per_day"]),
                number(&values["projected"])
            ));
        }
    }
    if data["incomplete_usage"].as_bool() == Some(true) {
        out.push_str("Usage is incomplete; estimates use recorded known values.\n");
    }
    if let Some(reason) = data["quota_reason"].as_str() {
        out.push_str(&format!("Quota not configured: {reason}\n"));
    }
    if let Some(pressure) = data["quota_pressure"].as_object() {
        out.push_str(&format!(
            "Quota pressure ({}): used {}, projected {}, quota {}\n",
            pressure["unit"].as_str().unwrap_or(""),
            number(&pressure["used"]),
            number(&pressure["projected"]),
            number(&pressure["quota"])
        ));
        if let Some(reason) = pressure["reason"].as_str() {
            out.push_str(&format!("{reason}\n"));
        }
        out.push_str(&format!(
            "Possible overage cost USD: {} (estimate)\n",
            number(&pressure["projected_overage_cost_usd"])
        ));
    }
    out.push_str(&format!(
        "Configured cycle price USD: {}. Observed-day coverage: {}\n",
        number(&data["plan"]["price_usd"]),
        number(&data["coverage_ratio"])
    ));
    out
}

#[cfg(test)]
mod tests;
