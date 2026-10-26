//! A first build's whole-corpus time, predicted stage by stage from sample
//! runs of one build at two or more sizes. Each stage is fitted as fixed
//! seconds plus seconds per input MB (least squares), because the stages at
//! a build's end (search entries, trigram index, secondary indexes, derive)
//! need not grow in proportion to the input, and a single sample cannot
//! tell.

use super::*;

/// One recorded sample run: its input and its times.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Point {
    pub input_mb: f64,
    pub seconds: f64,
    pub stages: BTreeMap<String, f64>,
}

impl Point {
    fn of(metrics: &Value) -> Option<Self> {
        Some(Self {
            input_mb: metrics["input_mb"]
                .as_f64()
                .filter(|n| n.is_finite() && *n > 0.0)?,
            seconds: metrics["seconds"]
                .as_f64()
                .filter(|n| n.is_finite() && *n >= 0.0)?,
            stages: stages_of(metrics),
        })
    }
}

/// Fixed seconds and seconds per MB fitting `(input MB, seconds)` pairs.
fn line(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if sxx.is_nan() || sxx <= 0.0 {
        return None;
    }
    let slope = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / sxx;
    Some((my - slope * mx, slope))
}

/// The predicted seconds of a build over `input_mb`, in total and per stage,
/// from points at two or more distinct sizes.
pub(super) fn fit(points: &[Point], input_mb: f64) -> Result<Value, String> {
    let mut sizes: Vec<f64> = points.iter().map(|p| p.input_mb).collect();
    sizes.sort_by(f64::total_cmp);
    sizes.dedup();
    if sizes.len() < 2 {
        return Err("a fit needs runs of one build at two or more sample sizes".into());
    }
    let predict = |pairs: Vec<(f64, f64)>| -> Value {
        match line(&pairs) {
            Some((fixed, per_mb)) => json!({
                "seconds": fixed + per_mb * input_mb,
                "fixed_seconds": fixed,
                "seconds_per_gb": per_mb * 1000.0,
                "observed": pairs.iter().map(|(x, y)| json!([x, y])).collect::<Vec<_>>(),
            }),
            None => Value::Null,
        }
    };
    let names: std::collections::BTreeSet<&String> =
        points.iter().flat_map(|p| p.stages.keys()).collect();
    let stages: serde_json::Map<String, Value> = names
        .into_iter()
        .map(|name| {
            let pairs = points
                .iter()
                .map(|p| (p.input_mb, p.stages.get(name).copied().unwrap_or(0.0)))
                .collect();
            (name.clone(), predict(pairs))
        })
        .collect();
    Ok(json!({
        "method": "fixed_plus_per_mb_least_squares",
        "input_mb": input_mb,
        "sample_input_mb": sizes,
        "seconds": predict(points.iter().map(|p| (p.input_mb, p.seconds)).collect()),
        "stages": stages,
        "assumptions": [
            "Each stage's time is fixed seconds plus seconds per input MB over the sampled sizes and up to the whole corpus.",
            "Pool stages are thread-seconds and do not add up to the wall time, which is fitted on its own."
        ],
    }))
}

/// A build's recorded sample runs, as fit points.
struct Recorded {
    points: Vec<Point>,
    /// The corpus input MB the latest run stands for.
    corpus: Option<f64>,
    runs: Vec<i64>,
}

/// The latest disk sample run of `build` at each sample, as fit points, with
/// the corpus input MB the latest one stands for.
fn points(build: &str) -> Result<Recorded, String> {
    let store = Store::open().map_err(|e| e.to_string())?;
    let runs = bench::runs(&store, "polylogue", 500).map_err(|e| e.to_string())?;
    let mut seen = std::collections::BTreeSet::new();
    let mut points = Vec::new();
    let mut corpus = None;
    let mut ids = Vec::new();
    for run in runs {
        let of_build = run.metrics["build"].as_str().map_or_else(
            || run.head.as_deref() == Some(build),
            |recorded| recorded == build,
        );
        if !(of_build
            && run.storage == "disk"
            && matches!(run.bench.as_str(), "first_build" | "first_build-pair"))
        {
            continue;
        }
        if !seen.insert(run.sample.clone()) {
            continue;
        }
        if let Some(point) = Point::of(&run.metrics) {
            corpus = corpus.or(run.metrics["corpus_input_mb"].as_f64());
            ids.push(run.id);
            points.push(point);
        }
    }
    Ok(Recorded {
        points,
        corpus,
        runs: ids,
    })
}

pub(super) fn estimate(build: Option<String>) -> Result<(), String> {
    let build = match build {
        Some(build) => build,
        None => head()?,
    };
    let Recorded {
        points,
        corpus,
        runs,
    } = points(&build)?;
    let corpus = corpus.ok_or("no run of this build records the corpus size it stands for")?;
    let mut report = fit(&points, corpus)?;
    report["build"] = json!(build);
    report["runs"] = json!(runs);
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

pub(super) fn validate(path: &Path, build: Option<String>) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let actual = text
        .lines()
        .rev()
        .find_map(|l| {
            serde_json::from_str::<Value>(l.trim())
                .ok()
                .filter(|v| v.get("seconds").is_some())
        })
        .ok_or("no first_build result line in that file")?;
    let build = match build {
        Some(build) => build,
        None => head()?,
    };
    let Recorded { points, runs, .. } = points(&build)?;
    let input_mb = actual["input_mb"]
        .as_f64()
        .ok_or("the supplied run has no input_mb")?;
    let estimate = fit(&points, input_mb)?;
    let mut report = comparison(&estimate, &actual)?;
    report["build"] = json!(build);
    report["runs"] = json!(runs);
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}

/// A fitted estimate against a full run, in total and per stage.
pub(super) fn comparison(estimate: &Value, actual: &Value) -> Result<Value, String> {
    let real = actual["seconds"]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 0.0)
        .ok_or("the supplied run has no valid seconds value")?;
    let row = |estimated: Option<f64>, real: Option<f64>| {
        json!({
            "estimated": estimated, "actual": real,
            "relative_error": estimated.zip(real).filter(|(_, r)| *r > 0.0).map(|(e, r)| (e - r) / r),
        })
    };
    let actual_stages = stages_of(actual);
    let mut stages = serde_json::Map::new();
    for (name, fitted) in estimate["stages"].as_object().into_iter().flatten() {
        stages.insert(
            name.clone(),
            row(fitted["seconds"].as_f64(), actual_stages.get(name).copied()),
        );
    }
    Ok(json!({
        "seconds": row(estimate["seconds"]["seconds"].as_f64(), Some(real)),
        "stages": stages,
        "actual_input_mb": actual["input_mb"],
        "estimate": estimate,
        "input_match": "not_verified",
    }))
}

/// The second result's change from the first, in total and per stage.
pub(super) fn difference(a: &Value, b: &Value) -> Value {
    let change = |x: Option<f64>, y: Option<f64>| {
        json!({
            "a": x, "b": y,
            "change": x.zip(y).filter(|(x, _)| *x > 0.0).map(|(x, y)| (y - x) / x),
        })
    };
    let (sa, sb) = (stages_of(a), stages_of(b));
    let names: std::collections::BTreeSet<&String> = sa.keys().chain(sb.keys()).collect();
    let mut out = serde_json::Map::new();
    for key in ["seconds", "disk_written_mb", "disk_read_mb", "peak_rss_mb"] {
        out.insert(key.into(), change(a[key].as_f64(), b[key].as_f64()));
    }
    out.insert(
        "stages".into(),
        names
            .into_iter()
            .map(|n| (n.clone(), change(sa.get(n).copied(), sb.get(n).copied())))
            .collect::<serde_json::Map<_, _>>()
            .into(),
    );
    out.into()
}

#[cfg(test)]
mod tests;
