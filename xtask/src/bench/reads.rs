//! Opt-in native measurements use the existing end-to-end test owner and devkit history.

use super::*;
use devkit::{Nextest, Run};

const SCENARIO: &str = "cli::read_bench::read_benchmark";
const POPULATION: &str = "native-reads-v1";

pub(super) fn execute(repetitions: u32, full_walk: bool) -> ExitCode {
    let mut args = vec!["--repetitions".to_owned(), repetitions.to_string()];
    if full_walk {
        args.push("--full-walk".into());
    }
    let mut run = Run::start("polylogue", "bench reads", &args);
    let _ = run.step("reads", |run| {
        if !cfg!(target_os = "linux") {
            return Err("native read measurements require Linux process RSS counters".into());
        }
        let id = run.id().ok_or("benchmark history is unavailable")?;
        let out = run.root.join(bench_home()).join("reads");
        fs::create_dir_all(&out).map_err(|e|e.to_string())?;
        let report = out.join(format!("run-{id}.json"));
        if report.exists() {
            return Err(format!("benchmark output already exists: {}", report.display()).into());
        }
        let load = LoadMonitor::start()?;
        run.nextest(
            "native reads",
            &Nextest {
                args: [
                    "--locked", "--release", "-p", "polylogue", "--test", "e2e",
                    "--run-ignored", "only", "-E", &format!("test({SCENARIO})"),
                ].map(str::to_owned).to_vec(),
                tmpfs: true,
                env: vec![
                    ("POLYLOGUE_READ_BENCH_REPORT".into(), report.display().to_string()),
                    ("POLYLOGUE_READ_BENCH_REPETITIONS".into(), repetitions.to_string()),
                    ("POLYLOGUE_READ_BENCH_FULL_WALK".into(), u8::from(full_walk).to_string()),
                ],
                ..Nextest::default()
            },
        )?;
        let load = load.finish()?;
        let metrics = read_result(&fs::read_to_string(&report).map_err(|e|e.to_string())?, repetitions, full_walk)?;
        let store = Store::open().map_err(|e|e.to_string())?;
        let bench_id = bench::record(&store, &run.root, "polylogue", "reads", POPULATION, &out, &load, &metrics).map_err(|e|e.to_string())?;
        println!("{}", json!({"run":bench_id,"history_run":id,"result":metrics,"load":load.to_json(),"load_checks_passed":load.checks_passed()}));
        Ok(())
    });
    run.finish()
}

fn read_result(text: &str, repetitions: u32, full_walk: bool) -> Result<Value, String> {
    let result: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if result["population"]["id"] != POPULATION
        || result["repetitions"].as_u64() != Some(u64::from(repetitions))
        || result["answers_verified"] != true
        || result["cancelled_client_recovered"] != true
        || result["mode"] != if full_walk { "full_walk" } else { "windows" }
    {
        return Err(
            "native read benchmark omitted its matching population or completed answer checks"
                .into(),
        );
    }
    let population = &result["population"];
    let messages = population["large_messages"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("native read population has no messages")?;
    let source_bytes = population["body_bytes"]
        .as_u64()
        .and_then(|bytes| bytes.checked_mul(messages))
        .filter(|n| *n > 0)
        .ok_or("native read population has no source bytes")?;
    let groups: Vec<&Value> = if full_walk {
        vec![&result["read"]]
    } else {
        let reference = &result["read"]["reference"];
        if reference["source_messages_verified"].as_u64() != Some(messages)
            || reference["rendered_bytes"]
                .as_u64()
                .is_none_or(|n| n < source_bytes)
        {
            return Err("native reference read omitted source content".into());
        }
        ["first", "middle", "tail"]
            .iter()
            .map(|name| &result["read"]["windows"][name])
            .collect()
    };
    for group in groups {
        let walks = group["walks"]
            .as_array()
            .ok_or("native read walks are missing")?;
        if walks.len() != repetitions as usize + 1 {
            return Err("native read repetitions are incomplete".into());
        }
        let digest = walks[0]["rendered_blake3"]
            .as_str()
            .filter(|s| s.len() == 64)
            .ok_or("native read digest is missing")?;
        for walk in walks {
            if walk["pages"].as_u64().is_none_or(|n| n < 2)
                || walk["rendered_blake3"] != digest
                || walk["rss_after"]["rss_bytes"]
                    .as_u64()
                    .is_none_or(|n| n == 0)
                || if full_walk {
                    walk["source_messages_verified"].as_u64() != Some(messages)
                        || walk["rendered_bytes"]
                            .as_u64()
                            .is_none_or(|n| n < source_bytes)
                } else {
                    walk["reference_slice_verified"] != true
                        || walk["rendered_bytes"].as_u64().is_none_or(|n| n == 0)
                }
            {
                return Err(
                    "native read walk omitted matching content, bounded pages or RSS".into(),
                );
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
