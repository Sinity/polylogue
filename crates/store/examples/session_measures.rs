//! Read latency over an explicit bench copy. Never opens the default archive.
use polylogue_store::{Archive, ReadPool, archive, derive, measure};
use serde_json::json;
use std::{env, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(args.next().ok_or("archive path required")?);
    let operation = args
        .next()
        .ok_or("derive, outlook, similar or facets required")?;
    let repetitions = args.next().map_or(Ok(3), |n| n.parse::<usize>())?;
    let archive = Archive::at(&root);
    if operation == "derive" {
        let mut conn = archive::open_write(&archive.index_db(), 64)?;
        let started = Instant::now();
        let count = derive::run(&mut conn)?;
        println!(
            "{}",
            json!({"operation":operation,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"derived_pairs":count,"index_digest":blake3::hash(polylogue_store::schema::INDEX.as_bytes()).to_hex().to_string()})
        );
        return Ok(());
    }
    let pool = ReadPool::open(&archive)?;
    let reader = pool.reader()?;
    let now = reader.query(
        "SELECT max(at)+1 AS now FROM message WHERE role='assistant' AND written_by='model'",
        &[],
    )?[0]["now"]
        .as_i64()
        .ok_or("no timed responses")?;
    let keys = reader.query(
        "SELECT pk FROM session WHERE kind='main' AND stub=0 ORDER BY id",
        &[],
    )?;
    let selection = measure::Selection {
        sessions: Some(keys.iter().filter_map(|r| r["pk"].as_i64()).collect()),
        ..Default::default()
    };
    let reference = selection
        .sessions
        .as_ref()
        .and_then(|s| s.first())
        .ok_or("no main session")?;
    let mut samples = vec![];
    let mut rows = 0;
    for _ in 0..repetitions {
        let started = Instant::now();
        let answer = match operation.as_str() {
            "outlook" => serde_json::to_value(measure::outlook::outlook(
                &reader,
                &selection,
                now,
                "bench:linear",
                "bench",
            )?)?,
            "facets" => {
                serde_json::to_value(measure::facets::facets(&reader, &selection, "deep")?)?
            }
            "similar" => {
                let sql =
                    measure::comparison::similarity_query(&format!("SELECT {reference} AS _pk"));
                // The first query page counts matches and reads one row past 20.
                let count = reader.query(&format!("SELECT count(*) AS n FROM ({sql})"), &[])?;
                let matches = reader.query(
                    &format!("SELECT * FROM ({sql}) ORDER BY score DESC,session LIMIT 21"),
                    &[],
                )?;
                json!({"matches":matches,"count":count})
            }
            _ => return Err("operation is derive, outlook, similar or facets".into()),
        };
        std::hint::black_box(&answer);
        rows = answer["matches"].as_array().map_or(0, Vec::len);
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({"operation":operation,"samples_ms":samples,"median_ms":sorted.get(sorted.len()/2),"returned_matches":rows,"selected_sessions":keys.len(),"clock":now})
    );
    Ok(())
}
