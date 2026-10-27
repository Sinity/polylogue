//! Paired SQL and Store reads over the same synthetic metadata archive.
use super::*;
use std::time::Instant;

fn sample(mut read: impl FnMut() -> usize) -> Value {
    let mut times = Vec::new();
    let mut count = 0;
    for _ in 0..5 {
        let start = Instant::now();
        count = read();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let raw = times.clone();
    times.sort_by(f64::total_cmp);
    json!({"median_ms":times[2],"samples_ms":raw,"rows":count})
}

#[test]
#[ignore = "million-message paired timings; run through AgentCTL measure"]
fn million_message_measurements() {
    let temp = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(temp.path()).unwrap();
    let session = f
        .session(SessionFacts {
            id: "codex:synthetic",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    let conn =
        crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP).unwrap();
    let start = at("2010-01-01T12:00:00Z");
    conn.execute_batch("DROP INDEX message_at; DROP INDEX event_at;")
        .unwrap();
    let setup = Instant::now();
    conn.execute(&format!("WITH RECURSIVE n(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM n WHERE i<999999)
        INSERT INTO message(id,session_pk,order_input_pk,order_input_id,seq,role,written_by,model,at,input_pk,cost_nusd)
        SELECT printf('codex:%07d',i),{session},1,'fixture',i,'assistant','model',CASE WHEN i%10=0 THEN NULL ELSE 'synthetic' END,?1+(i%5844)*86400000,1,5000000 FROM n"), [start]).unwrap();
    conn.execute(
        &format!(
            "WITH RECURSIVE n(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM n WHERE i<99999)
        INSERT INTO event(id,session_pk,seq,kind,at,input_pk,data)
        SELECT printf('event:%07d',i),{session},i,'synthetic',?1+(i%5844)*86400000,1,x'' FROM n"
        ),
        [start],
    )
    .unwrap();
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    let day = LogicalDay::named("Europe/Warsaw", 240).unwrap();
    let date = "2024-03-31".parse().unwrap();
    let baseline = {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        dates(&r, &day).unwrap();
        // The reviewed page queries, before per-source ranges and time indexes.
        let source = "SELECT m.at, 'message' AS kind,m.id FROM message m UNION ALL SELECT e.at,'event',e.id FROM event e UNION ALL SELECT n.created_at,'note',n.id FROM a.note n";
        let (since, until) = day.day(date);
        let (month_since, month_until) = day.month(date);
        let page_sql = format!(
            "WITH activity AS ({source}),dated AS (SELECT *,pl_timeline_date(at) AS date FROM activity) SELECT at,kind,id,date FROM dated WHERE date IS NOT NULL AND at >= {since} AND at < {until} AND date='2024-03-31' ORDER BY date IS NULL,CASE WHEN date IS NULL THEN 0 ELSE at END,kind,id LIMIT 51"
        );
        let years_sql = format!(
            "WITH activity AS ({source}),dated AS (SELECT *,pl_timeline_date(at) AS date FROM activity) SELECT at,kind,id,date FROM dated WHERE date IS NOT NULL AND substr(date,-5)='03-31' AND at < {until} ORDER BY date IS NULL,CASE WHEN date IS NULL THEN 0 ELSE at END,kind,id LIMIT 51"
        );
        let calendar_sql = format!(
            "WITH activity AS ({source}) SELECT pl_timeline_date(at) AS date,count(*) AS records FROM activity WHERE at >= {month_since} AND at < {month_until} GROUP BY date ORDER BY date"
        );
        let cost_sql = format!(
            "WITH activity AS (SELECT m.at,'message' AS kind,m.id FROM message m WHERE {}),dated AS (SELECT *,pl_timeline_date(at) AS date FROM activity) SELECT at,kind,id,date FROM dated ORDER BY date IS NULL,CASE WHEN date IS NULL THEN 0 ELSE at END,kind,id LIMIT 51",
            crate::measure::RESPONSE_ROWS
        );
        json!({"day":sample(||r.query(&page_sql,&[]).unwrap().len()),"across_years":sample(||r.query(&years_sql,&[]).unwrap().len()),"calendar":sample(||r.query(&calendar_sql,&[]).unwrap().len()),"cost_page":sample(||r.query(&cost_sql,&[]).unwrap().len()),"plans":{"day":r.query(&format!("EXPLAIN QUERY PLAN {page_sql}"),&[]).unwrap(),"cost_page":r.query(&format!("EXPLAIN QUERY PLAN {cost_sql}"),&[]).unwrap()}})
    };
    let index_clock = Instant::now();
    crate::schema::ensure_index(&conn).unwrap();
    let index_ms = index_clock.elapsed().as_secs_f64() * 1000.0;
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let usage = crate::measure::Selection::default();
    let cost = Selection {
        responses: Some(&usage),
        ..all(None, false, false)
    };
    let after = json!({"day":sample(||page(&r,&day,&all(Some(date),false,false),None,50).unwrap()["rows"].as_array().unwrap().len()),"across_years":sample(||page(&r,&day,&all(Some(date),true,false),None,50).unwrap()["rows"].as_array().unwrap().len()),"calendar":sample(||calendar(&r,&day,date).unwrap().len()),"cost_page":sample(||page(&r,&day,&cost,None,50).unwrap()["rows"].as_array().unwrap().len()),"plans":{"message":r.query("EXPLAIN QUERY PLAN SELECT at,id FROM message WHERE at>=?1 AND at<?2 ORDER BY at,id LIMIT 51",&[SqlValue::Int(day.day(date).0),SqlValue::Int(day.day(date).1)]).unwrap(),"event":r.query("EXPLAIN QUERY PLAN SELECT at,id FROM event WHERE at>=?1 AND at<?2 ORDER BY at,id LIMIT 51",&[SqlValue::Int(day.day(date).0),SqlValue::Int(day.day(date).1)]).unwrap()}});
    let report = json!({"messages":1000000,"events":100000,"years":16,"page_size":50,"samples_per_action":5,"setup_ms":setup_ms,"index_ms":index_ms,"before":baseline,"after":after,"load":std::fs::read_to_string("/proc/loadavg").ok(),"memory":std::fs::read_to_string("/proc/self/status").ok().map(|s|s.lines().filter(|l|l.starts_with("VmRSS:")||l.starts_with("VmHWM:")).collect::<Vec<_>>().join("; "))});
    let path = std::env::var_os("POLYLOGUE_TIMELINE_REPORT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/timeline-million.json")
        });
    std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("TIMELINE_MEASUREMENTS={report}");
}
