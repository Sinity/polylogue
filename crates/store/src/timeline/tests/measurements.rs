//! Paired SQL and Store reads over the same synthetic metadata archive.
use super::*;
use std::time::Instant;

fn sample(name: &str, mut read: impl FnMut() -> usize) -> Value {
    let mut times = Vec::new();
    let mut count = 0;
    for _ in 0..5 {
        let start = Instant::now();
        count = read();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        eprintln!(
            "{name}: sample {} took {:.1} ms",
            times.len(),
            times.last().unwrap()
        );
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
    conn.execute_batch("DROP INDEX message_at; DROP INDEX event_at; DROP INDEX message_session_at; DROP INDEX message_model_at; DROP INDEX message_session_model_at;")
        .unwrap();
    let setup = Instant::now();
    conn.execute(
        "WITH RECURSIVE n(i) AS (VALUES(1) UNION ALL SELECT i+1 FROM n WHERE i<999)
        INSERT INTO session(pk,id,origin,native_id,kind,changed)
        SELECT ?1+i,printf('codex:synthetic-%04d',i),'codex',printf('synthetic-%04d',i),'main',1 FROM n",
        [session],
    )
    .unwrap();
    conn.execute(&format!("WITH RECURSIVE n(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM n WHERE i<999999)
        INSERT INTO message(id,session_pk,order_input_pk,order_input_id,seq,role,written_by,model,at,input_pk,cost_nusd)
        SELECT printf('codex:%07d',i),CASE WHEN i>=999000 THEN {session}+999 ELSE {session}+(i%999) END,1,'fixture',i,'assistant','model',CASE WHEN i>=999900 THEN 'rare' WHEN i%10=0 THEN NULL ELSE 'synthetic' END,?1+(CASE WHEN i>=999000 THEN 5843 ELSE i%5844 END)*86400000,1,5000000 FROM n"), [start]).unwrap();
    conn.execute(
        &format!(
            "WITH RECURSIVE n(i) AS (VALUES(0) UNION ALL SELECT i+1 FROM n WHERE i<99999)
        INSERT INTO event(id,session_pk,seq,kind,at,input_pk,data)
        SELECT printf('event:%07d',i),{session}+(i%1000),i,'synthetic',?1+(i%5844)*86400000,1,?2 FROM n"
        ),
        rusqlite::params![start, crate::encoding::pack_json(&json!({})).unwrap()],
    )
    .unwrap();
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
    eprintln!("synthetic archive: {setup_ms:.1} ms");
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
            "m.role='assistant' AND m.written_by='model' AND coalesce(json_extract(m.attrs_json,'$.production.response_boundary_unknown'),0)=0 AND NOT EXISTS(SELECT 1 FROM event e WHERE e.session_pk=m.session_pk AND e.kind='response_boundaries_unknown')"
        );
        json!({"day":sample("before day",||r.query(&page_sql,&[]).unwrap().len()),"across_years":sample("before across years",||r.query(&years_sql,&[]).unwrap().len()),"calendar":sample("before calendar",||r.query(&calendar_sql,&[]).unwrap().len()),"cost_page":sample("before cost page",||r.query(&cost_sql,&[]).unwrap().len()),"plans":{"day":r.query(&format!("EXPLAIN QUERY PLAN {page_sql}"),&[]).unwrap(),"cost_page":r.query(&format!("EXPLAIN QUERY PLAN {cost_sql}"),&[]).unwrap()}})
    };
    let late_session = crate::measure::Selection {
        sessions: Some(vec![session + 999]),
        ..Default::default()
    };
    let late_cost = Selection { responses: Some(&late_session), ..all(None, false, false) };
    let usage = crate::measure::Selection::default();
    let rare_cost = Selection { responses: Some(&usage), model: Some("rare"), ..all(None, false, false) };
    let selected_rare = crate::measure::Selection {
        sessions: Some(vec![session + 998, session + 999]),
        ..Default::default()
    };
    let selected_rare_cost = Selection { responses: Some(&selected_rare), model: Some("rare"), ..all(None, false, false) };
    // The reviewed narrow reads already had global time indexes, but still
    // walked them from the earliest response to find sparse matches.
    let global_index_clock = Instant::now();
    conn.execute_batch("CREATE INDEX message_at ON message(at,id); CREATE INDEX event_at ON event(at,id);").unwrap();
    let global_index_ms = global_index_clock.elapsed().as_secs_f64() * 1000.0;
    let sparse_before = {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        let read = |name: &str, selection: &Selection<'_>| {
            let (source, params) = sources(selection, None);
            let sql = format!("SELECT * FROM ({}) WHERE {} ORDER BY at,id LIMIT 51", source[0].replace("m.session_pk IN", "+m.session_pk IN"), representable("at", false));
            json!({"timing":sample(name, || r.query(&sql, &params).unwrap().len()), "plan":r.query(&format!("EXPLAIN QUERY PLAN {sql}"), &params).unwrap()})
        };
        json!({"late_session":read("before late session", &late_cost), "rare_model":read("before rare model", &rare_cost),"selected_rare_model":read("before selected rare model", &selected_rare_cost)})
    };
    let index_clock = Instant::now();
    crate::schema::ensure_index(&conn).unwrap();
    let index_ms = index_clock.elapsed().as_secs_f64() * 1000.0;
    eprintln!("global time indexes: {global_index_ms:.1} ms; session and model indexes: {index_ms:.1} ms");
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let usage = crate::measure::Selection::default();
    let cost = Selection {
        responses: Some(&usage),
        ..all(None, false, false)
    };
    let selected = crate::measure::Selection {
        sessions: Some((session..session + 1000).collect()),
        ..Default::default()
    };
    let selected_cost = Selection {
        responses: Some(&selected),
        ..all(None, false, false)
    };
    let plans = |selection: &Selection<'_>, bounds| {
        row_queries(&r, selection, bounds, None, 51, false).unwrap().into_iter()
            .map(|(sql, params)| r.query(&format!("EXPLAIN QUERY PLAN {sql}"), &params).unwrap())
            .collect::<Vec<_>>()
    };
    let response_plan = plans(&selected_cost, None);
    let day_plans = plans(&all(Some(date), false, false), Some(day.day(date)));
    let after = json!({"day":sample("after day",||page(&r,&day,&all(Some(date),false,false),None,50).unwrap()["rows"].as_array().unwrap().len()),"across_years":sample("after across years",||page(&r,&day,&all(Some(date),true,false),None,50).unwrap()["rows"].as_array().unwrap().len()),"calendar":sample("after calendar",||calendar(&r,&day,date).unwrap().len()),"cost_page":sample("after cost page",||page(&r,&day,&cost,None,50).unwrap()["rows"].as_array().unwrap().len()),"selected_cost_page":sample("after selected cost page",||page(&r,&day,&selected_cost,None,50).unwrap()["rows"].as_array().unwrap().len()),"late_session":sample("after late session",||page(&r,&day,&late_cost,None,50).unwrap()["rows"].as_array().unwrap().len()),"rare_model":sample("after rare model",||page(&r,&day,&rare_cost,None,50).unwrap()["rows"].as_array().unwrap().len()),"selected_rare_model":sample("after selected rare model",||page(&r,&day,&selected_rare_cost,None,50).unwrap()["rows"].as_array().unwrap().len()),"plans":{"selected_rare_model":plans(&selected_rare_cost,None),"late_session":plans(&late_cost,None),"rare_model":plans(&rare_cost,None),"selected_cost_page":response_plan,"day_sources":day_plans,"message":r.query("EXPLAIN QUERY PLAN SELECT at,id FROM message WHERE at>=?1 AND at<?2 ORDER BY at,id LIMIT 51",&[SqlValue::Int(day.day(date).0),SqlValue::Int(day.day(date).1)]).unwrap(),"event":r.query("EXPLAIN QUERY PLAN SELECT at,id FROM event WHERE at>=?1 AND at<?2 ORDER BY at,id LIMIT 51",&[SqlValue::Int(day.day(date).0),SqlValue::Int(day.day(date).1)]).unwrap()}});
    let report = json!({"messages":1000000,"events":100000,"sessions":1000,"years":16,"page_size":50,"samples_per_action":5,"setup_ms":setup_ms,"index_ms":global_index_ms+index_ms,"global_time_indexes_ms":global_index_ms,"selection_time_indexes_ms":index_ms,"before":baseline,"sparse_before":sparse_before,"late_session_messages":1000,"rare_model_messages":100,"after":after,"load":std::fs::read_to_string("/proc/loadavg").ok(),"memory":std::fs::read_to_string("/proc/self/status").ok().map(|s|s.lines().filter(|l|l.starts_with("VmRSS:")||l.starts_with("VmHWM:")).collect::<Vec<_>>().join("; "))});
    let path = std::env::var_os("POLYLOGUE_TIMELINE_REPORT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/timeline-million.json")
        });
    std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("TIMELINE_MEASUREMENTS={report}");
}
