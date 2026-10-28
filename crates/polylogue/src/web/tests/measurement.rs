use super::*;
use base64::Engine as _;

static MEASUREMENT: std::sync::Mutex<()> = std::sync::Mutex::new(());

async fn measure(r: &Reader, label: &str, path: &str, size: usize) {
    let load = std::fs::read_to_string("/proc/loadavg").ok();
    let mut times = Vec::new();
    let mut bytes = 0;
    for _ in 0..6 {
        let began = std::time::Instant::now();
        let (status, _, body) = send(r, authed(get(path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        bytes = body.len();
        times.push(began.elapsed().as_secs_f64() * 1000.0);
    }
    println!(
        "HTML_MEASUREMENT {}",
        serde_json::json!({"label":label,"size":size,"milliseconds":times,"bytes":bytes,"loadavg":load})
    );
}

#[test]
#[ignore = "performance measurement through the measure operation"]
fn hundred_thousand_message_html_cost() {
    let _serial = MEASUREMENT.lock().unwrap();
    let r = reader();
    let conn = rusqlite::Connection::open(r.web.daemon.archive.index_db()).unwrap();
    conn.execute_batch("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<99999)
        INSERT INTO message(id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk)
        SELECT 'claude-code:large-'||i,m.session_pk,m.order_input_pk,m.order_input_id,i,m.role,m.written_by,m.input_pk FROM n CROSS JOIN message m WHERE m.id='claude-code:u1';
        INSERT INTO block(message_pk,session_pk,idx,kind,text_id,body)
        SELECT m.pk,m.session_pk,b.idx,b.kind,b.text_id,b.body FROM message m CROSS JOIN block b WHERE b.message_pk=(SELECT pk FROM message WHERE id='claude-code:u1') AND m.id LIKE 'claude-code:large-%';").unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM message", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        100000
    );
    rt().block_on(async {
        let base = "/s/claude-code%3As1?with_parents=false";
        measure(&r, "first", base, 100000).await;
        measure(
            &r,
            "anchor-90000",
            &format!("{base}&message=claude-code%3Alarge-90000"),
            100000,
        )
        .await;
        let latest = format!("{base}&latest=true");
        measure(&r, "latest", &latest, 100000).await;
        let (status, _, body) = send(&r, authed(get(&latest)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        let previous = body
            .split("data-previous href=\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap()
            .replace("&amp;", "&");
        measure(&r, "previous-from-latest", &previous, 100000).await;
        measure(
            &r,
            "outline",
            "/api/v1/outline?session=claude-code%3As1&page_size=50",
            100000,
        )
        .await;
    });
}

#[test]
#[ignore = "performance measurement through the measure operation"]
fn document_previous_cost() {
    let _serial = MEASUREMENT.lock().unwrap();
    let r = reader();
    let source = r.w._dir.path().join("measured-document");
    std::fs::create_dir_all(source.join("projects")).unwrap();
    std::fs::write(source.join("projects/project.json"),serde_json::json!({"uuid":"measured","name":"Synthetic project","docs":[{"uuid":"doc","filename":"Synthetic document","content":"synthetic paragraph λ🙂 "}]}).to_string()).unwrap();
    assert!(matches!(
        handle(
            Request::Import {
                request_id: "measure-document".into(),
                paths: vec![source]
            },
            &r.web.daemon
        ),
        Response::Ok { .. }
    ));
    r.w.wait_for(|s| s["store"]["documents"].as_u64().is_some_and(|n| n > 0));
    let id = "claude-ai:project:measured:doc:doc";
    let conn = rusqlite::Connection::open(r.web.daemon.archive.index_db()).unwrap();
    rt().block_on(async {
        for size in [1000,100000] {
            conn.execute("WITH RECURSIVE n(i) AS (SELECT coalesce(max(idx),0)+1 FROM document_block UNION ALL SELECT i+1 FROM n WHERE i+1<?1)
                INSERT INTO document_block(document_pk,idx,kind,body,text_id,data)
                SELECT b.document_pk,n.i,b.kind,b.body,b.text_id,b.data FROM n CROSS JOIN document_block b JOIN document d ON d.pk=b.document_pk WHERE b.idx=0 AND d.id=?2",rusqlite::params![size as i64,id]).unwrap();
            let base=format!("/api/v1/read?session=document:{}&budget=64&slices=true",polylogue_render::html::encode_path_segment(id));
            let (status,_,body)=send(&r,authed(get(&base)).body(Body::empty()).unwrap()).await;
            assert_eq!(status,StatusCode::OK,"{body}");
            let first:serde_json::Value=serde_json::from_str(&body).unwrap();
            let token=first["cursor"].as_str().unwrap();
            // Issue a cursor for the last synthetic block using this test archive's key.
            let encoding=base64::engine::general_purpose::URL_SAFE_NO_PAD;
            let cursor=String::from_utf8(encoding.decode(token).unwrap()).unwrap();
            let parsed:serde_json::Value=serde_json::from_str(&cursor).unwrap();
            let seal=parsed["seal"].as_str().unwrap();
            let unsigned=cursor.replace("\"position\":[0,0,0]",&format!("\"position\":[{},0,0]",size-1)).replace(seal,"");
            let key=r.web.daemon.reads.reader().unwrap().cursor_key().unwrap();
            let seal=blake3::keyed_hash(&key,unsigned.as_bytes()).to_hex().to_string();
            let current=encoding.encode(unsigned.replace("\"seal\":\"\"",&format!("\"seal\":\"{seal}\"")));
            let (status,_,body)=send(&r,authed(get(&format!("{base}&cursor={current}"))).body(Body::empty()).unwrap()).await;
            assert_eq!(status,StatusCode::OK,"{body}");
            let current:serde_json::Value=serde_json::from_str(&body).unwrap();
            let path=format!("/d/{}?budget=64&cursor={}",polylogue_render::html::encode_path_segment(id),current["previous"].as_str().unwrap());
            measure(&r,"document-previous",&path,size).await;
        }
    });
}
