use super::*;

#[test]
#[ignore = "performance measurement through the measure operation"]
fn hundred_thousand_message_html_cost() {
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
        for path in ["/s/claude-code%3As1?with_parents=false", "/s/claude-code%3As1?with_parents=false&message=claude-code%3Alarge-90000"] {
            let mut times = Vec::new();
            let mut bytes = 0;
            for _ in 0..6 {
                let began = std::time::Instant::now();
                let (status, _, body) = send(&r, authed(get(path)).body(Body::empty()).unwrap()).await;
                assert_eq!(status,StatusCode::OK,"{body}");
                assert!(body.contains("Source message"));
                bytes = body.len();
                times.push(began.elapsed().as_secs_f64()*1000.0);
            }
            println!("HTML_MEASUREMENT {}",serde_json::json!({"messages":100000,"path":path,"milliseconds":times,"bytes":bytes}));
        }
    });
}
