//! One answer whichever way it is read: whole, in pages of any size, or
//! kept and read back.

use super::*;

/// Each row's identity: its ID, or a tool call's message and command.
fn identities(v: &Value) -> Vec<Value> {
    v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| match r.get("id") {
            Some(id) => id.clone(),
            None => serde_json::json!([r["message"], r["command"]]),
        })
        .collect()
}

#[test]
fn every_way_of_reading_gives_the_same_answer() {
    let mut w = morrow();
    for (n, text) in [
        "kind:*",
        "messages where parser | union (messages where chrono)",
        "tool_calls where outcome:error",
        "sessions | sort by cost desc",
        "origin:claude-code | except (cost > 2)",
    ]
    .into_iter()
    .enumerate()
    {
        let mut whole = ask(text);
        whole.page_size = Some(1000);
        let whole = run_q(&w, whole).unwrap();
        let expected = identities(&whole);
        for size in [1, 2, 3] {
            let mut rows = vec![];
            let mut cursor: Option<String> = None;
            loop {
                let mut request = ask(text);
                request.page_size = Some(size);
                request.cursor = cursor.as_deref();
                let page = run_q(&w, request).unwrap();
                assert_eq!(page["count"], whole["count"], "{text}");
                rows.extend(identities(&page));
                match page["next"].as_str() {
                    Some(next) => cursor = Some(next.to_owned()),
                    None => break,
                }
            }
            assert_eq!(rows, expected, "{text}, pages of {size}");
        }
        let kept = {
            let pool = w.fx.reads().unwrap();
            let reader = pool.reader().unwrap();
            keep(&reader, ask(text)).unwrap()
        };
        let id = format!("kept-{n}");
        w.fx.kept_result(&id, text, &kept.to_string()).unwrap();
        let read_back = q(&w, &format!("result:{id}"));
        assert_eq!(read_back["count"], whole["count"], "{text}");
        assert_eq!(identities(&read_back), expected, "{text}, kept");
    }
}
