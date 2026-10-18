//! The `polylogue mcp` stdio server against a running daemon, driven as an
//! agent would: discover, check status, search, page a read and write a note.
use super::*;
use std::{
    process::ChildStdin,
    sync::mpsc::{self, Receiver},
};

const META: &str = r#"{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{},"io.modelcontextprotocol/clientInfo":{"name":"scenario","version":"1"}}"#;

struct Client {
    _child: OwnedChild,
    input: ChildStdin,
    lines: Receiver<String>,
    next_id: u64,
}

impl Client {
    fn start(world: &World) -> Self {
        let machine = world.input("machine-id", b"00112233445566778899aabbccddeeff\n");
        let mut child = world
            .command()
            .arg("mcp")
            .env("POLYLOGUE_MACHINE_ID_FILE", machine)
            .env("CODEX_THREAD_ID", "scenario-agent")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        let (send, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in stdout.lines() {
                if send.send(line.unwrap()).is_err() {
                    break;
                }
            }
        });
        Self {
            _child: OwnedChild(child),
            input,
            lines,
            next_id: 0,
        }
    }

    fn rpc(&mut self, method: &str, mut params: Value) -> Value {
        self.next_id += 1;
        params["_meta"] = serde_json::from_str(META).unwrap();
        let request = json!({"jsonrpc":"2.0","id":self.next_id,"method":method,"params":params});
        writeln!(self.input, "{request}").unwrap();
        let line = self
            .lines
            .recv_timeout(Duration::from_secs(40))
            .unwrap_or_else(|e| panic!("{method}: no reply: {e}"));
        let reply: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(reply["id"], self.next_id, "{reply}");
        reply
    }

    /// Returns the structured result; `isError` must agree with its outcome.
    fn tool(&mut self, name: &str, arguments: Value) -> Value {
        let reply = self.rpc("tools/call", json!({"name":name,"arguments":arguments}));
        let result = &reply["result"];
        assert!(result.is_object(), "{name}: {reply}");
        let content = &result["structuredContent"];
        assert_eq!(result["isError"], content["outcome"] == "error", "{reply}");
        content.clone()
    }
}

#[test]
fn mcp_client_searches_pages_a_dialogue_read_and_writes_a_note() {
    let world = World::new();
    let input = world.input("mcp-morrow.jsonl", MORROW);
    let receipt = world.import(&input, "01900000-0000-7000-8000-000000000911");
    world.read(
        &reference(),
        receipt["retained"]["id"].as_str().unwrap(),
        12,
    );
    let mut client = Client::start(&world);

    let discovery = client.rpc("server/discover", json!({}));
    assert_eq!(discovery["result"]["resultType"], "complete", "{discovery}");

    let status = client.tool("status", json!({}));
    assert_ne!(status["outcome"], "error", "{status}");

    let search = client.tool("search", json!({"query":"relative dates"}));
    // Prose search reports a partial population, so it is degraded, not ok.
    assert_eq!(search["outcome"], "degraded", "{search}");
    assert_eq!(search["rows"][0]["id"], reference(), "{search}");

    // A small budget forces several pages. Each MCP page must match the page
    // the CLI returns for the same call.
    let mut call = json!({"ref":reference(),"view":"dialogue","budget_tokens":1000});
    let mut texts = vec![];
    let mut pages = 0;
    loop {
        let page = client.tool("read", call.clone());
        assert_ne!(page["outcome"], "error", "{page}");
        let cli = model_call(&world, &call, Some(1000));
        assert_eq!(page["shown"], cli["shown"]);
        for span in page["shown"].as_array().unwrap() {
            texts.push(span["text"].as_str().unwrap().to_owned());
        }
        pages += 1;
        if page["next"].is_null() {
            break;
        }
        assert_eq!(page["next"]["tool"], "read");
        call = page["next"]["arguments"].clone();
    }
    assert!(pages > 1, "budget did not force paging");
    let dialogue = texts.concat();
    assert!(dialogue.starts_with("Set up a new Rust library crate called morrow"));
    assert!(dialogue.ends_with("Next step: implement relative dates in parse.rs."));

    let target = model_ref(&fixture_records()[6]["uuid"].as_str().unwrap());
    let note = client.tool(
        "note",
        json!({"targets":[target],"value":"Check the parse module next."}),
    );
    assert_eq!(note["outcome"], "ok", "{note}");
    let written = client.tool("read", json!({"ref":note["ref"]}));
    assert_eq!(written["outcome"], "ok", "{written}");
    assert!(written.to_string().contains("Check the parse module next."));
}
