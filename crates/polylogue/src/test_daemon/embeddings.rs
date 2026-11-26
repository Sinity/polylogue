//! Deterministic vectors and a local query embedding endpoint for synthetic archives.
use super::*;
use std::io::{Read, Write};

/// Reads one request from `peer`, answers it with `status` and `body`, and
/// returns the request's JSON body: the exact inputs submitted.
pub(crate) fn serve_request(
    mut peer: std::net::TcpStream,
    status: u16,
    body: &str,
) -> serde_json::Value {
    peer.set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut chunk = [0; 1024];
    let body_start = loop {
        let n = peer.read(&mut chunk).unwrap();
        assert!(n > 0);
        bytes.extend_from_slice(&chunk[..n]);
        if let Some(end) = bytes.windows(4).position(|v| v == b"\r\n\r\n") {
            let headers = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
            let size: usize = headers
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            if bytes.len() >= end + 4 + size {
                break end + 4;
            }
        }
    };
    let reply = format!(
        "HTTP/1.1 {status} Synthetic\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    peer.write_all(reply.as_bytes()).unwrap();
    serde_json::from_slice(&bytes[body_start..]).unwrap()
}

impl TestDaemon {
    /// Serves one query vector; later reads reuse the daemon's warm vector.
    pub fn local_embeddings(&self) -> std::thread::JoinHandle<()> {
        let listener = self.embedding_endpoint();
        std::thread::spawn(move || {
            let (peer, _) = listener.accept().unwrap();
            serve_request(peer, 200, r#"{"data":[{"index":0,"embedding":[1.0,0.0]}]}"#);
        })
    }

    /// Points the daemon's local embedding provider at the returned
    /// listener and stores a vector for every text: high for
    /// `claude-code:s1`'s texts, low for the rest.
    pub(crate) fn embedding_endpoint(&self) -> std::net::TcpListener {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        self.set(
            "embed.endpoint",
            &format!("http://{}", listener.local_addr().unwrap()),
        );
        self.set("embed.provider", "local");
        self.set("embed.calls", "on");
        self.set("embed.model", "synthetic");
        {
            let reader = self.daemon.reads.reader().unwrap();
            let rows = reader.query("SELECT hex(t.hash) AS h, EXISTS(SELECT 1 FROM block b JOIN session s ON s.pk=b.session_pk WHERE b.text_id=t.id AND s.id='claude-code:s1') AS high FROM text t",&[]).unwrap();
            let vectors = rows
                .iter()
                .map(|row| {
                    let h = row["h"].as_str().unwrap();
                    polylogue_store::vectors::Vector {
                        text_hash: (0..h.len())
                            .step_by(2)
                            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
                            .collect(),
                        chunk: 0,
                        values: if row["high"] == 1 {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        },
                    }
                })
                .collect::<Vec<_>>();
            reader
                .vectors_writer()
                .unwrap()
                .put(
                    "local/synthetic/2/document/unit",
                    &vectors,
                    &polylogue_model::WallClock::new(jiff::Timestamp::now),
                )
                .unwrap();
        }
        listener
    }
}

/// A ranked command stopped at its vector-store read until dropped.
pub(crate) struct VectorLoad(
    Option<std::thread::JoinHandle<command_surface::Answer<serde_json::Value>>>,
);

pub(crate) fn vector_query() -> super::Call {
    let mut call = super::Call::new(
        "rows",
        serde_json::json!({"selection": "messages where similar:parser"}),
    );
    call.deadline = command_surface::Deadline::None;
    call
}

impl VectorLoad {
    pub(crate) fn start(daemon: &TestDaemon) -> Self {
        use polylogue_store::fault::{self, Action};
        drop(daemon.embedding_endpoint());
        let reader = daemon.daemon.reads.reader().unwrap();
        reader
            .vectors_writer()
            .unwrap()
            .put_query_vector(
                "local/synthetic/2/query/unit",
                "parser",
                &[1., 0.],
                &daemon.daemon.clock,
            )
            .unwrap();
        drop(reader);
        fault::clear();
        fault::arm("vectors.read", "local/synthetic/", Action::Hold, 0, 1);
        let daemon = std::sync::Arc::clone(&daemon.daemon);
        let thread = std::thread::spawn(move || vector_query().answer(&daemon));
        let loading = Self(Some(thread));
        assert!(
            fault::wait_held_within(std::time::Duration::from_secs(10)),
            "request did not reach the vector store"
        );
        loading
    }
}

impl VectorLoad {
    pub(crate) fn finish(mut self) -> command_surface::Answer<serde_json::Value> {
        polylogue_store::fault::release();
        let answer = self.0.take().unwrap().join().unwrap();
        polylogue_store::fault::clear();
        answer
    }
}

impl Drop for VectorLoad {
    fn drop(&mut self) {
        if let Some(thread) = self.0.take() {
            polylogue_store::fault::release();
            let _ = thread.join();
            polylogue_store::fault::clear();
        }
    }
}
