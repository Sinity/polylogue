//! Deterministic vectors and a local query embedding endpoint for synthetic archives.
use super::*;
use std::io::{Read, Write};

impl TestDaemon {
    /// Serves one query vector; later reads reuse the daemon's warm vector.
    pub fn local_embeddings(&self) -> std::thread::JoinHandle<f64> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        self.set(
            "embed.endpoint",
            &format!("http://{}", listener.local_addr().unwrap()),
        );
        self.set("embed.provider", "local");
        self.set("embed.model", "synthetic");
        let provider = std::thread::spawn(move || {
            let (mut peer, _) = listener.accept().unwrap();
            let started = std::time::Instant::now();
            peer.set_read_timeout(Some(std::time::Duration::from_secs(10)))
                .unwrap();
            let mut bytes = Vec::new();
            let mut chunk = [0; 1024];
            loop {
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
                        break;
                    }
                }
            }
            let body = r#"{"data":[{"index":0,"embedding":[1.0,0.0]}]}"#;
            write!(peer,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
            started.elapsed().as_secs_f64() * 1000.0
        });
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
                .put("local/synthetic/2/document/unit", &vectors)
                .unwrap();
        }
        provider
    }
}
