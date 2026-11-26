use super::*;

/// A client that takes `open` bytes without waiting, then nothing until
/// the reply is finished, then everything.
struct Client {
    open: usize,
    received: Vec<u8>,
    waited: usize,
}

impl Sink for Client {
    fn try_send(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let taken = bytes.len().min(self.open);
        self.open -= taken;
        self.received.extend_from_slice(&bytes[..taken]);
        Ok(taken)
    }
    fn send(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.waited += bytes.len();
        self.received.extend_from_slice(bytes);
        Ok(())
    }
}

/// A slow client never makes a write wait: what it does not take goes to
/// memory and then to the spill file, and every byte arrives in order once
/// the reply is finished.
#[test]
fn a_slow_client_gets_every_byte_in_order_without_a_write_waiting() {
    let dir = tempfile::tempdir().unwrap();
    let mut spill = Spill::new(
        Client {
            open: 10,
            received: Vec::new(),
            waited: 0,
        },
        dir.path().to_owned(),
    );
    let mut sent = Vec::new();
    for n in 0..3_000_u32 {
        let piece = format!("event {n} {}\n", "x".repeat(n as usize % 997));
        spill.write(piece.as_bytes()).unwrap();
        sent.extend_from_slice(piece.as_bytes());
    }
    assert!(sent.len() > MEMORY, "the reply outgrows the memory share");
    assert_eq!(spill.sink().waited, 0, "no write waited for the client");
    assert_eq!(spill.sink().received.len(), 10);
    let client = spill.finish().unwrap();
    assert_eq!(client.received, sent);
}

/// A client that keeps up takes each write as it is made, with nothing held.
#[test]
fn a_client_that_keeps_up_takes_each_write_at_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut spill = Spill::new(
        Client {
            open: usize::MAX,
            received: Vec::new(),
            waited: 0,
        },
        dir.path().to_owned(),
    );
    spill.write(b"begin").unwrap();
    spill.write(b"record").unwrap();
    assert_eq!(spill.sink().received, b"beginrecord");
    let client = spill.finish().unwrap();
    assert_eq!(
        (client.received.as_slice(), client.waited),
        (&b"beginrecord"[..], 0)
    );
}
