//! Unix socket framing and listener lifecycle. Application operations live
//! in `service`; each complete JSON request receives one JSON response.

use super::Request;
use super::service::{self, Daemon, error, handle};
use polylogue_engine::Config;
use polylogue_store::Archive;
use std::io::{BufRead, BufReader, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::Duration;

fn serve_client(stream: UnixStream, d: &Daemon) {
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let mut out = stream;
    for line in BufReader::new(read_half).lines() {
        let Ok(line) = line else {
            return;
        };
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(r) => handle(r, d),
            Err(e) => error("bad_request", e),
        };
        let Ok(mut text) = serde_json::to_string(&response) else {
            return;
        };
        text.push('\n');
        if out.write_all(text.as_bytes()).is_err() {
            return;
        }
    }
}

/// Runs the daemon until its listener stops. Each transport is dropped
/// before the application workers and archive lock.
pub fn serve() -> std::io::Result<()> {
    let root = Archive::default_root();
    #[cfg(feature = "sinex")]
    let backing = crate::sinex::configured();
    #[cfg(not(feature = "sinex"))]
    let backing = service::Backing::Standalone;
    let runtime = service::start(&root, Config::from_env(), backing)?;
    let server = Server::start(Arc::clone(&runtime.daemon))?;
    eprintln!(
        "polylogue: serving {} on {}",
        root.display(),
        runtime.daemon.archive.socket().display()
    );
    let _web = match crate::web::start(Arc::clone(&runtime.daemon)) {
        Ok(web) => {
            eprintln!("polylogue: web reader on http://127.0.0.1:{}/", web.port);
            Some(web)
        }
        Err(error) => {
            eprintln!("polylogue: web reader not started: {error}");
            None
        }
    };
    server.wait(|| runtime.stop_admission())
}

/// Owns socket admission and every accepted client thread.
pub struct Server {
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<std::io::Result<()>>>,
    socket: Option<PathBuf>,
    quiescing: Receiver<()>,
}

impl Server {
    pub fn start(daemon: Arc<Daemon>) -> std::io::Result<Self> {
        let listener = bind(&daemon)?;
        let (stop, stopped) = channel();
        let (quiesce, quiescing) = channel();
        let mut server = Self {
            stop: Some(stop),
            thread: None,
            socket: Some(daemon.archive.socket()),
            quiescing,
        };
        listener.set_nonblocking(true)?;
        server.thread = Some(
            std::thread::Builder::new()
                .name("polylogue-socket".into())
                .spawn(move || listen(listener, daemon, stopped, quiesce))?,
        );
        Ok(server)
    }

    pub fn shutdown(&mut self) -> std::io::Result<()> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        self.join()
    }

    fn join(&mut self) -> std::io::Result<()> {
        let result = self.thread.take().map_or(Ok(()), |thread| {
            thread
                .join()
                .map_err(|_| std::io::Error::other("socket listener panicked"))?
        });
        let cleanup =
            self.socket
                .take()
                .map_or(Ok(()), |socket| match std::fs::remove_file(socket) {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
                    result => result,
                });
        result.and(cleanup)
    }

    fn wait(mut self, before_join: impl FnOnce()) -> std::io::Result<()> {
        let _ = self.quiescing.recv();
        before_join();
        self.join()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: socket shutdown: {error}");
        }
    }
}

/// Binds the archive's socket. A socket file nobody answers on is left
/// from a stopped daemon; the archive lock held by `daemon` proves no
/// other daemon owns this archive.
pub fn bind(daemon: &Daemon) -> std::io::Result<UnixListener> {
    let socket = daemon.archive.socket();
    if socket.exists() {
        std::fs::remove_file(&socket)?;
    }
    UnixListener::bind(&socket)
}

fn listen(
    listener: UnixListener,
    daemon: Arc<Daemon>,
    stopped: Receiver<()>,
    quiesce: Sender<()>,
) -> std::io::Result<()> {
    let mut clients: Vec<(UnixStream, JoinHandle<()>)> = Vec::new();
    let mut failure = None;
    loop {
        for _ in 0..32 {
            match listener.accept() {
                Ok((stream, _)) => {
                    let control = match stream.try_clone() {
                        Ok(control) => control,
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    };
                    let daemon = Arc::clone(&daemon);
                    match std::thread::Builder::new()
                        .name("polylogue-client".into())
                        .spawn(move || serve_client(stream, &daemon))
                    {
                        Ok(thread) => clients.push((control, thread)),
                        Err(error) => {
                            failure = Some(error);
                            break;
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        let mut i = 0;
        while i < clients.len() {
            if clients[i].1.is_finished() {
                let (_, thread) = clients.swap_remove(i);
                if thread.join().is_err() {
                    failure.get_or_insert_with(|| std::io::Error::other("socket client panicked"));
                }
            } else {
                i += 1;
            }
        }
        if failure.is_some()
            || !matches!(
                stopped.recv_timeout(Duration::from_millis(20)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            )
        {
            break;
        }
    }
    drop(listener);
    let _ = quiesce.send(());
    // Closing peers makes incomplete input and blocked response writes fail
    // explicitly. A request already in handle finishes before its thread joins.
    for (stream, _) in &clients {
        let _ = stream.shutdown(Shutdown::Both);
    }
    for (_, thread) in clients {
        if thread.join().is_err() {
            failure.get_or_insert_with(|| std::io::Error::other("socket client panicked"));
        }
    }
    failure.map_or(Ok(()), Err)
}

#[cfg(test)]
mod tests;
