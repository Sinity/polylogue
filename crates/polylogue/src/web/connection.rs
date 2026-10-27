//! A connection carries its disconnect signal into each HTTP request.
use crate::daemon::disconnect;
use axum::extract::connect_info::Connected;
use polylogue_store::read::limit::ReadCancellation;
use std::io;
use std::os::fd::AsFd;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

#[derive(Clone, Debug)]
pub(super) struct Peer(pub ReadCancellation);

impl Connected<axum::serve::IncomingStream<'_, super::OwnerListener>> for Peer {
    fn connect_info(stream: axum::serve::IncomingStream<'_, super::OwnerListener>) -> Self {
        stream.remote_addr().clone()
    }
}

pub(super) struct Connection {
    stream: tokio::net::TcpStream,
    peer: Peer,
    watch: tokio::task::JoinHandle<()>,
}

impl Connection {
    pub(super) fn new(stream: tokio::net::TcpStream) -> io::Result<(Self, Peer)> {
        let fd = stream.as_fd().try_clone_to_owned()?;
        let peer = Peer(ReadCancellation::default());
        let cancellation = peer.0.clone();
        let watch = tokio::spawn(async move {
            let mut tick = tokio::time::interval(disconnect::INTERVAL);
            loop {
                tick.tick().await;
                if disconnect::closed(&fd) {
                    cancellation.cancel();
                    break;
                }
            }
        });
        Ok((
            Self {
                stream,
                peer: peer.clone(),
                watch,
            },
            peer,
        ))
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.peer.0.cancel();
        self.watch.abort();
    }
}

impl AsyncRead for Connection {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(cx, buffer)
    }
}

impl AsyncWrite for Connection {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, bytes)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}
