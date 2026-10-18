//! A byte buffer writer that refuses to grow past a limit. Encoders write into
//! it so an oversized value fails before it is allocated.
use std::io::{self, Write};

pub(crate) enum Overflow {
    /// Writing would need `required` bytes in total, more than the limit.
    Limit {
        required: usize,
    },
    Alloc,
}

pub(crate) struct LimitedVec {
    bytes: Vec<u8>,
    limit: usize,
    overflow: Option<Overflow>,
}

impl Write for LimitedVec {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.overflow.is_some() {
            return Err(io::Error::other("limited buffer already failed"));
        }
        let required = self.bytes.len().saturating_add(bytes.len());
        if required > self.limit {
            self.overflow = Some(Overflow::Limit { required });
            return Err(io::Error::other("limited buffer is full"));
        }
        if self.bytes.try_reserve_exact(bytes.len()).is_err() {
            self.overflow = Some(Overflow::Alloc);
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "limited buffer allocation failed",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Runs `f` against a buffer of at most `limit` bytes. An overflow takes
/// precedence over the encoder error it caused.
pub(crate) fn encode<E>(
    limit: usize,
    f: impl FnOnce(&mut LimitedVec) -> Result<(), E>,
) -> Result<Vec<u8>, Failure<E>> {
    let mut writer = LimitedVec {
        bytes: Vec::new(),
        limit,
        overflow: None,
    };
    let result = f(&mut writer);
    match writer.overflow {
        Some(Overflow::Limit { required }) => Err(Failure::Limit { required }),
        Some(Overflow::Alloc) => Err(Failure::Alloc),
        None => match result {
            Ok(()) => Ok(writer.bytes),
            Err(e) => Err(Failure::Encode(e)),
        },
    }
}

pub(crate) enum Failure<E> {
    Limit { required: usize },
    Alloc,
    Encode(E),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_reports_the_total_size_needed_and_wins_over_the_encoder_error() {
        let fits = encode(4, |w| w.write_all(b"abcd")).ok().unwrap();
        assert_eq!(fits, b"abcd");
        match encode(4, |w| {
            w.write_all(b"abc")?;
            w.write_all(b"de")
        }) {
            Err(Failure::Limit { required }) => assert_eq!(required, 5),
            _ => panic!("expected a limit failure"),
        }
    }
}
