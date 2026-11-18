//! Daemon messages carry the priorities understood by the service journal.

macro_rules! log {
    (error, $message:literal $(, $argument:expr)* $(,)?) => {
        eprintln!("<3>polylogue: {}", format_args!($message $(, $argument)*))
    };
    (warning, $message:literal $(, $argument:expr)* $(,)?) => {
        eprintln!("<4>polylogue: {}", format_args!($message $(, $argument)*))
    };
    (info, $message:literal $(, $argument:expr)* $(,)?) => {
        eprintln!("<6>polylogue: {}", format_args!($message $(, $argument)*))
    };
}

pub(crate) use log;
