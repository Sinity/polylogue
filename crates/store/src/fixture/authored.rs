//! Explicit synthetic actors and clocks for authored request fixtures.
use polylogue_model::{
    Author, Timestamp,
    authorship::{Authorship, Channel},
    user::{UserRequest, UserWrite},
};

pub fn request(id: String, write: UserWrite, authorship: Authorship) -> UserRequest {
    UserRequest {
        id,
        machine: "synthetic-machine".into(),
        authorship,
        at: Timestamp::from_unix_ms(1).unwrap(),
        write,
    }
}
pub fn operator_request(id: String, write: UserWrite) -> UserRequest {
    request(
        id,
        write,
        Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        },
    )
}
