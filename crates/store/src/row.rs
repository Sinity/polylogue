//! Rows as typed structs. [`row!`] declares a row: plain data whose one field
//! list gives the column names, the reading of a result row by position, the
//! binding of an insert, and a compact byte form for sorted files. The
//! statements that read or insert a row are declared in the same [`row!`],
//! and the every-statement test prepares each one against a fresh index and
//! archive and compares its columns with the fields.
//!
//! A field's type says how its column is stored ([`Column`]): an integer,
//! `bool`, `f64`, `String`, a BLOB as `Vec<u8>` or `[u8; 32]`, a key
//! ([`SessionPk`] and the others), a stored ID, a `stored_names!` enum (each
//! listed once in `name_columns!` below), a [`Timestamp`] as milliseconds, a
//! JSON text as [`Json<T>`], and a nullable column as `Option<T>` of any of
//! these.

use crate::{Result, StoreError};
use polylogue_model::{StoredName, Timestamp};
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ToSql, ToSqlOutput, Value, ValueRef};
use std::borrow::Cow;
use std::marker::PhantomData;
use std::ops::ControlFlow;

/// Declares a row struct and the statements that read or insert it:
///
/// ```ignore
/// row! {
///     /// One stored link between two sessions.
///     pub struct SessionLink {
///         pub parent: String,
///         pub branch_point: Option<String>,
///         pub copies_prefix: bool,
///     }
///     statements {
///         pub(crate) const OF_CHILD = "SELECT parent_id AS parent, branch_point, copies_prefix
///             FROM session_link WHERE child_id = ?1";
///     }
/// }
/// ```
///
/// The struct derives `Debug`, `Clone` and `PartialEq`; further derives go
/// in its attributes. A reading statement names its columns as the fields,
/// in field order (`AS` where a column or expression is named otherwise); an
/// inserting statement is `INSERT ... INTO table(fields...) ...` with one
/// parameter per field, bound in field order. Each row type is listed in the
/// every-statement test (`row/tests.rs`).
macro_rules! row {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($(#[$field_meta:meta])* $field_vis:vis $field:ident : $type:ty),+ $(,)?
        }
        $(statements {
            $($(#[$statement_meta:meta])* $statement_vis:vis const $statement:ident = $sql:expr;)*
        })?
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq)]
        $vis struct $name {
            $($(#[$field_meta])* $field_vis $field: $type,)+
        }

        impl $name {
            $($(
                $(#[$statement_meta])*
                $statement_vis const $statement: $crate::row::Statement<Self> =
                    $crate::row::Statement::declared(
                        concat!(stringify!($name), "::", stringify!($statement)),
                        $sql,
                    );
            )*)?
        }

        impl $crate::row::Row for $name {
            const FIELDS: &'static [&'static str] = &[$(stringify!($field)),+];
            const STATEMENTS: &'static [$crate::row::Statement<Self>] = &[$($(Self::$statement),*)?];

            fn from_row(row: &::rusqlite::Row<'_>) -> ::rusqlite::Result<Self> {
                let mut at = $crate::row::Place::default();
                Ok(Self {
                    $($field: $crate::row::read::<$type>(row, at.next(), stringify!($field))?,)+
                })
            }

            fn bind(&self, statement: &mut ::rusqlite::Statement<'_>) -> ::rusqlite::Result<()> {
                let mut at = $crate::row::Place::default();
                $($crate::row::bind(statement, at.next() + 1, &self.$field)?;)+
                Ok(())
            }

            fn to_bytes(&self, out: &mut Vec<u8>) {
                $($crate::row::Column::put(&self.$field, out);)+
            }

            fn from_bytes(input: &mut &[u8]) -> $crate::Result<Self> {
                Ok(Self {
                    $($field: $crate::row::take::<$type>(input, stringify!($name), stringify!($field))?,)+
                })
            }
        }
    };
}
pub(crate) use row;

/// A struct declared with [`row!`].
pub trait Row: Sized + 'static {
    /// The field names, which are the column names, in order.
    const FIELDS: &'static [&'static str];
    /// The statements declared with the row, for the every-statement test.
    const STATEMENTS: &'static [Statement<Self>];
    /// Reads a result row whose columns are the fields in order.
    fn from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Self>;
    /// Binds the fields to parameters `1..=FIELDS.len()`.
    fn bind(&self, statement: &mut rusqlite::Statement<'_>) -> rusqlite::Result<()>;
    /// Appends the row's byte form: each field in order, self-delimiting.
    fn to_bytes(&self, out: &mut Vec<u8>);
    /// Reads one row's byte form from the front of `input` and advances it.
    fn from_bytes(input: &mut &[u8]) -> Result<Self>;
}

/// A statement that reads or inserts rows of `R`. Declared statements are
/// constants made by [`row!`]; a statement whose text is built at run time
/// (a column list or filter chosen per call) is made with
/// [`Statement::built`] and has its columns checked each time it runs in a
/// debug build.
pub struct Statement<R> {
    name: &'static str,
    sql: Cow<'static, str>,
    row: PhantomData<fn() -> R>,
}

impl<R> Statement<R> {
    #[doc(hidden)]
    pub const fn declared(name: &'static str, sql: &'static str) -> Self {
        Self {
            name,
            sql: Cow::Borrowed(sql),
            row: PhantomData,
        }
    }

    pub fn built(sql: impl Into<Cow<'static, str>>) -> Self {
        Self {
            name: "a built statement",
            sql: sql.into(),
            row: PhantomData,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn sql(&self) -> &str {
        &self.sql
    }
}

/// How a field's column is read, bound and written in the byte form.
pub trait Column: Sized {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self>;
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>>;
    fn put(&self, out: &mut Vec<u8>);
    fn take(input: &mut &[u8]) -> Option<Self>;
}

/// The row helpers on a connection.
pub(crate) trait Rows {
    /// Every row the statement reads.
    fn all<R: Row>(
        &self,
        statement: &Statement<R>,
        params: impl rusqlite::Params,
    ) -> Result<Vec<R>>;
    /// The first row the statement reads, if any.
    fn maybe<R: Row>(
        &self,
        statement: &Statement<R>,
        params: impl rusqlite::Params,
    ) -> Result<Option<R>>;
    /// The first row the statement reads; none is an error.
    fn one<R: Row>(&self, statement: &Statement<R>, params: impl rusqlite::Params) -> Result<R>;
    /// Visits the rows in order until `visit` breaks, reading no further.
    fn each<R: Row>(
        &self,
        statement: &Statement<R>,
        params: impl rusqlite::Params,
        visit: impl FnMut(R) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>>;
    /// Inserts one row through an inserting statement.
    fn insert<R: Row>(&self, statement: &Statement<R>, row: &R) -> Result<()>;
    /// The one column of the first row a statement run repeatedly reads, if
    /// any. One-off text reads through `ConnExt::scalar`, outside the
    /// statement cache.
    fn value<T: Column>(&self, sql: &str, params: impl rusqlite::Params) -> Result<Option<T>>;
    /// The one column of every row a statement run repeatedly reads.
    fn values<T: Column>(&self, sql: &str, params: impl rusqlite::Params) -> Result<Vec<T>>;
}

impl Rows for rusqlite::Connection {
    fn all<R: Row>(
        &self,
        statement: &Statement<R>,
        params: impl rusqlite::Params,
    ) -> Result<Vec<R>> {
        let mut out = Vec::new();
        let _ = self.each(statement, params, |row| {
            out.push(row);
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(out)
    }

    fn maybe<R: Row>(
        &self,
        statement: &Statement<R>,
        params: impl rusqlite::Params,
    ) -> Result<Option<R>> {
        let mut first = None;
        let _ = self.each(statement, params, |row| {
            first = Some(row);
            Ok(ControlFlow::Break(()))
        })?;
        Ok(first)
    }

    fn one<R: Row>(&self, statement: &Statement<R>, params: impl rusqlite::Params) -> Result<R> {
        self.maybe(statement, params)?
            .ok_or_else(|| rusqlite::Error::QueryReturnedNoRows.into())
    }

    fn each<R: Row>(
        &self,
        statement: &Statement<R>,
        params: impl rusqlite::Params,
        mut visit: impl FnMut(R) -> Result<ControlFlow<()>>,
    ) -> Result<ControlFlow<()>> {
        let mut prepared = self.prepare_cached(statement.sql())?;
        if cfg!(debug_assertions) {
            check_reads::<R>(statement.name, &prepared).map_err(StoreError::internal)?;
        }
        let mut rows = prepared.query(params)?;
        while let Some(row) = rows.next()? {
            if visit(R::from_row(row)?)?.is_break() {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }

    fn insert<R: Row>(&self, statement: &Statement<R>, row: &R) -> Result<()> {
        let mut prepared = self.prepare_cached(statement.sql())?;
        if cfg!(debug_assertions) {
            check_inserts::<R>(statement.name, statement.sql(), &prepared)
                .map_err(StoreError::internal)?;
        }
        row.bind(&mut prepared)?;
        prepared.raw_execute()?;
        Ok(())
    }

    fn value<T: Column>(&self, sql: &str, params: impl rusqlite::Params) -> Result<Option<T>> {
        let mut prepared = self.prepare_cached(sql)?;
        let mut rows = prepared.query(params)?;
        Ok(match rows.next()? {
            Some(row) => Some(read(row, 0, "value")?),
            None => None,
        })
    }

    fn values<T: Column>(&self, sql: &str, params: impl rusqlite::Params) -> Result<Vec<T>> {
        let mut prepared = self.prepare_cached(sql)?;
        let mut rows = prepared.query(params)?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(read(row, 0, "value")?);
        }
        Ok(out)
    }
}

/// Whether a reading statement's columns are `R`'s fields in order.
pub(crate) fn check_reads<R: Row>(
    name: &str,
    prepared: &rusqlite::Statement<'_>,
) -> std::result::Result<(), String> {
    let columns = prepared.column_names();
    if columns != R::FIELDS {
        return Err(format!(
            "{name} reads columns {columns:?}, but its row has fields {:?}",
            R::FIELDS
        ));
    }
    Ok(())
}

/// Whether an inserting statement names `R`'s fields in order, in the
/// column list after its table, and takes one parameter for each.
pub(crate) fn check_inserts<R: Row>(
    name: &str,
    sql: &str,
    prepared: &rusqlite::Statement<'_>,
) -> std::result::Result<(), String> {
    if prepared.column_count() != 0 {
        return Err(format!(
            "{name} returns columns; an insert binds its row and returns none"
        ));
    }
    let columns = inserted_columns(sql)
        .ok_or_else(|| format!("{name} is not INSERT ... INTO table(columns) ...: {sql}"))?;
    if columns != R::FIELDS {
        return Err(format!(
            "{name} inserts columns {columns:?}, but its row has fields {:?}",
            R::FIELDS
        ));
    }
    if prepared.parameter_count() != R::FIELDS.len() {
        return Err(format!(
            "{name} takes {} parameters for {} fields",
            prepared.parameter_count(),
            R::FIELDS.len()
        ));
    }
    Ok(())
}

/// The every-statement test's walk over declared statements: each is
/// prepared on one connection and its columns compared with its row's
/// fields.
#[cfg(test)]
pub(crate) struct Check<'c> {
    pub(crate) conn: &'c rusqlite::Connection,
    pub(crate) checked: usize,
    pub(crate) failures: Vec<String>,
}

#[cfg(test)]
impl Check<'_> {
    pub(crate) fn row<R: Row>(&mut self) {
        for statement in R::STATEMENTS {
            self.checked += 1;
            let checked = match self.conn.prepare(statement.sql()) {
                Err(error) => Err(format!("{} does not prepare: {error}", statement.name())),
                Ok(prepared) if prepared.column_count() > 0 => {
                    check_reads::<R>(statement.name(), &prepared)
                }
                Ok(prepared) => check_inserts::<R>(statement.name(), statement.sql(), &prepared),
            };
            if let Err(failure) = checked {
                self.failures.push(failure);
            }
        }
    }
}

/// The column list of `INSERT [OR ...] INTO table(a, b, ...)`.
fn inserted_columns(sql: &str) -> Option<Vec<&str>> {
    let head = sql.trim_start();
    if !head.get(..6)?.eq_ignore_ascii_case("insert") {
        return None;
    }
    let upper = head.to_ascii_uppercase();
    let into = upper.find(" INTO ")? + " INTO ".len();
    let open = into + head[into..].find('(')?;
    let close = open + head[open..].find(')')?;
    Some(head[open + 1..close].split(',').map(str::trim).collect())
}

/// The next column of a row being read or bound.
#[doc(hidden)]
#[derive(Default)]
pub struct Place(usize);

impl Place {
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> usize {
        self.0 += 1;
        self.0 - 1
    }
}

/// A field's value that does not decode, named by its field.
#[derive(Debug, thiserror::Error)]
#[error("field {field}: {error}")]
struct FieldError {
    field: &'static str,
    error: FromSqlError,
}

#[doc(hidden)]
pub fn read<T: Column>(
    row: &rusqlite::Row<'_>,
    index: usize,
    field: &'static str,
) -> rusqlite::Result<T> {
    let value = row.get_ref(index)?;
    T::from_sql(value).map_err(|error| match error {
        FromSqlError::InvalidType => {
            rusqlite::Error::InvalidColumnType(index, field.to_owned(), value.data_type())
        }
        FromSqlError::OutOfRange(n) => rusqlite::Error::IntegralValueOutOfRange(index, n),
        error => rusqlite::Error::FromSqlConversionFailure(
            index,
            value.data_type(),
            Box::new(FieldError { field, error }),
        ),
    })
}

struct Bound<'a, T>(&'a T);

impl<T: Column> ToSql for Bound<'_, T> {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        self.0.to_sql()
    }
}

#[doc(hidden)]
pub fn bind<T: Column>(
    statement: &mut rusqlite::Statement<'_>,
    index: usize,
    value: &T,
) -> rusqlite::Result<()> {
    statement.raw_bind_parameter(index, Bound(value))
}

#[doc(hidden)]
pub fn take<T: Column>(input: &mut &[u8], row: &str, field: &str) -> Result<T> {
    T::take(input).ok_or_else(|| {
        StoreError::stored_data(format!("a sorted {row} row has an unreadable {field}"))
    })
}

fn other(error: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> FromSqlError {
    FromSqlError::Other(error.into())
}

// The byte form: integers as LEB128 varints (signed ones zigzagged), lengths
// before byte strings, one tag byte before an optional value.

fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn take_varint(input: &mut &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for (i, byte) in input.iter().enumerate().take(10) {
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            *input = &input[i + 1..];
            return Some(value);
        }
    }
    None
}

fn put_signed(value: i64, out: &mut Vec<u8>) {
    put_varint(((value << 1) ^ (value >> 63)) as u64, out);
}

fn take_signed(input: &mut &[u8]) -> Option<i64> {
    let value = take_varint(input)?;
    Some((value >> 1) as i64 ^ -((value & 1) as i64))
}

fn put_bytes(bytes: &[u8], out: &mut Vec<u8>) {
    put_varint(bytes.len() as u64, out);
    out.extend_from_slice(bytes);
}

fn take_bytes<'a>(input: &mut &'a [u8]) -> Option<&'a [u8]> {
    let length = usize::try_from(take_varint(input)?).ok()?;
    let (bytes, rest) = input.split_at_checked(length)?;
    *input = rest;
    Some(bytes)
}

fn take_str<'a>(input: &mut &'a [u8]) -> Option<&'a str> {
    std::str::from_utf8(take_bytes(input)?).ok()
}

macro_rules! integer_columns {
    ($put:ident, $take:ident: $($type:ty => $wide:expr),+) => {$(
        impl Column for $type {
            fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
                <$type as FromSql>::column_result(value)
            }
            fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
                ToSql::to_sql(self)
            }
            fn put(&self, out: &mut Vec<u8>) {
                $put($wide(*self), out);
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                $take(input)?.try_into().ok()
            }
        }
    )+};
}
integer_columns!(put_signed, take_signed: i64 => std::convert::identity, i32 => i64::from);
integer_columns!(
    put_varint, take_varint: u64 => std::convert::identity, u32 => u64::from, usize => usize_wide
);

fn usize_wide(value: usize) -> u64 {
    value as u64
}

impl Column for bool {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
        bool::column_result(value)
    }
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        ToSql::to_sql(self)
    }
    fn put(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let (&byte, rest) = input.split_first()?;
        *input = rest;
        match byte {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }
}

impl Column for f64 {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
        f64::column_result(value)
    }
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        ToSql::to_sql(self)
    }
    fn put(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let (bytes, rest) = input.split_first_chunk::<8>()?;
        *input = rest;
        Some(f64::from_le_bytes(*bytes))
    }
}

impl Column for String {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
        String::column_result(value)
    }
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        ToSql::to_sql(self)
    }
    fn put(&self, out: &mut Vec<u8>) {
        put_bytes(self.as_bytes(), out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        take_str(input).map(str::to_owned)
    }
}

impl Column for Vec<u8> {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
        Vec::column_result(value)
    }
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        ToSql::to_sql(self)
    }
    fn put(&self, out: &mut Vec<u8>) {
        put_bytes(self, out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        take_bytes(input).map(<[u8]>::to_vec)
    }
}

/// A hash: a BLOB of exactly 32 bytes.
impl Column for [u8; 32] {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let bytes = value.as_blob()?;
        bytes
            .try_into()
            .map_err(|_| other(format!("a stored hash has {} bytes, not 32", bytes.len())))
    }
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Borrowed(ValueRef::Blob(self)))
    }
    fn put(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(self);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let (bytes, rest) = input.split_first_chunk::<32>()?;
        *input = rest;
        Some(*bytes)
    }
}

impl<T: Column> Column for Option<T> {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
        match value {
            ValueRef::Null => Ok(None),
            value => T::from_sql(value).map(Some),
        }
    }
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        match self {
            None => Ok(ToSqlOutput::Owned(Value::Null)),
            Some(value) => value.to_sql(),
        }
    }
    fn put(&self, out: &mut Vec<u8>) {
        match self {
            None => out.push(0),
            Some(value) => {
                out.push(1);
                value.put(out);
            }
        }
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        let (&tag, rest) = input.split_first()?;
        *input = rest;
        match tag {
            0 => Some(None),
            1 => T::take(input).map(Some),
            _ => None,
        }
    }
}

/// An instant stored as Unix milliseconds.
impl Column for Timestamp {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let ms = value.as_i64()?;
        Timestamp::from_unix_ms(ms)
            .ok_or_else(|| other(format!("stored time {ms} ms is out of range")))
    }
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Owned(Value::Integer(self.unix_ms())))
    }
    fn put(&self, out: &mut Vec<u8>) {
        put_signed(self.unix_ms(), out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        Timestamp::from_unix_ms(take_signed(input)?)
    }
}

/// A column holding JSON, read as `T` from text or a BLOB and bound as text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Json<T>(pub T);

impl<T: serde::Serialize + serde::de::DeserializeOwned> Column for Json<T> {
    fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
        serde_json::from_slice(value.as_bytes()?)
            .map(Json)
            .map_err(other)
    }
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        serde_json::to_string(&self.0)
            .map(|text| ToSqlOutput::Owned(Value::Text(text)))
            .map_err(|error| rusqlite::Error::ToSqlConversionFailure(Box::new(error)))
    }
    fn put(&self, out: &mut Vec<u8>) {
        let text = serde_json::to_vec(&self.0).expect("a JSON column's value serializes");
        put_bytes(&text, out);
    }
    fn take(input: &mut &[u8]) -> Option<Self> {
        serde_json::from_slice(take_bytes(input)?).ok().map(Json)
    }
}

/// Identities stored as their text form.
macro_rules! text_columns {
    ($($type:ty),+) => {$(
        impl Column for $type {
            fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
                let text = value.as_str()?;
                <$type>::parse(text)
                    .map_err(|error| other(format!("stored identity {text:?} does not parse: {error}")))
            }
            fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
                Ok(ToSqlOutput::Owned(Value::Text(self.to_string())))
            }
            fn put(&self, out: &mut Vec<u8>) {
                put_bytes(self.to_string().as_bytes(), out);
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                <$type>::parse(take_str(input)?).ok()
            }
        }
    )+};
}
text_columns!(
    polylogue_model::SessionId,
    polylogue_model::MessageId,
    polylogue_model::DocumentId
);

/// `stored_names!` enums, stored by name. An enum a row stores is listed
/// here once.
macro_rules! name_columns {
    ($($type:ty),+) => {$(
        impl Column for $type {
            fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
                let name = value.as_str()?;
                <$type as StoredName>::from_name(name)
                    .ok_or_else(|| other(format!("unknown stored name {name:?}")))
            }
            fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
                Ok(ToSqlOutput::Borrowed(ValueRef::Text(StoredName::as_str(*self).as_bytes())))
            }
            fn put(&self, out: &mut Vec<u8>) {
                put_bytes(StoredName::as_str(*self).as_bytes(), out);
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                <$type as StoredName>::from_name(take_str(input)?)
            }
        }
    )+};
}
name_columns!(
    polylogue_model::Role,
    polylogue_model::PromptKind,
    polylogue_model::CallOutcome,
    polylogue_model::Route
);

/// Store keys: the integer primary key of a row, typed by its table, bound
/// and read as an SQLite integer.
macro_rules! keys {
    ($($(#[$meta:meta])* $name:ident;)+) => {$(
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub i64);

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }
        impl ToSql for $name {
            fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
                Ok(ToSqlOutput::Owned(Value::Integer(self.0)))
            }
        }
        impl FromSql for $name {
            fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
                value.as_i64().map($name)
            }
        }
        impl Column for $name {
            fn from_sql(value: ValueRef<'_>) -> FromSqlResult<Self> {
                value.as_i64().map($name)
            }
            fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
                ToSql::to_sql(self)
            }
            fn put(&self, out: &mut Vec<u8>) {
                put_signed(self.0, out);
            }
            fn take(input: &mut &[u8]) -> Option<Self> {
                take_signed(input).map($name)
            }
        }
    )+};
}
keys! {
    /// `session.pk`.
    SessionPk;
    /// `message.pk`.
    MessagePk;
    /// `block.pk`.
    BlockPk;
    /// `event.pk`.
    EventPk;
    /// `input_ref.pk`.
    InputPk;
    /// `text.id`.
    TextKey;
    /// `content_group.id`.
    GroupKey;
}

#[cfg(test)]
mod tests;
