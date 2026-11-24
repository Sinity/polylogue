//! Relevance over several full-text files, equal to relevance over one.
//!
//! FTS5's own `bm25()` reads the statistics of the table it runs on: its row
//! count, its token count and how many of its rows hold each phrase. Over
//! several files those are each file's, so the same text would rank
//! differently depending on which file holds it and what lies beside it.
//! Here a search first reads every file's statistics through its statistics
//! row (key 0, `search.sql`) with `pl_search_stats`, sums them with
//! `pl_search_total`, and ranks each hit with `pl_bm25`, the same formula
//! and arithmetic as FTS5's `bm25()` with the sums in place of one table's
//! figures. A file where nothing matches still counts its rows, since the
//! statistics query matches its statistics row too. Entries and tokens come
//! from each file's `totals`, which count removed entries out; FTS5's own
//! totals for a contentless-delete table never do.
//!
//! The statistics travel as a blob of little-endian 64-bit integers: rows,
//! tokens, then the rows holding each phrase of the expression in order.
#![allow(unsafe_code)]

use crate::{Result, StoreError};
use rusqlite::Connection;
use rusqlite::ffi;
use rusqlite::functions::{Aggregate, Context, FunctionFlags};
use std::ffi::{CStr, c_int, c_void};

/// Registers the ranking functions on `conn`.
pub(super) fn register(conn: &Connection) -> Result<()> {
    // SAFETY: the handle belongs to `conn`, which is open for the whole
    // call; FTS5 keeps the functions with the connection.
    unsafe {
        let api = fts5_api(conn.handle())?;
        create(api, c"pl_bm25", bm25)?;
        create(api, c"pl_search_stats", stats)?;
    }
    conn.create_aggregate_function(
        "pl_search_total",
        1,
        FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
        Total,
    )?;
    Ok(())
}

/// The FTS5 API of the connection `db`.
unsafe fn fts5_api(db: *mut ffi::sqlite3) -> Result<*mut ffi::fts5_api> {
    let mut statement = std::ptr::null_mut();
    // SAFETY: `db` is an open connection; the statement is finalized before
    // returning, and the bound pointer outlives the step that fills it.
    unsafe {
        let code = ffi::sqlite3_prepare_v2(
            db,
            c"SELECT fts5(?1)".as_ptr(),
            -1,
            &mut statement,
            std::ptr::null_mut(),
        );
        if code != ffi::SQLITE_OK {
            return Err(StoreError::internal("FTS5 is not available"));
        }
        let mut api: *mut ffi::fts5_api = std::ptr::null_mut();
        ffi::sqlite3_bind_pointer(
            statement,
            1,
            (&mut api as *mut *mut ffi::fts5_api).cast(),
            c"fts5_api_ptr".as_ptr(),
            None,
        );
        ffi::sqlite3_step(statement);
        ffi::sqlite3_finalize(statement);
        if api.is_null() {
            return Err(StoreError::internal("FTS5 is not available"));
        }
        Ok(api)
    }
}

type Function = unsafe extern "C" fn(
    *const ffi::Fts5ExtensionApi,
    *mut ffi::Fts5Context,
    *mut ffi::sqlite3_context,
    c_int,
    *mut *mut ffi::sqlite3_value,
);

unsafe fn create(api: *mut ffi::fts5_api, name: &CStr, function: Function) -> Result<()> {
    // SAFETY: `api` is the connection's FTS5 API, valid while it is open.
    let code = unsafe {
        let create = (*api)
            .xCreateFunction
            .ok_or_else(|| StoreError::internal("FTS5 cannot add functions"))?;
        create(
            api,
            name.as_ptr(),
            std::ptr::null_mut(),
            Some(function),
            None,
        )
    };
    if code == ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(StoreError::internal("FTS5 refused a ranking function"))
    }
}

/// FTS5's `bm25()` constants.
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// What one statement's `pl_bm25` computes once: each phrase's IDF and the
/// average row length, from the summed statistics.
struct Prepared {
    idf: Vec<f64>,
    average: f64,
    frequency: Vec<f64>,
}

unsafe extern "C" fn drop_prepared(data: *mut c_void) {
    // SAFETY: the pointer was made by `Box::into_raw` in `bm25` and FTS5
    // frees it once.
    drop(unsafe { Box::from_raw(data.cast::<Prepared>()) });
}

/// The statistics blob as integers.
fn decode(bytes: &[u8]) -> Option<Vec<i64>> {
    if bytes.len() % 8 != 0 || bytes.len() < 16 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight bytes")))
            .collect(),
    )
}

unsafe fn fail(ctx: *mut ffi::sqlite3_context, message: &CStr) {
    // SAFETY: `ctx` is the context of the running call.
    unsafe { ffi::sqlite3_result_error(ctx, message.as_ptr(), -1) };
}

unsafe fn value_bytes<'a>(value: *mut ffi::sqlite3_value) -> &'a [u8] {
    // SAFETY: `value` is an argument of the running call; its bytes stay
    // valid until the call returns.
    unsafe {
        let data = ffi::sqlite3_value_blob(value);
        let length = ffi::sqlite3_value_bytes(value);
        if data.is_null() || length <= 0 {
            &[]
        } else {
            std::slice::from_raw_parts(data.cast::<u8>(), length as usize)
        }
    }
}

/// `pl_bm25(table, statistics)`: the row's relevance, lower first.
unsafe extern "C" fn bm25(
    api: *const ffi::Fts5ExtensionApi,
    fts: *mut ffi::Fts5Context,
    ctx: *mut ffi::sqlite3_context,
    count: c_int,
    values: *mut *mut ffi::sqlite3_value,
) {
    // SAFETY: FTS5 calls this with its API, the query's context and the
    // call's arguments, all valid for the duration of the call.
    unsafe {
        let api = &*api;
        let (Some(get), Some(set), Some(phrases), Some(inst_count), Some(inst), Some(column_size)) = (
            api.xGetAuxdata,
            api.xSetAuxdata,
            api.xPhraseCount,
            api.xInstCount,
            api.xInst,
            api.xColumnSize,
        ) else {
            return fail(ctx, c"FTS5 lacks a function pl_bm25 needs");
        };
        let mut prepared = get(fts, 0).cast::<Prepared>();
        if prepared.is_null() {
            if count != 1 {
                return fail(ctx, c"pl_bm25 takes the table and its statistics");
            }
            let Some(stats) = decode(value_bytes(*values)) else {
                return fail(ctx, c"pl_bm25 was given no statistics");
            };
            let phrase_count = phrases(fts).max(0) as usize;
            if stats.len() != phrase_count + 2 {
                return fail(ctx, c"pl_bm25 statistics are for another expression");
            }
            let (rows, tokens) = (stats[0], stats[1]);
            let idf = stats[2..]
                .iter()
                .map(|&hits| {
                    let idf = (((rows - hits) as f64 + 0.5) / (hits as f64 + 0.5)).ln();
                    if idf <= 0.0 { 1e-6 } else { idf }
                })
                .collect();
            let made = Box::into_raw(Box::new(Prepared {
                idf,
                average: tokens as f64 / rows as f64,
                frequency: vec![0.0; phrase_count],
            }));
            if set(fts, made.cast(), Some(drop_prepared)) != ffi::SQLITE_OK {
                return fail(ctx, c"pl_bm25 could not keep its statistics");
            }
            prepared = made;
        }
        let prepared = &mut *prepared;
        prepared.frequency.iter_mut().for_each(|f| *f = 0.0);
        let mut instances = 0;
        if inst_count(fts, &mut instances) != ffi::SQLITE_OK {
            return fail(ctx, c"pl_bm25 could not read the row's matches");
        }
        for i in 0..instances {
            let (mut phrase, mut column, mut offset) = (0, 0, 0);
            if inst(fts, i, &mut phrase, &mut column, &mut offset) != ffi::SQLITE_OK {
                return fail(ctx, c"pl_bm25 could not read the row's matches");
            }
            if let Some(frequency) = prepared.frequency.get_mut(phrase as usize) {
                *frequency += 1.0;
            }
        }
        let mut length = 0;
        if column_size(fts, -1, &mut length) != ffi::SQLITE_OK {
            return fail(ctx, c"pl_bm25 could not read the row's length");
        }
        let d = f64::from(length);
        let mut score = 0.0;
        for (idf, frequency) in prepared.idf.iter().zip(&prepared.frequency) {
            score += idf
                * ((frequency * (K1 + 1.0))
                    / (frequency + K1 * (1.0 - B + B * d / prepared.average)));
        }
        ffi::sqlite3_result_double(ctx, -1.0 * score);
    }
}

unsafe extern "C" fn count_row(
    _api: *const ffi::Fts5ExtensionApi,
    _fts: *mut ffi::Fts5Context,
    rows: *mut c_void,
) -> c_int {
    // SAFETY: `rows` is the counter `stats` passed to xQueryPhrase.
    unsafe { *rows.cast::<i64>() += 1 };
    ffi::SQLITE_OK
}

/// `pl_search_stats(table, entries, tokens)` on a file's statistics row,
/// matched by the search's expression `OR` the statistics row's character,
/// with the file's totals: the entries, tokens and entries holding each
/// phrase of the expression, leaving the statistics row out.
unsafe extern "C" fn stats(
    api: *const ffi::Fts5ExtensionApi,
    fts: *mut ffi::Fts5Context,
    ctx: *mut ffi::sqlite3_context,
    count: c_int,
    values: *mut *mut ffi::sqlite3_value,
) {
    // SAFETY: as for `bm25`.
    unsafe {
        let api = &*api;
        let (Some(rowid), Some(phrases), Some(query_phrase), Some(inst_count), Some(inst)) = (
            api.xRowid,
            api.xPhraseCount,
            api.xQueryPhrase,
            api.xInstCount,
            api.xInst,
        ) else {
            return fail(ctx, c"FTS5 lacks a function pl_search_stats needs");
        };
        if rowid(fts) != 0 {
            return fail(ctx, c"pl_search_stats runs on the statistics row");
        }
        // The expression's phrases, then the statistics row's own.
        let Some(phrase_count) = usize::try_from(phrases(fts) - 1).ok() else {
            return fail(ctx, c"pl_search_stats needs the statistics phrase");
        };
        if count != 2 {
            return fail(
                ctx,
                c"pl_search_stats takes the table and the file's totals",
            );
        }
        let mut out = Vec::with_capacity(phrase_count + 2);
        out.push(ffi::sqlite3_value_int64(*values));
        out.push(ffi::sqlite3_value_int64(*values.add(1)));
        for phrase in 0..phrase_count {
            let mut hits = 0i64;
            if query_phrase(
                fts,
                phrase as c_int,
                (&mut hits as *mut i64).cast(),
                Some(count_row),
            ) != ffi::SQLITE_OK
            {
                return fail(ctx, c"pl_search_stats could not count a phrase");
            }
            out.push(hits);
        }
        // A phrase the statistics row itself holds counted that row.
        let mut instances = 0;
        if inst_count(fts, &mut instances) != ffi::SQLITE_OK {
            return fail(ctx, c"pl_search_stats could not read the row's matches");
        }
        let mut counted = vec![false; phrase_count];
        for i in 0..instances {
            let (mut phrase, mut column, mut offset) = (0, 0, 0);
            if inst(fts, i, &mut phrase, &mut column, &mut offset) != ffi::SQLITE_OK {
                return fail(ctx, c"pl_search_stats could not read the row's matches");
            }
            if let Some(seen) = counted.get_mut(phrase as usize)
                && !*seen
            {
                *seen = true;
                out[phrase as usize + 2] -= 1;
            }
        }
        let bytes: Vec<u8> = out.iter().flat_map(|n| n.to_le_bytes()).collect();
        ffi::sqlite3_result_blob(
            ctx,
            bytes.as_ptr().cast(),
            bytes.len() as c_int,
            ffi::SQLITE_TRANSIENT(),
        );
    }
}

/// `pl_search_total(statistics)`: the files' statistics summed.
struct Total;

impl Aggregate<Option<Vec<i64>>, Option<Vec<u8>>> for Total {
    fn init(&self, _: &mut Context<'_>) -> rusqlite::Result<Option<Vec<i64>>> {
        Ok(None)
    }

    fn step(&self, ctx: &mut Context<'_>, sum: &mut Option<Vec<i64>>) -> rusqlite::Result<()> {
        let bytes: Vec<u8> = ctx.get(0)?;
        let file = decode(&bytes).ok_or_else(|| {
            rusqlite::Error::UserFunctionError("pl_search_total was given no statistics".into())
        })?;
        match sum {
            None => *sum = Some(file),
            Some(sum) if sum.len() == file.len() => {
                sum.iter_mut().zip(file).for_each(|(sum, n)| *sum += n);
            }
            Some(_) => {
                return Err(rusqlite::Error::UserFunctionError(
                    "the files' statistics are for different expressions".into(),
                ));
            }
        }
        Ok(())
    }

    fn finalize(
        &self,
        _: &mut Context<'_>,
        sum: Option<Option<Vec<i64>>>,
    ) -> rusqlite::Result<Option<Vec<u8>>> {
        Ok(sum
            .flatten()
            .map(|sum| sum.iter().flat_map(|n| n.to_le_bytes()).collect()))
    }
}
