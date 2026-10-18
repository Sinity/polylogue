//! Unsafe calls into SQLite, libc and the bundled C code. Every handle passed
//! stays valid for the whole synchronous call and no pointer escapes.
#![allow(unsafe_code)]
use rusqlite::Connection;
use std::{fs::File, io, os::fd::AsRawFd, path::Path};

unsafe extern "C" {
    fn sqlite3_register_cksumvfs(unused: *const std::ffi::c_char) -> i32;
    fn polylogue_reserve_bytes(db: *mut rusqlite::ffi::sqlite3) -> i32;
}

pub(crate) fn register() -> i32 {
    // The argument is ignored; the VFS lives for the whole process.
    unsafe { sqlite3_register_cksumvfs(std::ptr::null()) }
}
pub(crate) fn reserve(conn: &Connection) -> i32 {
    // The borrowed connection cannot close or be used concurrently here.
    unsafe { polylogue_reserve_bytes(conn.handle()) }
}

/// Drops an idle connection's page cache. Unlike `shrink_memory`, this leaves
/// other connections' pages alone.
pub(crate) fn reset_scratch_cache(conn: &Connection) -> rusqlite::Result<()> {
    if !conn.is_autocommit() {
        return Err(rusqlite::Error::InvalidQuery);
    }
    // "main" is a static C string and this opcode ignores the last pointer.
    // Callers finalize their statements first.
    let code = unsafe {
        rusqlite::ffi::sqlite3_file_control(
            conn.handle(),
            c"main".as_ptr(),
            rusqlite::ffi::SQLITE_FCNTL_RESET_CACHE,
            std::ptr::null_mut(),
        )
    };
    if code == rusqlite::ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(code),
            None,
        ))
    }
}

/// Page-cache bytes used by this connection.
#[cfg(test)]
pub(crate) fn cache_bytes(conn: &Connection) -> rusqlite::Result<i32> {
    let mut current = 0;
    let mut unused_highwater = 0;
    let code = unsafe {
        rusqlite::ffi::sqlite3_db_status(
            conn.handle(),
            rusqlite::ffi::SQLITE_DBSTATUS_CACHE_USED,
            &mut current,
            &mut unused_highwater,
            0,
        )
    };
    if code == rusqlite::ffi::SQLITE_OK {
        Ok(current)
    } else {
        Err(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(code),
            None,
        ))
    }
}

/// Renames `source` to `destination`, failing if the destination exists.
pub(crate) fn publish(source: &Path, destination: &Path) -> io::Result<()> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let source = CString::new(source.as_os_str().as_bytes())?;
    let destination = CString::new(destination.as_os_str().as_bytes())?;
    // RENAME_NOREPLACE makes the existence check atomic with the rename.
    if unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    } == 0
    {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Removes the progress handler on drop, while the borrowed flag is still alive.
pub(crate) struct CancelGuard<'a>(&'a Connection);
impl Drop for CancelGuard<'_> {
    fn drop(&mut self) {
        unsafe {
            rusqlite::ffi::sqlite3_progress_handler(self.0.handle(), 0, None, std::ptr::null_mut());
        }
    }
}
/// Makes SQLite interrupt the running statement once `cancelled` is set.
pub(crate) fn cancellation<'a>(
    conn: &'a Connection,
    cancelled: &'a std::sync::atomic::AtomicBool,
) -> CancelGuard<'a> {
    unsafe extern "C" fn check(value: *mut std::ffi::c_void) -> i32 {
        // The guard keeps the flag borrowed while the handler is installed.
        let cancelled = unsafe { &*value.cast::<std::sync::atomic::AtomicBool>() };
        i32::from(cancelled.load(std::sync::atomic::Ordering::Acquire))
    }
    unsafe {
        rusqlite::ffi::sqlite3_progress_handler(
            conn.handle(),
            1000,
            Some(check),
            std::ptr::from_ref(cancelled).cast_mut().cast(),
        );
    }
    CancelGuard(conn)
}

/// Sets the btrfs no-copy-on-write flag on a directory, so files created in it
/// later inherit it. Does nothing on other filesystems.
pub(crate) fn no_cow(path: &Path) -> io::Result<()> {
    let file = File::open(path)?;
    let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
    // fstatfs writes exactly the initialized statfs object and borrows a live descriptor.
    if unsafe { libc::fstatfs(file.as_raw_fd(), &mut fs) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if fs.f_type != 0x9123683e {
        return Ok(());
    } // BTRFS_SUPER_MAGIC
    let mut flags: libc::c_long = 0;
    // Linux FS_IOC_GETFLAGS/SETFLAGS expect a pointer to a long. Directory stays open.
    if unsafe { libc::ioctl(file.as_raw_fd(), 0x80086601_u64, &mut flags) } != 0 {
        return Err(io::Error::last_os_error());
    }
    flags |= 0x00800000; // FS_NOCOW_FL, before any child exists.
    if unsafe { libc::ioctl(file.as_raw_fd(), 0x40086602_u64, &flags) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

// The checksum VFS reports a bad page through this callback on the thread that
// made the SQLite call. Callers clear it before each call and take it right
// after, so an unrelated error never picks up an old location.
#[derive(Debug)]
pub(crate) struct ChecksumFault {
    pub file: std::path::PathBuf,
    pub page: u64,
    pub offset: u64,
    pub page_size: u32,
}
thread_local! {
    static CHECKSUM_FAULT: std::cell::RefCell<Option<ChecksumFault>> = const { std::cell::RefCell::new(None) };
}
pub(crate) fn take_checksum_fault() -> Option<ChecksumFault> {
    CHECKSUM_FAULT.with(|slot| slot.borrow_mut().take())
}
#[unsafe(no_mangle)]
unsafe extern "C" fn polylogue_checksum_fault(
    filename: *const std::ffi::c_char,
    offset: i64,
    size: i32,
) {
    use std::os::unix::ffi::OsStringExt;
    if filename.is_null()
        || offset < 0
        || size < 512
        || !(size as u32).is_power_of_two()
        || offset % i64::from(size) != 0
    {
        return;
    }
    // The VFS owns a NUL-terminated filename for the complete synchronous call.
    // Copy before returning; no borrowed C pointer escapes.
    let file = unsafe { std::ffi::CStr::from_ptr(filename) }
        .to_bytes()
        .to_vec();
    let fault = ChecksumFault {
        file: std::ffi::OsString::from_vec(file).into(),
        page: offset as u64 / size as u64 + 1,
        offset: offset as u64,
        page_size: size as u32,
    };
    CHECKSUM_FAULT.with(|slot| {
        *slot.borrow_mut() = Some(fault);
    });
}
