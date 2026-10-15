/* The extension and bundled SQLite must be the same upstream release. */
#include "sqlite3.h"
#if SQLITE_VERSION_NUMBER != 3053002
#error "Update cksumvfs.c from the bundled SQLite release before changing SQLite"
#endif
int polylogue_reserve_bytes(sqlite3 *db) {
  int n = 8;
  return sqlite3_file_control(db, "main", SQLITE_FCNTL_RESERVE_BYTES, &n);
}
