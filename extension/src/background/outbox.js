// Captures waiting for the host to acknowledge them. A capture is put here
// before it is sent and removed once the host has synced it to disk, so a
// capture survives a stopped service worker, a browser restart and an
// extension update. This is all the extension keeps; everything else lives
// in Polylogue.

const DATABASE = "polylogue";
const STORE = "outbox";

function done(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export class Outbox {
  constructor(indexedDB = globalThis.indexedDB) {
    this.indexedDB = indexedDB;
    this.opening = null;
  }

  database() {
    if (!this.opening) {
      const request = this.indexedDB.open(DATABASE, 1);
      request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "key", autoIncrement: true });
      this.opening = done(request);
      this.opening.catch(() => { this.opening = null; });
    }
    return this.opening;
  }

  /** Runs `work` on the store in one transaction and waits for it to commit. */
  async run(mode, work) {
    const db = await this.database();
    const tx = db.transaction(STORE, mode);
    work(tx.objectStore(STORE));
    await new Promise((resolve, reject) => {
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  }

  /** Keeps `capture` ({provider, conversation, body: Blob}); its key. */
  async add({ provider, conversation, body }, current = () => true) {
    const bytes = await body.arrayBuffer();
    if (!current()) return null;
    let key;
    await this.run("readwrite", (store) => {
      const request = store.add({ provider, conversation, bytes, added_at: new Date().toISOString(), attempts: 0, last_error: null });
      request.onsuccess = () => { key = request.result; };
    });
    return key;
  }

  /** Every waiting capture, oldest first, with its body as a Blob. */
  async all() {
    let rows = [];
    await this.run("readonly", (store) => {
      const request = store.getAll();
      request.onsuccess = () => { rows = request.result; };
    });
    return rows.map(({ bytes, ...row }) => ({ ...row, body: new Blob([bytes], { type: "application/json" }) }));
  }

  async count() {
    let count = 0;
    await this.run("readonly", (store) => {
      const request = store.count();
      request.onsuccess = () => { count = request.result; };
    });
    return count;
  }

  async remove(key) {
    await this.run("readwrite", (store) => store.delete(key));
  }

  async failed(key, error) {
    await this.run("readwrite", (store) => {
      const request = store.get(key);
      request.onsuccess = () => {
        if (request.result) {
          store.put({ ...request.result, attempts: request.result.attempts + 1, last_error: String(error) });
        }
      };
    });
  }
}
