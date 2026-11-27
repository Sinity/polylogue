// What is known about the capture of the conversation on a page, as one
// value from a closed set. The background's page answer, the host's answer
// about conversations and the extension's last capture are read here and
// nowhere else. Loaded as a classic script by the popup and the in-page
// panel, and imported for its global by the service worker.
(function (root) {
  // The extension's background did not answer about the page.
  const COULD_NOT_CHECK = "could_not_check";
  const NOT_A_CHAT_PAGE = "not_a_chat_page";
  const NO_CONVERSATION_OPEN = "no_conversation_open";
  // The messaging host is not installed for this browser.
  const HOST_NOT_INSTALLED = "host_not_installed";
  // The messaging host is installed and failed to answer.
  const HOST_DID_NOT_ANSWER = "host_did_not_answer";
  // The daemon has read a capture of the conversation.
  const IN_ARCHIVE = "in_archive";
  // As above, and that capture names files whose bytes are not in yet.
  const IN_ARCHIVE_FILES_DOWNLOADING = "in_archive_files_downloading";
  // A capture is in the spool and the daemon, which answers, has not read it.
  const STORED_BEING_READ = "stored_being_read";
  // A capture is in the spool and the daemon does not answer.
  const STORED_DAEMON_NOT_ANSWERING = "stored_daemon_not_answering";
  // Nothing is held, or nothing is known to be, and the last capture failed.
  const LAST_CAPTURE_FAILED = "last_capture_failed";
  // The daemon answers and nothing holds the conversation.
  const NOT_CAPTURED = "not_captured";
  // Nothing is in the spool and the daemon does not answer.
  const DAEMON_DID_NOT_ANSWER = "daemon_did_not_answer";

  const STATES = Object.freeze([
    COULD_NOT_CHECK, NOT_A_CHAT_PAGE, NO_CONVERSATION_OPEN, HOST_NOT_INSTALLED, HOST_DID_NOT_ANSWER,
    IN_ARCHIVE, IN_ARCHIVE_FILES_DOWNLOADING, STORED_BEING_READ, STORED_DAEMON_NOT_ANSWERING,
    LAST_CAPTURE_FAILED, NOT_CAPTURED, DAEMON_DID_NOT_ANSWER,
  ]);

  /**
   * The capture state of a page. `page` is the background's answer about a
   * tab: `{provider, conversation, host, daemon, state, downloading, last}`,
   * or `{error}` when it could not answer, or null for no page. `host` is
   * `ok`, `missing` or `failed`; `daemon`, `state` and `downloading` are the
   * host's answer about the conversation; `last` is the extension's last
   * capture of it. What is held comes before how the last capture went.
   */
  function of(page) {
    if (page?.error) return COULD_NOT_CHECK;
    if (!page?.provider) return NOT_A_CHAT_PAGE;
    if (!page.conversation) return NO_CONVERSATION_OPEN;
    if (page.host === "missing") return HOST_NOT_INSTALLED;
    if (page.host !== "ok") return HOST_DID_NOT_ANSWER;
    if (page.state === "archived") return page.downloading ? IN_ARCHIVE_FILES_DOWNLOADING : IN_ARCHIVE;
    if (page.state === "spooled") return page.daemon === "running" ? STORED_BEING_READ : STORED_DAEMON_NOT_ANSWERING;
    if (page.last?.outcome === "failed") return LAST_CAPTURE_FAILED;
    if (page.state === "missing") return NOT_CAPTURED;
    return DAEMON_DID_NOT_ANSWER;
  }

  /** The capture state of each conversation in the host's answer about some conversations of `provider`. */
  function ofArchiveAnswer(provider, answer) {
    return answer.conversations.map((row) => of({
      provider, conversation: row.id, host: "ok", daemon: answer.daemon, state: row.state, downloading: row.downloading,
    }));
  }

  /** Joins the spool's facts and the daemon's reply in the order asked. */
  async function conversations(host, provider, ids) {
    const spool = await host.spooled(provider, ids);
    let archive = null;
    try { archive = await host.call("internal conversations", { provider, ids }); }
    catch (error) { if (error.code !== "daemon_required") throw error; }
    return {
      daemon: archive ? "running" : "no_answer",
      conversations: ids.map((id, i) => {
        const row = archive?.conversations[i];
        return { ...row, id, state: row?.messages != null ? "archived" : spool.spooled[i] ? "spooled" : archive ? "missing" : "unknown" };
      }),
    };
  }

  /** Reads attachment names, then joins the archive's stored bytes to them. */
  async function captureFiles(host, body) {
    const listed = await host.call("internal capture-files", { capture: await body.text() });
    let held = null;
    const { provider, id: conversation } = listed.conversation;
    try { held = await host.call("internal stored-files", { provider, conversation }); }
    catch (error) { if (error.code !== "daemon_required") throw error; }
    return { ...listed, downloading: held?.downloading === true,
      files: listed.files.map((file) => {
        const stored = held?.files.find((item) => item.id === file.file_id);
        return { ...file, stored: stored ? { blob: stored.blob, size: stored.size } : null };
      }),
    };
  }

  /** A cheap setting read shows whether the daemon answers and actions are on. */
  async function status(host) {
    try {
      const setting = await host.call("set", { key: "browser.actions" });
      return { daemon: "running", chat_actions: setting.value?.type === "boolean" && setting.value.value === true };
    } catch (error) {
      if (error.code !== "daemon_required") throw error;
      return { daemon: "no_answer", chat_actions: false };
    }
  }

  /** Whether the daemon has read a capture of the conversation. */
  function inArchive(state) {
    return state === IN_ARCHIVE || state === IN_ARCHIVE_FILES_DOWNLOADING;
  }

  /** Whether a capture of the conversation is in the archive or the spool. */
  function holdsCapture(state) {
    return inArchive(state) || state === STORED_BEING_READ || state === STORED_DAEMON_NOT_ANSWERING;
  }

  root.polylogueCaptureState = Object.freeze({
    COULD_NOT_CHECK, NOT_A_CHAT_PAGE, NO_CONVERSATION_OPEN, HOST_NOT_INSTALLED, HOST_DID_NOT_ANSWER,
    IN_ARCHIVE, IN_ARCHIVE_FILES_DOWNLOADING, STORED_BEING_READ, STORED_DAEMON_NOT_ANSWERING,
    LAST_CAPTURE_FAILED, NOT_CAPTURED, DAEMON_DID_NOT_ANSWER,
    STATES, of, ofArchiveAnswer, inArchive, holdsCapture, conversations, captureFiles, status,
  });
})(globalThis);
