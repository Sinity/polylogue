const DEFAULT_CLEANUP_TIMEOUT_MS = 2000;

function boundedCleanup(cleanup, timeoutMs) {
  let timeout = null;
  const deadline = new Promise((resolve) => {
    timeout = setTimeout(resolve, timeoutMs);
    timeout.unref?.();
  });
  return Promise.race([Promise.resolve().then(cleanup), deadline]).finally(() => {
    if (timeout !== null) clearTimeout(timeout);
  });
}

export function createOwnedTargetCleanup({
  control,
  targetId,
  processLike = process,
  timeoutMs = DEFAULT_CLEANUP_TIMEOUT_MS,
}) {
  let cleanupPromise = null;
  let signalReceived = false;
  const handlers = new Map();

  const removeSignalHandlers = () => {
    for (const [signalName, handler] of handlers) processLike.off(signalName, handler);
    handlers.clear();
  };

  const close = () => {
    if (cleanupPromise === null) {
      cleanupPromise = boundedCleanup(() => control(["close", targetId]), timeoutMs);
    }
    return cleanupPromise;
  };

  for (const signalName of ["SIGINT", "SIGTERM"]) {
    const handler = () => {
      if (signalReceived) return;
      signalReceived = true;
      void close()
        .catch(() => undefined)
        .finally(() => {
          removeSignalHandlers();
          processLike.kill(processLike.pid, signalName);
        });
    };
    handlers.set(signalName, handler);
    processLike.on(signalName, handler);
  }

  return {
    async finish() {
      try {
        await close();
      } finally {
        if (!signalReceived) removeSignalHandlers();
      }
    },
  };
}
