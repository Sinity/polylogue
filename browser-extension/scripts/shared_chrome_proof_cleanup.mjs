// The close command is bounded by the control subprocess itself, which
// terminates the child on its own timeout. Racing it here would abandon a slow
// close that is still making progress and could leave the shared target open.

export function createOwnedTargetCleanup({
  control,
  targetId,
  processLike = process,
  afterClose = async () => {},
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
      cleanupPromise = Promise.resolve().then(() => control(["close", targetId])).finally(afterClose);
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
