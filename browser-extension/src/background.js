import { createBackgroundAdapters } from "./background/adapters.js";
import { startBackgroundRuntime } from "./background/runtime.js";

// MV3 service workers are disposable. This file is intentionally only the
// composition root: domain state is reconstructed by the runtime from its
// durable adapters whenever Chrome starts or wakes the worker.
const adapters = createBackgroundAdapters();
startBackgroundRuntime(adapters);

export { adapters };
