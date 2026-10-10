import { readFileSync } from "node:fs";
import { Script, createContext } from "node:vm";

// Borrow the original worker functions without starting browser listeners.
export function receiverConfigurationOwner(chrome, probeReceiverStatus = async () => { throw new Error("probe_not_expected"); }) {
  const source = readFileSync(new URL("../../src/background/runtime.js", import.meta.url), "utf8");
  const definitions = source.slice(source.indexOf("async function receiverSettings()"), source.indexOf("function hostnameForUrl("));
  const health = source.slice(source.indexOf("async function checkReceiverHealth("), source.indexOf("async function appendCaptureLog("));
  const handler = source.slice(source.indexOf('    if (message.type === "polylogue.configureReceiver")'), source.indexOf('    if (message.type === "polylogue.backfill.start")'));
  const resetHandler = source.slice(source.indexOf('    if (message.type === "polylogue.receiverPairing.reset")'), source.indexOf('    if (message.type === "polylogue.ambient.configure")'));
  const context = createContext({ Date, URL, runtimeChrome: chrome, probeReceiverStatus,
    RECEIVER_PAIRING_KEY: "polylogueReceiverPairing", RECEIVER_API_SCHEMA: "polylogue-browser-capture/v1",
    DEFAULT_RECEIVER: "http://127.0.0.1:8765", trustedReceiverHealthCache: null });
  new Script("let storageMutationQueue = Promise.resolve();" + source.slice(source.indexOf("function serializeStorageMutation("), source.indexOf("function replaceLegacyAcceptedMessageIdentities(")) + definitions + health + "\nglobalThis.health = checkReceiverHealth; globalThis.reset = clearReceiverPairing; globalThis.dispatch = async function(message,sendResponse){" + handler + resetHandler + "};").runInContext(context);
  return { health: context.health, reset: context.reset, async send(message) {
    let result;
    await context.dispatch(message, value => { result = value; });
    return result;
  } };
}
