// Fire an event the way the browser does for a real operator gesture.
//
// Content-script controls act only on `event.isTrusted` events, because page
// scripts share the realm and can dispatch synthetic ones. `dispatchEvent`
// always clears the trusted flag, so tests that model an operator gesture go
// through jsdom's own internal dispatch, which is how jsdom fires the trusted
// events it originates itself.
import utils from "jsdom/lib/jsdom/living/generated/utils.js";

export function dispatchTrusted(target, event) {
  const eventImpl = utils.implForWrapper(event);
  eventImpl.isTrusted = true;
  return utils.implForWrapper(target)._dispatch(eventImpl);
}

export function trustedClick(element) {
  const view = element.ownerDocument.defaultView;
  return dispatchTrusted(element, new view.MouseEvent("click", { bubbles: true, cancelable: true, composed: true }));
}
