import { render } from 'preact';

import { isRecord, requiredString } from '../contracts/runtime';
import { SessionReadIsland } from '../islands/session-read';
import '../styles.css';

const root = document.getElementById('session-read-island');
const bootstrap = document.getElementById('session-read-bootstrap');
if (root !== null && bootstrap !== null) {
  try {
    const parsed: unknown = JSON.parse(bootstrap.textContent ?? '');
    if (!isRecord(parsed)) {
      throw new TypeError('session read bootstrap payload is malformed');
    }
    const nextOffset = parsed.next_offset;
    if (nextOffset !== null && typeof nextOffset !== 'number') {
      throw new TypeError('session read bootstrap next_offset must be a number or null');
    }
    render(
      <SessionReadIsland sessionId={requiredString(parsed, 'session_id')} initialNextOffset={nextOffset} />,
      root,
    );
  } catch {
    // Semantic SSR stays available if an old asset cannot parse a new payload.
  }
}
