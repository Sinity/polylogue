import { render } from 'preact';

import { isRecord, nullableNumber, nullableString } from '../contracts/runtime';
import { SessionListIsland } from '../islands/session-list';
import '../styles.css';

const root = document.getElementById('session-list-island');
const bootstrap = document.getElementById('session-list-bootstrap');
if (root !== null && bootstrap !== null) {
  try {
    const parsed: unknown = JSON.parse(bootstrap.textContent ?? '');
    if (!isRecord(parsed) || !isRecord(parsed.filters)) {
      throw new TypeError('session list bootstrap payload is malformed');
    }
    const filters = {
      origin: nullableString(parsed.filters, 'origin') ?? undefined,
      since: nullableString(parsed.filters, 'since') ?? undefined,
      repo: nullableString(parsed.filters, 'repo') ?? undefined,
    };
    render(
      <SessionListIsland filters={filters} initialNextOffset={nullableNumber(parsed, 'next_offset')} />,
      root,
    );
  } catch {
    // Semantic SSR stays available if an old asset cannot parse a new payload.
  }
}
