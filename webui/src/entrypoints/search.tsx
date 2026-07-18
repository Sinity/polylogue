import { render } from 'preact';

import { isRecord, nullableString, requiredString } from '../contracts/runtime';
import { SearchIsland } from '../islands/search';
import '../styles.css';

const root = document.getElementById('search-island');
const bootstrap = document.getElementById('search-bootstrap');
if (root !== null && bootstrap !== null) {
  try {
    const parsed: unknown = JSON.parse(bootstrap.textContent ?? '');
    if (!isRecord(parsed)) {
      throw new TypeError('search bootstrap payload is malformed');
    }
    render(
      <SearchIsland query={requiredString(parsed, 'query')} initialCursor={nullableString(parsed, 'next_cursor')} />,
      root,
    );
  } catch {
    // Semantic SSR stays available if an old asset cannot parse a new payload.
  }
}
