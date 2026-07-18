import { render } from 'preact';

import { parseObservabilityPayload } from '../contracts/observability';
import { ObservabilityIsland } from '../islands/observability';
import '../styles.css';

const root = document.getElementById('observability-island');
const bootstrap = document.getElementById('observability-bootstrap');
if (root !== null && bootstrap !== null) {
  try {
    render(<ObservabilityIsland initial={parseObservabilityPayload(JSON.parse(bootstrap.textContent ?? ''))} />, root);
  } catch {
    // Semantic SSR stays available if an old asset cannot parse a new payload.
  }
}
