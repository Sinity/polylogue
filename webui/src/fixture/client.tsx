import { hydrate } from 'preact';

import '../design-system/design-system.css';
import { App } from './app';
import { routeFromUrl } from './routes';

const root = document.getElementById('app');
if (!root) throw new Error('WebUI fixture root is missing');

hydrate(<App route={routeFromUrl(new URL(window.location.href))} />, root);
