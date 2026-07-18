import '@testing-library/jest-dom/vitest';

import { afterEach, beforeEach } from 'vitest';
import { render } from 'preact';

beforeEach(() => {
  document.documentElement.removeAttribute('data-theme');
  document.body.replaceChildren();
  window.localStorage.clear();
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    }),
  });
});

afterEach(() => {
  render(null, document.body);
  document.body.replaceChildren();
});
