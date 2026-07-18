import { useEffect, useState } from 'preact/hooks';

import { Button } from './layout';

export type ThemePreference = 'system' | 'light' | 'dark';
const STORAGE_KEY = 'polylogue.webui.theme';
const ORDER: ReadonlyArray<ThemePreference> = ['system', 'light', 'dark'];

function validPreference(value: string | null): value is ThemePreference {
  return value === 'system' || value === 'light' || value === 'dark';
}

export function readThemePreference(storage: Pick<Storage, 'getItem'> = window.localStorage): ThemePreference {
  const value = storage.getItem(STORAGE_KEY);
  return validPreference(value) ? value : 'system';
}

export function applyThemePreference(
  preference: ThemePreference,
  root: HTMLElement = document.documentElement,
  storage: Pick<Storage, 'setItem' | 'removeItem'> = window.localStorage,
): void {
  if (preference === 'system') {
    root.removeAttribute('data-theme');
    storage.removeItem(STORAGE_KEY);
  } else {
    root.dataset.theme = preference;
    storage.setItem(STORAGE_KEY, preference);
  }
}

export function ThemeToggle() {
  const [preference, setPreference] = useState<ThemePreference>('system');

  useEffect(() => {
    const stored = readThemePreference();
    setPreference(stored);
    applyThemePreference(stored);
  }, []);

  const currentIndex = ORDER.indexOf(preference);
  const next = ORDER[(currentIndex + 1) % ORDER.length] ?? 'system';
  return (
    <Button
      className="pl-theme-toggle"
      variant="quiet"
      aria-label={`Theme preference: ${preference}. Activate for ${next}.`}
      onClick={() => {
        setPreference(next);
        applyThemePreference(next);
      }}
    >
      Theme: {preference}
    </Button>
  );
}
