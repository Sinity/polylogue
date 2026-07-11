import { existsSync } from 'node:fs';

import { defineConfig } from '@playwright/test';

const executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ?? [
  '/etc/profiles/per-user/sinity/bin/google-chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
].find(existsSync);

export default defineConfig({
  testDir: './tests',
  fullyParallel: false,
  workers: 1,
  timeout: 45_000,
  expect: { timeout: 10_000 },
  reporter: [['line']],
  use: {
    headless: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'off',
    launchOptions: executablePath ? { executablePath } : undefined,
  },
});
