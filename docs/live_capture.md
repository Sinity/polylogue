# Live Capture Considerations

Not every provider offers an API for historical exports. The notes below outline the current limitations and the precautions to take before attempting unattended capture.

## ChatGPT (consumer web)

- There is no public endpoint for the chatgpt.com history. Any continuous capture must automate the browser (Playwright, Selenium, etc.) and call undocumented endpoints such as `backend-api/conversation`.
- These flows are brittle—OpenAI can change the private API without warning—and may violate the site’s Terms of Service. Treat them as opt-in tooling with explicit user consent and clear warnings in the CLI.

## Claude.ai

- Claude’s web workspace likewise exposes only manual “Download data” exports. Historical access requires scripting a login session and replaying browser requests with stored tokens.
- As with ChatGPT, the approach is fragile and should remain optional. Document the risks and provide escape hatches before enabling any automation in Polylogue.
