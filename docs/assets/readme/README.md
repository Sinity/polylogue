# README Media

These assets are generated from Polylogue's fixture corpus workflow. Do not use
private archive screenshots or recordings for committed README media.

Regenerate the terminal media with:

```bash
bash docs/assets/readme/render-readme-media.sh
```

Regenerate the static-site screenshot by seeding a demo workspace, rendering the
site, and capturing `index.html` with a clean browser profile:

```bash
rm -rf .local/readme-demo .local/chrome-readme-shot
eval "$(devtools lab-corpus seed --count 8 --output-dir .local/readme-demo --env-only 2>/dev/null | grep '^export ')"
google-chrome --headless=new --disable-gpu --no-sandbox \
  --user-data-dir="$PWD/.local/chrome-readme-shot" \
  --window-size=1440,1100 \
  --screenshot=docs/assets/readme/synthetic-site.png \
  "file://$PWD/.local/readme-demo/data/polylogue/site/index.html"
```
