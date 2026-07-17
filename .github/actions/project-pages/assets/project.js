(() => {
  const root = document.documentElement;
  const saved = localStorage.getItem('sinity-project-theme');
  const preferred = matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  root.dataset.theme = saved || preferred;
  document.addEventListener('click', event => {
    const toggle = event.target.closest('[data-theme-toggle]');
    if (!toggle) return;
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('sinity-project-theme', root.dataset.theme);
  });
})();
