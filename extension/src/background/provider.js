// One request lane per provider. Durable bytes remain exclusively in the outbox.
export class ProviderDeferred extends Error {
  constructor(until, reason = "the daily backfill budget is spent; continuing tomorrow") {
    super(reason);
    this.until = until;
  }
}

export class ProviderScheduler {
  constructor({ now = Date.now, sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)), random = Math.random, storage = null, dailyBudget = 10000 } = {}) {
    this.now = now;
    this.sleep = sleep;
    this.random = random;
    this.storage = storage;
    this.dailyBudget = dailyBudget;
    this.providers = new Map();
  }

  run(provider, request, { background = false, before = null, owner = null, priority = background ? 1 : 0 } = {}) {
    const state = this.providers.get(provider) || { queue: [], running: false, loaded: false };
    this.providers.set(provider, state);
    return new Promise((resolve, reject) => {
      state.queue.push({ request, background, before, owner, priority, resolve, reject });
      if (!state.running) void this.drain(provider, state);
    });
  }

  async drain(provider, state) {
    state.running = true;
    try {
      while (state.queue.length) {
        state.queue.sort((a, b) => a.priority - b.priority);
        const work = state.queue.shift();
        try { work.resolve(await this.execute(provider, state, work)); }
        catch (error) { work.reject(error); }
      }
    } finally { state.running = false; }
  }

  async save(provider, state) {
    await this.storage?.set({ [`provider pacing ${provider}`]: {
      until: state.until, failures: state.failures, day: state.day, count: state.count,
    } });
  }

  // Content scripts request this permit immediately before each actual HTTP
  // fetch, including auth and download-link requests within one operation.
  async authorize(provider, owner = null) {
    const state = this.providers.get(provider);
    if (!state?.active || state.owner !== owner) throw new Error("provider request has no active owner");
    if (!state.background) return;
    const day = Math.floor(this.now() / 86400000);
    if (day !== state.day) { state.day = day; state.count = 0; }
    if (state.count >= this.dailyBudget) throw new ProviderDeferred((day + 1) * 86400000);
    state.count += 1;
    await this.save(provider, state);
  }

  async execute(provider, state, { request, background, before, owner }) {
    if (!state.loaded) {
      const key = `provider pacing ${provider}`;
      Object.assign(state, { until: 0, failures: 0, day: null, count: 0 }, (await this.storage?.get({ [key]: null }))?.[key]);
      state.loaded = true;
    }
    if (background && state.until > this.now()) throw new ProviderDeferred(state.until, "the provider asked to wait; continuing later");
    while (state.until > this.now()) await this.sleep(state.until - this.now());
    if (before) await before();
    state.background = background;
    state.owner = owner;
    state.active = true;
    try {
      const answer = await request();
      this.observe(state, answer);
      await this.save(provider, state);
      return answer;
    } catch (error) {
      this.observe(state, error);
      await this.save(provider, state);
      throw error;
    } finally { state.active = false; }
  }

  observe(state, answer) {
    const status = answer?.status;
    if (status !== 429 && !(status >= 500 && status <= 599)) {
      if (status >= 200 && status < 300) state.failures = 0;
      return;
    }
    const retry = answer.retry_after ?? answer.retryAfter;
    const seconds = retry === null || retry === undefined || retry === "" ? NaN : Number(retry);
    const delay = Number.isFinite(seconds) ? Math.max(0, seconds * 1000)
      : Math.max(0, Date.parse(retry) - this.now());
    state.failures += 1;
    const backoff = Math.min(300000, 1000 * 2 ** Math.min(state.failures, 8)) * (0.5 + this.random());
    state.until = Math.max(state.until, this.now() + (Number.isFinite(delay) ? delay : backoff));
  }
}
