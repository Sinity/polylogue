export type ExactCoverage = {
  readonly kind: "exact";
  readonly total: number;
};

export type QualifiedCoverage = {
  readonly kind: "qualified";
  readonly total: number | null;
  readonly qualification: "page" | "capped" | "sampled" | "estimate" | "unknown";
};

export type Coverage = ExactCoverage | QualifiedCoverage;

export type Page<T, TEnvelope = unknown> = {
  readonly items: ReadonlyArray<T>;
  readonly cursor: string | null;
  readonly coverage: Coverage;
  readonly queryRef: string | null;
  readonly resultRef: string | null;
  readonly envelope: TEnvelope;
};

export type QueryScalar = string | number | boolean;
export type QueryValue = QueryScalar | ReadonlyArray<QueryScalar> | null | undefined;
export type QueryParameters = Readonly<Record<string, QueryValue>>;

export type ClientRequest = {
  readonly method: "DELETE" | "GET" | "PATCH" | "POST" | "PUT";
  readonly path: string;
  readonly query?: QueryParameters;
  readonly body?: unknown;
};

export type RequestOptions = {
  readonly signal?: AbortSignal;
  readonly deadline?: Date | number;
  readonly timeoutMs?: number;
  readonly headers?: Readonly<Record<string, string>>;
};

export type DaemonErrorEnvelope = {
  readonly ok?: false;
  readonly error: string;
  readonly detail?: string | null;
  readonly field?: string | null;
  readonly [key: string]: unknown;
};

export interface ClientTransport {
  request<TResponse, TError = DaemonErrorEnvelope>(
    request: ClientRequest,
    options?: RequestOptions,
  ): Promise<TResponse>;
}

export class SameOriginViolationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SameOriginViolationError";
  }
}

export class DaemonProtocolError extends Error {
  readonly status: number;
  readonly responseText: string;

  constructor(message: string, options: { readonly status: number; readonly responseText: string }) {
    super(message);
    this.name = "DaemonProtocolError";
    this.status = options.status;
    this.responseText = options.responseText;
  }
}

export class DaemonHttpError<TPayload = DaemonErrorEnvelope> extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly payload: TPayload;
  readonly code: string | null;
  readonly detail: string | null;
  readonly field: string | null;
  readonly requestId: string | null;
  readonly credentialState: string | null;

  constructor(
    response: Response,
    payload: TPayload,
    options: {
      readonly code: string | null;
      readonly detail: string | null;
      readonly field: string | null;
    },
  ) {
    const summary = options.detail ?? options.code ?? `${response.status} ${response.statusText}`;
    super(`Polylogue daemon request failed: ${summary}`);
    this.name = "DaemonHttpError";
    this.status = response.status;
    this.statusText = response.statusText;
    this.payload = payload;
    this.code = options.code;
    this.detail = options.detail;
    this.field = options.field;
    this.requestId = response.headers.get("X-Request-ID");
    this.credentialState = response.headers.get("X-Polylogue-Web-Credential-State");
  }
}

export class RequestAbortedError extends Error {
  readonly reason: unknown;

  constructor(reason: unknown) {
    super("Polylogue daemon request was aborted");
    this.name = "RequestAbortedError";
    this.reason = reason;
  }
}

export class DeadlineExceededError extends Error {
  readonly deadlineMs: number;

  constructor(deadlineMs: number) {
    super("Polylogue daemon request deadline was exceeded");
    this.name = "DeadlineExceededError";
    this.deadlineMs = deadlineMs;
  }
}

export class TransportError extends Error {
  override readonly cause: unknown;

  constructor(cause: unknown) {
    super("Polylogue daemon request failed before an HTTP response was received");
    this.name = "TransportError";
    this.cause = cause;
  }
}

export class ContinuationProtocolError extends Error {
  readonly cursor: string;

  constructor(cursor: string) {
    super("Polylogue daemon returned a repeated continuation cursor");
    this.name = "ContinuationProtocolError";
    this.cursor = cursor;
  }
}

export type FetchTransportOptions = {
  readonly baseUrl?: string | URL;
  readonly fetch?: typeof fetch;
  readonly defaultTimeoutMs?: number;
};

const DEADLINE_ABORT_REASON = Symbol("polylogue-deadline");

function browserOrigin(): string | null {
  return typeof globalThis.location === "object" && typeof globalThis.location.origin === "string"
    ? globalThis.location.origin
    : null;
}

function normalizedOrigin(value: string | URL): URL {
  const url = new URL(value);
  if (url.username || url.password) {
    throw new SameOriginViolationError("WebUI transport origins must not contain credentials");
  }
  return new URL(url.origin);
}

function requestId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `webui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function errorFields(payload: unknown): {
  readonly code: string | null;
  readonly detail: string | null;
  readonly field: string | null;
} {
  if (typeof payload !== "object" || payload === null) {
    return { code: null, detail: null, field: null };
  }
  const record = payload as Readonly<Record<string, unknown>>;
  return {
    code: typeof record.error === "string" ? record.error : null,
    detail: typeof record.detail === "string" ? record.detail : null,
    field: typeof record.field === "string" ? record.field : null,
  };
}

function appendQuery(url: URL, query: QueryParameters | undefined): void {
  if (query === undefined) {
    return;
  }
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null) {
      continue;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        url.searchParams.append(key, String(item));
      }
      continue;
    }
    url.searchParams.set(key, String(value));
  }
}

function absoluteDeadline(options: RequestOptions, defaultTimeoutMs: number | null): number | null {
  const values: number[] = [];
  if (options.deadline instanceof Date) {
    values.push(options.deadline.getTime());
  } else if (typeof options.deadline === "number") {
    values.push(options.deadline);
  }
  if (typeof options.timeoutMs === "number") {
    values.push(Date.now() + Math.max(0, options.timeoutMs));
  } else if (defaultTimeoutMs !== null) {
    values.push(Date.now() + defaultTimeoutMs);
  }
  return values.length === 0 ? null : Math.min(...values);
}

function buildSignal(
  options: RequestOptions,
  defaultTimeoutMs: number | null,
): {
  readonly signal: AbortSignal;
  readonly deadlineMs: number | null;
  readonly cleanup: () => void;
} {
  const controller = new AbortController();
  const deadlineMs = absoluteDeadline(options, defaultTimeoutMs);
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const relayAbort = (): void => controller.abort(options.signal?.reason);

  if (options.signal?.aborted === true) {
    controller.abort(options.signal.reason);
  } else if (options.signal !== undefined) {
    options.signal.addEventListener("abort", relayAbort, { once: true });
  }

  if (deadlineMs !== null) {
    const remaining = deadlineMs - Date.now();
    if (remaining <= 0) {
      controller.abort(DEADLINE_ABORT_REASON);
    } else {
      timeout = setTimeout(() => controller.abort(DEADLINE_ABORT_REASON), remaining);
    }
  }

  return {
    signal: controller.signal,
    deadlineMs,
    cleanup: () => {
      if (timeout !== undefined) {
        clearTimeout(timeout);
      }
      options.signal?.removeEventListener("abort", relayAbort);
    },
  };
}

export class FetchTransport implements ClientTransport {
  readonly #origin: URL | null;
  readonly #fetch: typeof fetch;
  readonly #defaultTimeoutMs: number | null;

  constructor(options: FetchTransportOptions = {}) {
    const configuredOrigin = options.baseUrl ?? browserOrigin();
    this.#origin = configuredOrigin === null ? null : normalizedOrigin(configuredOrigin);
    this.#fetch = options.fetch ?? globalThis.fetch;
    if (typeof this.#fetch !== "function") {
      throw new TransportError("global fetch is unavailable");
    }
    const timeout = options.defaultTimeoutMs ?? 30_000;
    this.#defaultTimeoutMs = Number.isFinite(timeout) && timeout >= 0 ? timeout : null;
  }

  async request<TResponse, TError = DaemonErrorEnvelope>(
    request: ClientRequest,
    options: RequestOptions = {},
  ): Promise<TResponse> {
    if (this.#origin === null) {
      throw new SameOriginViolationError("FetchTransport requires a browser origin or an explicit baseUrl");
    }
    if (!request.path.startsWith("/") || request.path.startsWith("//")) {
      throw new SameOriginViolationError("Generated client paths must be origin-relative");
    }

    const url = new URL(request.path, this.#origin);
    if (url.origin !== this.#origin.origin) {
      throw new SameOriginViolationError("Generated client request escaped the configured origin");
    }
    appendQuery(url, request.query);

    const headers = new Headers(options.headers);
    if (!headers.has("Accept")) {
      headers.set("Accept", "application/json");
    }
    headers.set("X-Polylogue-Web-Client", "1");
    if (!headers.has("X-Request-ID")) {
      headers.set("X-Request-ID", requestId());
    }

    let body: BodyInit | undefined;
    if (request.body !== undefined) {
      headers.set("Content-Type", "application/json");
      body = JSON.stringify(request.body);
    }

    const bounded = buildSignal(options, this.#defaultTimeoutMs);
    try {
      const response = await this.#fetch(url, {
        method: request.method,
        headers,
        body: body ?? null,
        signal: bounded.signal,
        credentials: "same-origin",
        cache: "no-store",
        redirect: "error",
        referrerPolicy: "no-referrer",
      });
      const responseText = await response.text();
      let payload: unknown = undefined;
      if (responseText !== "") {
        try {
          payload = JSON.parse(responseText);
        } catch (error) {
          if (response.ok) {
            throw new DaemonProtocolError("Daemon returned invalid JSON", {
              status: response.status,
              responseText,
            });
          }
          payload = { error: "invalid_error_envelope", detail: responseText };
        }
      }
      if (!response.ok) {
        throw new DaemonHttpError<TError>(response, payload as TError, errorFields(payload));
      }
      return payload as TResponse;
    } catch (error) {
      if (error instanceof DaemonHttpError || error instanceof DaemonProtocolError) {
        throw error;
      }
      if (bounded.signal.aborted) {
        if (bounded.signal.reason === DEADLINE_ABORT_REASON && bounded.deadlineMs !== null) {
          throw new DeadlineExceededError(bounded.deadlineMs);
        }
        throw new RequestAbortedError(bounded.signal.reason);
      }
      throw new TransportError(error);
    } finally {
      bounded.cleanup();
    }
  }
}

export function iteratePages<TPage extends Page<unknown>>(
  load: (cursor: string | null) => Promise<TPage>,
): AsyncIterable<TPage> {
  return {
    async *[Symbol.asyncIterator](): AsyncIterator<TPage> {
      let cursor: string | null = null;
      const seen = new Set<string>();
      while (true) {
        const page = await load(cursor);
        if (page.cursor !== null && seen.has(page.cursor)) {
          throw new ContinuationProtocolError(page.cursor);
        }
        yield page;
        if (page.cursor === null) {
          return;
        }
        seen.add(page.cursor);
        cursor = page.cursor;
      }
    },
  };
}
