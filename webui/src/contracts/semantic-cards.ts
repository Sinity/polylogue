import { isRecord } from './runtime';

/**
 * Loose passthrough for ``SemanticCardSource`` (docs/schemas/semantic-card-v1.schema.json).
 * Every field is optional evidence; the renderer displays whichever are present.
 */
export type SemanticCardSource = Readonly<Record<string, string | number | boolean | null | undefined>>;

export interface SemanticCardField {
  readonly label: string;
  readonly value: string;
}

export interface SemanticCardPreview {
  readonly kind: string;
  readonly text: string;
  readonly line_count: number;
  readonly omitted_lines: number;
  readonly omitted_characters: number;
  readonly truncated: boolean;
  readonly strategy: string;
  readonly encoding_replacements: number;
}

export interface SemanticCardOutcome {
  readonly state: string;
  readonly is_error: boolean | null;
  readonly exit_code: number | null;
}

export interface SemanticCard {
  readonly kind: string;
  readonly title: string;
  readonly summary: string | null;
  readonly source: SemanticCardSource;
  readonly outcome: SemanticCardOutcome | null;
  readonly fields: readonly SemanticCardField[];
  readonly previews: readonly SemanticCardPreview[];
  readonly caveats: readonly string[];
}

export interface SemanticProse {
  readonly block_type: string | null;
  readonly language: string | null;
  readonly material_origin: string | null;
  readonly text: string;
}

export interface SemanticNotice {
  readonly kind: string;
  readonly count: number;
}

export type SemanticEntry =
  | { readonly entry_type: 'card'; readonly card: SemanticCard }
  | { readonly entry_type: 'prose'; readonly prose: SemanticProse }
  | { readonly entry_type: 'notice'; readonly notice: SemanticNotice };

function parseField(value: unknown): SemanticCardField | null {
  if (!isRecord(value) || typeof value.label !== 'string' || typeof value.value !== 'string') {
    return null;
  }
  return { label: value.label, value: value.value };
}

function parsePreview(value: unknown): SemanticCardPreview | null {
  if (!isRecord(value) || typeof value.kind !== 'string' || typeof value.text !== 'string') {
    return null;
  }
  return {
    kind: value.kind,
    text: value.text,
    line_count: typeof value.line_count === 'number' ? value.line_count : 0,
    omitted_lines: typeof value.omitted_lines === 'number' ? value.omitted_lines : 0,
    omitted_characters: typeof value.omitted_characters === 'number' ? value.omitted_characters : 0,
    truncated: value.truncated === true,
    strategy: typeof value.strategy === 'string' ? value.strategy : 'full',
    encoding_replacements: typeof value.encoding_replacements === 'number' ? value.encoding_replacements : 0,
  };
}

function parseOutcome(value: unknown): SemanticCardOutcome | null {
  if (!isRecord(value) || typeof value.state !== 'string') {
    return null;
  }
  return {
    state: value.state,
    is_error: typeof value.is_error === 'boolean' ? value.is_error : null,
    exit_code: typeof value.exit_code === 'number' ? value.exit_code : null,
  };
}

function parseCard(value: unknown): SemanticCard | null {
  if (!isRecord(value) || typeof value.kind !== 'string' || typeof value.title !== 'string') {
    return null;
  }
  const source = isRecord(value.source) ? (value.source as SemanticCardSource) : {};
  const fields = Array.isArray(value.fields) ? value.fields.map(parseField).filter((f): f is SemanticCardField => f !== null) : [];
  const previews = Array.isArray(value.previews)
    ? value.previews.map(parsePreview).filter((p): p is SemanticCardPreview => p !== null)
    : [];
  const caveats = Array.isArray(value.caveats) ? value.caveats.filter((c): c is string => typeof c === 'string') : [];
  return {
    kind: value.kind,
    title: value.title,
    summary: typeof value.summary === 'string' ? value.summary : null,
    source,
    outcome: parseOutcome(value.outcome),
    fields,
    previews,
    caveats,
  };
}

function parseProse(value: unknown): SemanticProse | null {
  if (!isRecord(value) || typeof value.text !== 'string') {
    return null;
  }
  return {
    block_type: typeof value.block_type === 'string' ? value.block_type : null,
    language: typeof value.language === 'string' ? value.language : null,
    material_origin: typeof value.material_origin === 'string' ? value.material_origin : null,
    text: value.text,
  };
}

function parseNotice(value: unknown): SemanticNotice | null {
  if (!isRecord(value)) {
    return null;
  }
  return {
    kind: typeof value.kind === 'string' ? value.kind : 'notice',
    count: typeof value.count === 'number' ? value.count : 0,
  };
}

/**
 * Parse one entry of a ``semantic_entries`` array. Returns ``null`` for a
 * malformed entry rather than throwing, so one bad entry cannot blank the
 * whole message - the card family this vertical renders must never drop
 * evidence, but it also must not let one degraded entry hide its siblings.
 */
export function parseSemanticEntry(value: unknown): SemanticEntry | null {
  if (!isRecord(value)) {
    return null;
  }
  if (value.entry_type === 'card') {
    const card = parseCard(value.card);
    return card ? { entry_type: 'card', card } : null;
  }
  if (value.entry_type === 'prose') {
    const prose = parseProse(value.prose);
    return prose ? { entry_type: 'prose', prose } : null;
  }
  if (value.entry_type === 'notice') {
    const notice = parseNotice(value.notice);
    return notice ? { entry_type: 'notice', notice } : null;
  }
  return null;
}

export function parseSemanticEntries(value: unknown): readonly SemanticEntry[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map(parseSemanticEntry).filter((entry): entry is SemanticEntry => entry !== null);
}
