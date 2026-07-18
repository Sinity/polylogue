import type {
  SemanticCard,
  SemanticCardOutcome,
  SemanticCardPreview,
  SemanticEntry,
  SemanticNotice,
  SemanticProse,
} from '../contracts/semantic-cards';

/** Matches SEM_CARD_LABEL in polylogue/daemon/web_shell_semantic_cards.py and
 * SEMANTIC_CARD_LABELS in polylogue/daemon/webui.py, so all three surfaces agree. */
const SEMANTIC_CARD_LABELS: Readonly<Record<string, string>> = {
  shell: 'shell',
  file_read: 'read',
  file_edit: 'edit',
  search: 'search',
  web: 'web',
  task: 'task',
  mcp: 'mcp',
  lineage: 'lineage',
  attachment: 'attachment',
  fallback: 'tool',
};

const OUTCOME_LABELS: Readonly<Record<string, string>> = { succeeded: 'ok', failed: 'FAILED' };

const SOURCE_FIELD_ORDER = [
  'session_id',
  'provider_family',
  'origin',
  'message_id',
  'block_id',
  'block_index',
  'tool_name',
  'tool_id',
  'attachment_id',
  'material_origin',
  'occurred_at',
  'duration_ms',
  'parent_message_id',
  'variant_index',
  'is_active_path',
  'is_active_leaf',
  'inherited_prefix',
  'result_message_id',
  'result_block_id',
  'result_block_index',
  'result_duration_ms',
  'result_material_origin',
  'result_inherited_prefix',
] as const;

function renderDiffLine(line: string, index: number) {
  let cssClass = 'diff-ctx';
  if (line.startsWith('+') && !line.startsWith('+++')) cssClass = 'diff-add';
  else if (line.startsWith('-') && !line.startsWith('---')) cssClass = 'diff-del';
  else if (line.startsWith('@')) cssClass = 'diff-hunk';
  return (
    <span key={index} class={cssClass}>
      {line}
      {'\n'}
    </span>
  );
}

function CardOutcome({ outcome }: { readonly outcome: SemanticCardOutcome | null }) {
  if (outcome === null) {
    return null;
  }
  const label = OUTCOME_LABELS[outcome.state] ?? 'unknown';
  const detailBits: string[] = [];
  if (outcome.is_error !== null) detailBits.push(`is_error=${outcome.is_error}`);
  if (outcome.exit_code !== null) detailBits.push(`exit ${outcome.exit_code}`);
  const title = detailBits.length ? `${label} (${detailBits.join(', ')})` : label;
  return (
    <span class="card__outcome" data-outcome-state={outcome.state} title={title}>
      {label}
    </span>
  );
}

function CardPreview({ preview }: { readonly preview: SemanticCardPreview }) {
  const metaBits = [`${preview.line_count} line${preview.line_count === 1 ? '' : 's'}`];
  if (preview.omitted_lines) metaBits.push(`${preview.omitted_lines} omitted`);
  if (preview.omitted_characters) metaBits.push(`${preview.omitted_characters} chars omitted`);
  if (preview.encoding_replacements) metaBits.push(`${preview.encoding_replacements} replacements`);
  return (
    <details class="card__preview" data-preview-kind={preview.kind}>
      <summary>
        {preview.kind} · {metaBits.join(', ')}
      </summary>
      <pre>{preview.kind === 'diff' ? preview.text.split('\n').map(renderDiffLine) : preview.text}</pre>
    </details>
  );
}

/**
 * Hyperlinks the registry's ``session:<id>``/``message:<id>`` field-value ref
 * convention (polylogue/rendering/semantic_cards.py `_build_lineage_card`/
 * `_build_task_card`) generically, so any card family adopting it - not just
 * lineage - gets working navigation without a bespoke ref field.
 */
function CardFieldValue({ value }: { readonly value: string }) {
  if (value.startsWith('session:') && value.length > 'session:'.length) {
    const sessionRef = value.slice('session:'.length);
    return (
      <a class="card__field-value" href={`/app/sessions/${encodeURIComponent(sessionRef)}`}>
        <code>{value}</code>
      </a>
    );
  }
  if (value.startsWith('message:') && value.length > 'message:'.length) {
    const messageRef = value.slice('message:'.length);
    return (
      <a class="card__field-value" href={`#msg-${encodeURIComponent(messageRef)}`}>
        <code>{value}</code>
      </a>
    );
  }
  return <code class="card__field-value">{value}</code>;
}

function CardSource({ source }: { readonly source: SemanticCard['source'] }) {
  const known = new Set<string>(SOURCE_FIELD_ORDER);
  const bits = SOURCE_FIELD_ORDER.filter((key) => source[key] !== undefined && source[key] !== null).map(
    (key) => `${key}=${String(source[key])}`,
  );
  // SemanticCardSource is a loose passthrough (polylogue/rendering/semantic_card_models.py);
  // a registry-added field not yet in SOURCE_FIELD_ORDER must still surface as
  // evidence rather than silently disappearing from the hydrated card.
  for (const [key, value] of Object.entries(source)) {
    if (!known.has(key) && value !== undefined && value !== null) {
      bits.push(`${key}=${String(value)}`);
    }
  }
  if (bits.length === 0) {
    return null;
  }
  return (
    <details class="card__source">
      <summary>evidence</summary>
      <code>{bits.join('\n')}</code>
    </details>
  );
}

export function SemanticCardView({ card }: { readonly card: SemanticCard }) {
  const label = SEMANTIC_CARD_LABELS[card.kind] ?? card.kind;
  const hasSummaryField = card.fields.some((field) => field.value === card.summary);
  return (
    <div class={`card card--${card.kind}`} data-card-kind={card.kind}>
      <div class="card__header">
        <span class="card__kind">{label}</span>
        <span class="card__title">{card.title}</span>
        <CardOutcome outcome={card.outcome} />
      </div>
      {card.summary && !hasSummaryField ? <p class="card__summary">{card.summary}</p> : null}
      {card.fields.map((field) => (
        <div key={field.label} class="card__field">
          <span class="card__field-label">{field.label}</span>
          <CardFieldValue value={field.value} />
        </div>
      ))}
      {card.previews.map((preview, index) => (
        <CardPreview key={index} preview={preview} />
      ))}
      {card.caveats.length > 0 ? (
        <ul class="card__caveats">
          {card.caveats.map((caveat, index) => (
            <li key={index}>{caveat}</li>
          ))}
        </ul>
      ) : null}
      <CardSource source={card.source} />
    </div>
  );
}

export function SemanticProseView({ prose }: { readonly prose: SemanticProse }) {
  const metaBits = [prose.block_type, prose.language, prose.material_origin ? `material:${prose.material_origin}` : null].filter(
    (bit): bit is string => Boolean(bit),
  );
  return (
    <div class="prose" data-semantic-block-type={prose.block_type ?? ''}>
      {metaBits.length > 0 ? (
        <div class="prose__meta">
          {metaBits.map((bit, index) => (
            <span key={index} class="chip">
              {bit}
            </span>
          ))}
        </div>
      ) : null}
      {prose.text ? <p class="prose__text">{prose.text}</p> : null}
    </div>
  );
}

export function SemanticNoticeView({ notice }: { readonly notice: SemanticNotice }) {
  const label = notice.kind === 'empty_thinking' ? 'thinking absent' : notice.kind;
  return (
    <div class="notice" data-notice-kind={notice.kind}>
      <strong>{label}</strong> · {notice.count} typed block{notice.count === 1 ? '' : 's'}
    </div>
  );
}

export function SemanticEntryView({ entry }: { readonly entry: SemanticEntry }) {
  if (entry.entry_type === 'card') return <SemanticCardView card={entry.card} />;
  if (entry.entry_type === 'prose') return <SemanticProseView prose={entry.prose} />;
  return <SemanticNoticeView notice={entry.notice} />;
}

export function SemanticEntries({ entries }: { readonly entries: readonly SemanticEntry[] }) {
  return (
    <>
      {entries.map((entry, index) => (
        <SemanticEntryView key={index} entry={entry} />
      ))}
    </>
  );
}
