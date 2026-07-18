import { isRecord, nullableString, optionalBoolean, requiredNumber, requiredString } from './runtime';
import { parseSemanticEntries, type SemanticEntry } from './semantic-cards';

export const SESSION_READ_MESSAGE_LIMIT = 30;

export interface SessionMessageRow {
  readonly id: string;
  readonly role: string;
  readonly material_origin: string;
  readonly text: string;
  readonly timestamp: string | null;
  readonly has_tool_use: boolean;
  readonly has_thinking: boolean;
  readonly has_paste_evidence: boolean;
  readonly semantic_entries: readonly SemanticEntry[];
  readonly semantic_card_suppressed: boolean;
}

export interface SessionMessagePage {
  readonly messages: readonly SessionMessageRow[];
  readonly total: number;
  readonly limit: number;
  readonly offset: number;
}

function parseSessionMessageRow(value: unknown, index: number): SessionMessageRow {
  if (!isRecord(value)) {
    throw new TypeError(`session message ${index} is not an object`);
  }
  return {
    id: requiredString(value, 'id'),
    role: requiredString(value, 'role'),
    material_origin: requiredString(value, 'material_origin'),
    text: requiredString(value, 'text'),
    timestamp: nullableString(value, 'timestamp'),
    has_tool_use: optionalBoolean(value, 'has_tool_use'),
    has_thinking: optionalBoolean(value, 'has_thinking'),
    has_paste_evidence: optionalBoolean(value, 'has_paste_evidence'),
    semantic_entries: parseSemanticEntries(value.semantic_entries),
    semantic_card_suppressed: optionalBoolean(value, 'semantic_card_suppressed'),
  };
}

/**
 * Parse the ``payload`` field of a ``SessionReadViewEnvelope`` returned by
 * ``GET /api/sessions/:id/read?view=messages``.
 */
export function parseSessionMessagePage(value: unknown): SessionMessagePage {
  if (!isRecord(value) || !isRecord(value.payload)) {
    throw new TypeError('response is not a session read-view envelope');
  }
  const payload = value.payload;
  if (!Array.isArray(payload.messages)) {
    throw new TypeError('session read payload messages must be an array');
  }
  return {
    messages: payload.messages.map(parseSessionMessageRow),
    total: requiredNumber(payload, 'total'),
    limit: requiredNumber(payload, 'limit'),
    offset: requiredNumber(payload, 'offset'),
  };
}
