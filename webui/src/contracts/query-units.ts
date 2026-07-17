import { isRecord, nullableNumber, nullableString, requiredNumber, requiredString } from './runtime';

export const ARCHIVE_OVERVIEW_EXPRESSION =
  'messages where words >= 0 | sort by time desc';
export const ARCHIVE_OVERVIEW_LIMIT = 6;

export interface ContinuationPage<T> {
  readonly items: readonly T[];
  readonly total: number;
  readonly limit: number;
  readonly offset: number;
  readonly next_offset: number | null;
  readonly query_ref: string;
  readonly result_ref: string;
  readonly continuation: string | null;
}

export interface MessageQueryRow {
  readonly unit: 'message';
  readonly message_id: string;
  readonly session_id: string;
  readonly origin: string;
  readonly title: string | null;
  readonly role: string;
  readonly message_type: string;
  readonly material_origin: string;
  readonly occurred_at_ms: number | null;
  readonly position: number;
  readonly word_count: number;
  readonly text: string;
}

export interface MessageQueryPage extends ContinuationPage<MessageQueryRow> {
  readonly mode: 'query-unit';
  readonly unit: 'message';
  readonly query: string;
}

function parseMessageRow(value: unknown, index: number): MessageQueryRow {
  if (!isRecord(value) || value.unit !== 'message') {
    throw new TypeError(`query-unit item ${index} is not a message row`);
  }
  return {
    unit: 'message',
    message_id: requiredString(value, 'message_id'),
    session_id: requiredString(value, 'session_id'),
    origin: requiredString(value, 'origin'),
    title: nullableString(value, 'title'),
    role: requiredString(value, 'role'),
    message_type: requiredString(value, 'message_type'),
    material_origin: requiredString(value, 'material_origin'),
    occurred_at_ms: nullableNumber(value, 'occurred_at_ms'),
    position: requiredNumber(value, 'position'),
    word_count: requiredNumber(value, 'word_count'),
    text: requiredString(value, 'text'),
  };
}

export function parseMessageQueryPage(value: unknown): MessageQueryPage {
  if (!isRecord(value) || value.mode !== 'query-unit' || value.unit !== 'message') {
    throw new TypeError('response is not a message query-unit envelope');
  }
  if (!Array.isArray(value.items)) {
    throw new TypeError('query-unit response items must be an array');
  }
  const continuation = nullableString(value, 'continuation');
  const nextOffset = nullableNumber(value, 'next_offset');
  return {
    mode: 'query-unit',
    unit: 'message',
    query: requiredString(value, 'query'),
    items: value.items.map(parseMessageRow),
    total: requiredNumber(value, 'total'),
    limit: requiredNumber(value, 'limit'),
    offset: requiredNumber(value, 'offset'),
    next_offset: nextOffset,
    query_ref: requiredString(value, 'query_ref'),
    result_ref: requiredString(value, 'result_ref'),
    continuation,
  };
}
