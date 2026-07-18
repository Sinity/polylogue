import { isRecord, nullableString, requiredNumber, requiredString } from './runtime';

export const SESSION_LIST_LIMIT = 20;

export interface SessionListRow {
  readonly id: string;
  readonly title: string;
  readonly origin: string;
  readonly date: string | null;
  readonly message_count: number;
  readonly word_count: number;
  readonly repo: string | null;
}

export interface SessionListPage {
  readonly items: readonly SessionListRow[];
  readonly total: number;
  readonly limit: number;
  readonly offset: number;
}

function parseSessionListRow(value: unknown, index: number): SessionListRow {
  if (!isRecord(value)) {
    throw new TypeError(`session list item ${index} is not an object`);
  }
  return {
    id: requiredString(value, 'id'),
    title: requiredString(value, 'title'),
    origin: requiredString(value, 'origin'),
    date: nullableString(value, 'date'),
    message_count: requiredNumber(value, 'message_count'),
    word_count: requiredNumber(value, 'word_count'),
    repo: nullableString(value, 'repo'),
  };
}

export function parseSessionListPage(value: unknown): SessionListPage {
  if (!isRecord(value)) {
    throw new TypeError('response is not a session list page');
  }
  if (!Array.isArray(value.items)) {
    throw new TypeError('session list response items must be an array');
  }
  return {
    items: value.items.map(parseSessionListRow),
    total: requiredNumber(value, 'total'),
    limit: requiredNumber(value, 'limit'),
    offset: requiredNumber(value, 'offset'),
  };
}
