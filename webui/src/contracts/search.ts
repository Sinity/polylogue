import { isRecord, nullableNumber, nullableString, requiredString } from './runtime';

export const SEARCH_RESULT_LIMIT = 20;

export interface SearchHitMatch {
  readonly rank: number | null;
  readonly message_id: string | null;
  readonly snippet: string | null;
  readonly score_kind: string | null;
  readonly matched_terms: readonly string[];
}

export interface SearchHitSession {
  readonly id: string;
  readonly title: string;
  readonly origin: string;
}

export interface SearchHit {
  readonly session: SearchHitSession;
  readonly match: SearchHitMatch;
}

export interface SearchResult {
  readonly hits: readonly SearchHit[];
  readonly total: number | null;
  readonly next_cursor: string | null;
  readonly retrieval_lane: string;
  readonly exactness: string | null;
}

function parseSearchHit(value: unknown, index: number): SearchHit {
  if (!isRecord(value) || !isRecord(value.session) || !isRecord(value.match)) {
    throw new TypeError(`search hit ${index} is not a session/match pair`);
  }
  const session = value.session;
  const match = value.match;
  const matchedTerms = Array.isArray(match.matched_terms)
    ? match.matched_terms.filter((term): term is string => typeof term === 'string')
    : [];
  return {
    session: {
      id: requiredString(session, 'id'),
      title: requiredString(session, 'title'),
      origin: requiredString(session, 'origin'),
    },
    match: {
      rank: nullableNumber(match, 'rank'),
      message_id: nullableString(match, 'message_id'),
      snippet: nullableString(match, 'snippet'),
      score_kind: nullableString(match, 'score_kind'),
      matched_terms: matchedTerms,
    },
  };
}

/**
 * Parse the ``SearchEnvelope`` returned by ``GET /api/sessions?query=...``.
 */
export function parseSearchResult(value: unknown): SearchResult {
  if (!isRecord(value)) {
    throw new TypeError('response is not a search envelope');
  }
  if (value.ok === false) {
    throw new TypeError(typeof value.detail === 'string' ? value.detail : 'the search request was rejected');
  }
  if (!Array.isArray(value.hits)) {
    throw new TypeError('search envelope hits must be an array');
  }
  return {
    hits: value.hits.map(parseSearchHit),
    total: nullableNumber(value, 'total'),
    next_cursor: nullableString(value, 'next_cursor'),
    retrieval_lane: requiredString(value, 'retrieval_lane'),
    exactness: nullableString(value, 'exactness'),
  };
}
