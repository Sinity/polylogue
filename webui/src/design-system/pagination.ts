import { useCallback, useEffect, useRef, useState } from 'preact/hooks';

export interface ContinuationPage<Item, Cursor> {
  items: ReadonlyArray<Item>;
  nextCursor: Cursor | null;
}

export type ContinuationLoader<Item, Cursor> = (
  cursor: Cursor,
  signal: AbortSignal,
) => Promise<ContinuationPage<Item, Cursor>>;

export interface ContinuationPaging<Item, Cursor> {
  items: ReadonlyArray<Item>;
  cursor: Cursor | null;
  hasMore: boolean;
  loading: boolean;
  error: string | null;
  loadMore: () => Promise<void>;
  reset: (items: ReadonlyArray<Item>, cursor: Cursor | null) => void;
}

export function useContinuationPaging<Item, Cursor>({
  initialItems,
  initialCursor,
  loadPage,
}: {
  initialItems: ReadonlyArray<Item>;
  initialCursor: Cursor | null;
  loadPage: ContinuationLoader<Item, Cursor>;
}): ContinuationPaging<Item, Cursor> {
  const [items, setItems] = useState<ReadonlyArray<Item>>(initialItems);
  const [cursor, setCursor] = useState<Cursor | null>(initialCursor);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const controller = useRef<AbortController | null>(null);

  useEffect(() => () => controller.current?.abort(), []);

  const reset = useCallback((nextItems: ReadonlyArray<Item>, nextCursor: Cursor | null) => {
    generation.current += 1;
    controller.current?.abort();
    controller.current = null;
    setItems(nextItems);
    setCursor(nextCursor);
    setLoading(false);
    setError(null);
  }, []);

  const loadMore = useCallback(async () => {
    if (cursor === null || loading) return;
    const requestGeneration = generation.current;
    const request = new AbortController();
    controller.current?.abort();
    controller.current = request;
    setLoading(true);
    setError(null);
    try {
      const page = await loadPage(cursor, request.signal);
      if (request.signal.aborted || requestGeneration !== generation.current) return;
      setItems((current) => [...current, ...page.items]);
      setCursor(page.nextCursor);
    } catch (caught: unknown) {
      if (request.signal.aborted || requestGeneration !== generation.current) return;
      setError(caught instanceof Error ? caught.message : 'Continuation request failed');
    } finally {
      if (!request.signal.aborted && requestGeneration === generation.current) setLoading(false);
    }
  }, [cursor, loadPage, loading]);

  return {
    items,
    cursor,
    hasMore: cursor !== null,
    loading,
    error,
    loadMore,
    reset,
  };
}
