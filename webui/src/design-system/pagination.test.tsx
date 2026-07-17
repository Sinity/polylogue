import { useState } from 'preact/hooks';
import { act } from 'preact/test-utils';
import { vi } from 'vitest';

import { useContinuationPaging, type ContinuationPage } from './pagination';
import { click, renderComponent } from '../test/render';

interface HarnessProps {
  loader: (cursor: string, signal: AbortSignal) => Promise<ContinuationPage<string, string>>;
}

function PagingHarness({ loader }: HarnessProps) {
  const paging = useContinuationPaging({ initialItems: ['first'], initialCursor: 'page-2', loadPage: loader });
  const [resetCount, setResetCount] = useState(0);
  return (
    <div>
      <output data-items>{paging.items.join('|')}</output>
      <output data-error>{paging.error ?? ''}</output>
      <button type="button" data-load onClick={() => void paging.loadMore()}>load</button>
      <button
        type="button"
        data-reset
        onClick={() => {
          paging.reset([`reset-${resetCount + 1}`], null);
          setResetCount((value) => value + 1);
        }}
      >reset</button>
    </div>
  );
}

describe('continuation paging hook', () => {
  it('appends a successful page and closes the continuation', async () => {
    const loader = vi.fn(async () => ({ items: ['second'], nextCursor: null }));
    const container = renderComponent(<PagingHarness loader={loader} />);

    click(container.querySelector('[data-load]')!);
    await act(async () => { await Promise.resolve(); });

    expect(container.querySelector('[data-items]')?.textContent).toBe('first|second');
    expect(loader).toHaveBeenCalledOnce();
  });

  it('aborts and ignores a stale page after reset', async () => {
    let resolvePage: ((page: ContinuationPage<string, string>) => void) | undefined;
    let observedSignal: AbortSignal | undefined;
    const loader = vi.fn((_cursor: string, signal: AbortSignal) => {
      observedSignal = signal;
      return new Promise<ContinuationPage<string, string>>((resolve) => { resolvePage = resolve; });
    });
    const container = renderComponent(<PagingHarness loader={loader} />);

    click(container.querySelector('[data-load]')!);
    click(container.querySelector('[data-reset]')!);
    expect(observedSignal?.aborted).toBe(true);
    await act(async () => {
      resolvePage?.({ items: ['stale'], nextCursor: null });
      await Promise.resolve();
    });

    expect(container.querySelector('[data-items]')?.textContent).toBe('reset-1');
    expect(container.querySelector('[data-items]')?.textContent).not.toContain('stale');
  });
});
