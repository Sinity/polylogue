import type { ComponentChildren, JSX } from 'preact';
import { useRef } from 'preact/hooks';

import { Button } from './layout';
import { EmptyState, UnknownState } from './states';
import type { Density } from './types';

export interface DataColumn<Row> {
  id: string;
  header: string;
  cell: (row: Row) => ComponentChildren;
  align?: 'start' | 'end';
  priority?: 'essential' | 'secondary';
}

export interface DataTableContinuation {
  hasMore: boolean;
  loading: boolean;
  error?: string | null;
  onLoadMore: () => void | Promise<void>;
  label?: string;
}

function tableRows(table: HTMLTableElement): HTMLTableRowElement[] {
  return Array.from(table.querySelectorAll<HTMLTableRowElement>('tbody tr[data-table-row]'));
}

function focusAt(table: HTMLTableElement, index: number): void {
  const rows = tableRows(table);
  if (rows.length === 0) return;
  const normalized = Math.max(0, Math.min(index, rows.length - 1));
  for (const row of rows) row.tabIndex = -1;
  const next = rows[normalized];
  if (next) {
    next.tabIndex = 0;
    next.focus();
  }
}

export function DataTable<Row>({
  caption,
  rows,
  columns,
  rowKey,
  density = 'comfortable',
  onRowActivate,
  continuation,
  absence = 'empty',
  absenceDescription = 'No rows match this view.',
}: {
  caption: string;
  rows: ReadonlyArray<Row>;
  columns: ReadonlyArray<DataColumn<Row>>;
  rowKey: (row: Row) => string;
  density?: Density;
  onRowActivate?: (row: Row) => void;
  continuation?: DataTableContinuation;
  absence?: 'empty' | 'unknown';
  absenceDescription?: string;
}) {
  const tableRef = useRef<HTMLTableElement>(null);

  if (rows.length === 0) {
    return absence === 'unknown' ? (
      <UnknownState description={absenceDescription} />
    ) : (
      <EmptyState description={absenceDescription} />
    );
  }

  const onKeyDown: JSX.KeyboardEventHandler<HTMLTableRowElement> = (event) => {
    const table = tableRef.current;
    if (!table) return;
    const rowsInTable = tableRows(table);
    const index = rowsInTable.indexOf(event.currentTarget);
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      focusAt(table, index + 1);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      focusAt(table, index - 1);
    } else if (event.key === 'Home') {
      event.preventDefault();
      focusAt(table, 0);
    } else if (event.key === 'End') {
      event.preventDefault();
      focusAt(table, rowsInTable.length - 1);
    } else if (event.key === 'Enter' && onRowActivate) {
      event.preventDefault();
      const rowIndex = Number(event.currentTarget.dataset.rowIndex);
      const row = rows[rowIndex];
      if (row) onRowActivate(row);
    }
  };

  return (
    <div class="pl-table-region" data-density={density}>
      <div class="pl-table-scroll" tabIndex={0} role="region" aria-label={`${caption}; horizontally scrollable`}>
        <table class="pl-data-table" ref={tableRef}>
          <caption>{caption}</caption>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.id} scope="col" data-align={column.align ?? 'start'} data-priority={column.priority}>
                  {column.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr
                key={rowKey(row)}
                data-table-row={rowKey(row)}
                data-row-index={rowIndex}
                tabIndex={rowIndex === 0 ? 0 : -1}
                onKeyDown={onKeyDown}
                onDblClick={() => onRowActivate?.(row)}
              >
                {columns.map((column, columnIndex) => {
                  const Cell = columnIndex === 0 ? 'th' : 'td';
                  return (
                    <Cell
                      key={column.id}
                      scope={columnIndex === 0 ? 'row' : undefined}
                      data-align={column.align ?? 'start'}
                      data-priority={column.priority}
                    >
                      {column.cell(row)}
                    </Cell>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {continuation ? (
        <div class="pl-table-continuation" aria-live="polite">
          {continuation.error ? <p class="pl-form-error">{continuation.error}</p> : null}
          {continuation.hasMore ? (
            <Button disabled={continuation.loading} onClick={() => void continuation.onLoadMore()}>
              {continuation.loading ? 'Loading more…' : (continuation.label ?? 'Load more')}
            </Button>
          ) : (
            <p class="pl-table-end">End of results</p>
          )}
        </div>
      ) : null}
    </div>
  );
}
