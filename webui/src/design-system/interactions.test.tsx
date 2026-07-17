import { useState } from 'preact/hooks';
import { act } from 'preact/test-utils';

import { DataTable, type DataColumn } from './data-table';
import { FacetChipGroup } from './facets';
import { ThemeToggle } from './theme';
import { click, keydown, renderComponent } from '../test/render';

function FacetHarness() {
  const [selected, setSelected] = useState<ReadonlySet<'exact' | 'unknown'>>(new Set());
  return (
    <FacetChipGroup
      label="Evidence state"
      options={[
        { value: 'exact', label: 'Exact', count: 2 },
        { value: 'unknown', label: 'Unknown', count: 1 },
      ]}
      selected={selected}
      onChange={setSelected}
    />
  );
}

interface Row { id: string; title: string; count: number }
const columns: ReadonlyArray<DataColumn<Row>> = [
  { id: 'title', header: 'Title', cell: (row) => row.title },
  { id: 'count', header: 'Count', align: 'end', cell: (row) => row.count },
];

describe('keyboard and theme interactions', () => {
  it('uses roving keyboard focus and aria-pressed for facet chips', () => {
    const container = renderComponent(<FacetHarness />);
    const buttons = Array.from(container.querySelectorAll<HTMLButtonElement>('[data-facet-chip]'));
    buttons[0]?.focus();
    keydown(buttons[0]!, 'ArrowRight');
    expect(document.activeElement).toBe(buttons[1]);
    expect(buttons[0]?.tabIndex).toBe(-1);
    expect(buttons[1]?.tabIndex).toBe(0);

    click(buttons[1]!);
    expect(buttons[1]?.getAttribute('aria-pressed')).toBe('true');
    keydown(buttons[1]!, 'Home');
    expect(document.activeElement).toBe(buttons[0]);
    expect(buttons[0]?.tabIndex).toBe(0);
    expect(buttons[1]?.tabIndex).toBe(-1);
  });

  it('moves through data rows and activates the focused row with Enter', () => {
    const activated: string[] = [];
    const container = renderComponent(
      <DataTable
        caption="Fixture rows"
        rows={[{ id: 'a', title: 'Alpha', count: 1 }, { id: 'b', title: 'Beta', count: 2 }]}
        columns={columns}
        rowKey={(row) => row.id}
        onRowActivate={(row) => activated.push(row.id)}
      />,
    );
    const rows = Array.from(container.querySelectorAll<HTMLTableRowElement>('[data-table-row]'));
    rows[0]?.focus();
    keydown(rows[0]!, 'ArrowDown');
    expect(document.activeElement).toBe(rows[1]);
    keydown(rows[1]!, 'Enter');
    expect(activated).toEqual(['b']);
    keydown(rows[1]!, 'Home');
    expect(document.activeElement).toBe(rows[0]);
  });

  it('cycles system, light, and dark preferences explicitly', async () => {
    const container = renderComponent(<ThemeToggle />);
    await act(async () => undefined);
    const button = container.querySelector('button')!;

    expect(button.textContent).toContain('system');
    click(button);
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(window.localStorage.getItem('polylogue.webui.theme')).toBe('light');
    click(button);
    expect(document.documentElement.dataset.theme).toBe('dark');
    click(button);
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false);
    expect(window.localStorage.getItem('polylogue.webui.theme')).toBeNull();
  });
});
