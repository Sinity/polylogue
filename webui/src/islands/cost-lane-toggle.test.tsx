import { fireEvent, render, screen } from '@testing-library/preact';
import { describe, expect, it } from 'vitest';
import { CostLaneToggle } from './cost-lane-toggle';

function renderTableFixture(): void {
  document.body.innerHTML = `
    <table data-lane-table>
      <tr><td data-lane="provider_reported_usd">$1</td><td data-lane="api_equivalent_usd">$2</td></tr>
    </table>
  `;
}

describe('CostLaneToggle', () => {
  it('sets data-lane-focus on every [data-lane-table] when a lane button is pressed', () => {
    renderTableFixture();
    render(<CostLaneToggle />);

    fireEvent.click(screen.getByRole('button', { name: 'API-equivalent' }));

    const table = document.querySelector('[data-lane-table]');
    expect(table).toHaveAttribute('data-lane-focus', 'api_equivalent_usd');
    expect(screen.getByRole('button', { name: 'API-equivalent' })).toHaveClass('is-active');
  });

  it('clears the focus attribute when "All lanes" is pressed again', () => {
    renderTableFixture();
    render(<CostLaneToggle />);

    fireEvent.click(screen.getByRole('button', { name: 'Catalog-priced' }));
    fireEvent.click(screen.getByRole('button', { name: 'All lanes' }));

    const table = document.querySelector('[data-lane-table]');
    expect(table).not.toHaveAttribute('data-lane-focus');
    expect(screen.getByRole('button', { name: 'All lanes' })).toHaveClass('is-active');
  });
});
