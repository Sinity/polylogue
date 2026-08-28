import { PUBLIC_ORIGINS } from '../generated/contracts';
import { EvidenceStateBadge, OriginBadge, UnknownOriginBadge } from './badges';
import { EmptyState, UnknownState } from './states';
import { renderComponent } from '../test/render';

describe('public badge and honest-state contracts', () => {
  it('renders every generated public Origin token without promoting unknown-export', () => {
    const container = renderComponent(
      <div>
        {PUBLIC_ORIGINS.map((origin) => <OriginBadge key={origin} origin={origin} />)}
        <UnknownOriginBadge />
      </div>,
    );

    // The count is the generator's, not a literal: `render webui-design-system`
    // owns the roster, so a hardcoded number only ever fossilises a past run.
    expect(PUBLIC_ORIGINS.length).toBeGreaterThan(0);
    expect(PUBLIC_ORIGINS).not.toContain('unknown-export');
    expect(container.querySelectorAll('[data-origin]')).toHaveLength(PUBLIC_ORIGINS.length);
    expect(container.querySelector('[data-origin-state="unknown"]')?.textContent).toContain('Unknown origin');
  });

  it('pairs every evidence color with a symbol and readable state text', () => {
    const container = renderComponent(<EvidenceStateBadge state="degraded" qualifiedBy="one dependency failed" />);
    const badge = container.querySelector('[data-evidence-state="degraded"]');

    expect(badge?.textContent).toContain('!');
    expect(badge?.textContent).toContain('Degraded');
    expect(badge?.textContent).toContain('one dependency failed');
  });

  it('keeps known empty and unknown as distinct DOM contracts', () => {
    const container = renderComponent(
      <div>
        <EmptyState description="The completed query returned zero rows." />
        <UnknownState description="The source was not consulted." />
      </div>,
    );

    const empty = container.querySelector('[data-honest-state="empty"]');
    const unknown = container.querySelector('[data-honest-state="unknown"]');
    expect(empty).not.toBeNull();
    expect(unknown).not.toBeNull();
    expect(empty?.textContent).toContain('zero rows');
    expect(unknown?.textContent).toContain('not consulted');
  });
});
