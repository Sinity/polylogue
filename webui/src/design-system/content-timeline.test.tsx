import { CodeBlock, DiffBlock, Disclosure } from './content';
import { Sparkline, Timeline } from './timeline';
import { renderComponent } from '../test/render';

describe('evidence content primitives', () => {
  it('renders code as text and does not create executable markup', () => {
    const hostile = '<script>window.compromised = true</script>';
    const container = renderComponent(<CodeBlock code={hostile} language="html" caption="Escaped fixture" />);

    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('code')?.textContent).toBe(hostile);
    expect(container.querySelector('pre')?.tabIndex).toBe(0);
  });

  it('adds non-color diff labels and preserves native disclosure semantics', () => {
    const container = renderComponent(
      <Disclosure summary="Changes" open>
        <DiffBlock diff={'- old\n+ new\n context'} />
      </Disclosure>,
    );

    expect(container.querySelector('details')?.open).toBe(true);
    expect(container.querySelector('summary')?.textContent).toBe('Changes');
    expect(container.textContent).toContain('Removed:');
    expect(container.textContent).toContain('Added:');
  });

  it('labels timeline and SVG values without relying on visual position', () => {
    const container = renderComponent(
      <div>
        <Timeline items={[{ id: 'one', at: '2026-06-12T09:00:00Z', label: 'Captured', state: 'exact' }]} />
        <Sparkline values={[1, 3, 2]} label="Fixture trend" />
      </div>,
    );

    expect(container.querySelector('ol time')?.getAttribute('datetime')).toBe('2026-06-12T09:00:00Z');
    expect(container.querySelector('svg')?.getAttribute('role')).toBe('img');
    expect(container.querySelector('title')?.textContent).toBe('Fixture trend');
    expect(container.querySelector('desc')?.textContent).toContain('1, 3, 2');
  });
});
