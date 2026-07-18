import { render, type ComponentChildren } from 'preact';
import { act } from 'preact/test-utils';

export function renderComponent(children: ComponentChildren): HTMLElement {
  const container = document.createElement('div');
  document.body.append(container);
  act(() => render(<>{children}</>, container));
  return container;
}

export function click(element: Element): void {
  act(() => { element.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })); });
}

export function keydown(element: Element, key: string): void {
  act(() => { element.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true })); });
}
