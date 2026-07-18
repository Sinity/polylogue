import { useRef } from 'preact/hooks';

export interface FacetOption<Value extends string> {
  value: Value;
  label: string;
  count?: number;
}

function focusButton(container: HTMLDivElement, index: number): void {
  const buttons = Array.from(container.querySelectorAll<HTMLButtonElement>('[data-facet-chip]'));
  if (buttons.length === 0) return;
  const normalized = ((index % buttons.length) + buttons.length) % buttons.length;
  for (const button of buttons) button.tabIndex = -1;
  const next = buttons[normalized];
  if (next) {
    next.tabIndex = 0;
    next.focus();
  }
}

export function FacetChipGroup<Value extends string>({
  label,
  options,
  selected,
  onChange,
}: {
  label: string;
  options: ReadonlyArray<FacetOption<Value>>;
  selected: ReadonlySet<Value>;
  onChange: (next: ReadonlySet<Value>) => void;
}) {
  const groupRef = useRef<HTMLDivElement>(null);

  return (
    <div class="pl-facets" role="group" aria-label={label} ref={groupRef}>
      <span class="pl-facets__label">{label}</span>
      <div class="pl-facets__chips">
        {options.map((option, index) => {
          const active = selected.has(option.value);
          return (
            <button
              key={option.value}
              type="button"
              class="pl-facet-chip"
              data-facet-chip={option.value}
              aria-pressed={active}
              tabIndex={index === 0 ? 0 : -1}
              onClick={() => {
                const next = new Set(selected);
                if (active) next.delete(option.value);
                else next.add(option.value);
                onChange(next);
              }}
              onKeyDown={(event) => {
                const container = groupRef.current;
                if (!container) return;
                if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
                  event.preventDefault();
                  focusButton(container, index + 1);
                } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
                  event.preventDefault();
                  focusButton(container, index - 1);
                } else if (event.key === 'Home') {
                  event.preventDefault();
                  focusButton(container, 0);
                } else if (event.key === 'End') {
                  event.preventDefault();
                  focusButton(container, options.length - 1);
                }
              }}
            >
              <span>{option.label}</span>
              {option.count === undefined ? null : <span class="pl-facet-chip__count">{option.count}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}
