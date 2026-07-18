import type { ComponentChildren, CSSProperties, JSX } from 'preact';

import type { ButtonProps, CommonProps, Space } from './types';
import { WEBUI_VERTICAL_CONTRACT, verticalHeadingId, type VerticalFrameProps } from './vertical-contract';

function classes(...values: Array<string | false | null | undefined>): string {
  return values.filter(Boolean).join(' ');
}

function spaceStyle(name: string, space: Space | undefined): CSSProperties | undefined {
  return space === undefined ? undefined : ({ [name]: `var(--pl-space-${space})` } as CSSProperties);
}

export function SkipLink({ target = 'main-content' }: { target?: string }) {
  return (
    <a class="pl-skip-link" href={`#${target}`}>
      Skip to main content
    </a>
  );
}

export function Stack({ children, className, space = 4 }: CommonProps & { space?: Space }) {
  return (
    <div class={classes('pl-stack', className)} style={spaceStyle('--pl-stack-space', space)}>
      {children}
    </div>
  );
}

export function Cluster({ children, className, space = 3 }: CommonProps & { space?: Space }) {
  return (
    <div class={classes('pl-cluster', className)} style={spaceStyle('--pl-cluster-space', space)}>
      {children}
    </div>
  );
}

export function Grid({ children, className, min = '18rem', space = 4 }: CommonProps & { min?: string; space?: Space }) {
  const style = {
    '--pl-grid-min': min,
    '--pl-grid-space': `var(--pl-space-${space})`,
  } as CSSProperties;
  return (
    <div class={classes('pl-grid', className)} style={style}>
      {children}
    </div>
  );
}

export function Surface({ children, className, as = 'section' }: CommonProps & { as?: 'div' | 'section' | 'article' }) {
  const Element = as;
  return <Element class={classes('pl-surface', className)}>{children}</Element>;
}

export function Button({ className, variant = 'secondary', type = 'button', ...props }: ButtonProps) {
  return <button {...props} type={type} class={classes('pl-button', `pl-button--${variant}`, className)} />;
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ComponentChildren;
}) {
  return (
    <header class="pl-page-header">
      <div>
        {eyebrow ? <p class="pl-eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
        {description ? <p class="pl-page-description">{description}</p> : null}
      </div>
      {actions ? <div class="pl-page-actions">{actions}</div> : null}
    </header>
  );
}

export function VerticalFrame({
  id,
  state,
  title,
  description,
  actions,
  children,
}: VerticalFrameProps) {
  const headingId = verticalHeadingId(id);
  return (
    <main
      id="main-content"
      tabIndex={-1}
      class="pl-main"
      data-webui-contract={WEBUI_VERTICAL_CONTRACT.version}
      data-webui-vertical={id}
      data-state={state}
      aria-labelledby={headingId}
    >
      <header class="pl-page-header">
        <div>
          <p class="pl-eyebrow">{id}</p>
          <h1 id={headingId}>{title}</h1>
          {description ? <p class="pl-page-description">{description}</p> : null}
        </div>
        {actions ? <div class="pl-page-actions">{actions}</div> : null}
      </header>
      {children}
    </main>
  );
}

export function SearchField({
  label,
  name = 'q',
  defaultValue,
  action = '/search',
  placeholder,
}: {
  label: string;
  name?: string;
  defaultValue?: string;
  action?: string;
  placeholder?: string;
}) {
  return (
    <form class="pl-search" method="get" action={action} role="search" aria-label={label}>
      <label for={`search-${name}`}>{label}</label>
      <div class="pl-search__controls">
        <input
          id={`search-${name}`}
          name={name}
          type="search"
          defaultValue={defaultValue}
          placeholder={placeholder}
          autoComplete="off"
        />
        <Button type="submit" variant="primary">
          Search
        </Button>
      </div>
    </form>
  );
}

export function VisuallyHidden({ children }: { children: ComponentChildren }) {
  return <span class="pl-sr-only">{children}</span>;
}

export function ExternalLink(props: JSX.HTMLAttributes<HTMLAnchorElement>) {
  return <a {...props} rel="noreferrer" />;
}
