import type { ComponentChildren, JSX } from 'preact';

export type ClassName = string | undefined;
export type Space = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8;
export type Density = 'comfortable' | 'compact';
export type HonestStateKind = 'loading' | 'empty' | 'unknown' | 'degraded' | 'error';
export type VerticalId = 'webui-02' | 'webui-03' | 'webui-04' | 'webui-05' | 'webui-06';
export type VerticalState = 'ready' | HonestStateKind;

export interface CommonProps {
  children?: ComponentChildren;
  className?: ClassName;
}

export type ButtonProps = Omit<JSX.ButtonHTMLAttributes<HTMLButtonElement>, 'className'> & {
  className?: string;
  variant?: 'primary' | 'secondary' | 'quiet';
};
