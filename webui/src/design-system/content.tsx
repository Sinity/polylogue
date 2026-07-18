import type { ComponentChildren } from 'preact';
import { useId } from 'preact/hooks';

import type { EvidenceState } from '../generated/contracts';
import { EvidenceStateBadge } from './badges';
import { Stack } from './layout';

export function CodeBlock({
  code,
  language = 'text',
  caption,
}: {
  code: string;
  language?: string;
  caption?: string;
}) {
  return (
    <figure class="pl-code-block">
      {caption ? <figcaption>{caption}</figcaption> : null}
      <pre tabIndex={0} aria-label={`${language} code`}>
        <code data-language={language}>{code}</code>
      </pre>
    </figure>
  );
}

function diffKind(line: string): 'added' | 'removed' | 'context' {
  if (line.startsWith('+') && !line.startsWith('+++')) return 'added';
  if (line.startsWith('-') && !line.startsWith('---')) return 'removed';
  return 'context';
}

export function DiffBlock({ diff, caption = 'Diff' }: { diff: string; caption?: string }) {
  const lines = diff.split('\n');
  return (
    <figure class="pl-code-block pl-diff-block">
      <figcaption>{caption}</figcaption>
      <pre tabIndex={0} aria-label={caption}>
        <code>
          {lines.map((line, index) => {
            const kind = diffKind(line);
            return (
              <span class="pl-diff-line" data-diff-kind={kind} key={`${index}:${line}`}>
                <span class="pl-sr-only">{kind === 'added' ? 'Added: ' : kind === 'removed' ? 'Removed: ' : ''}</span>
                {line || ' '}
                {index === lines.length - 1 ? '' : '\n'}
              </span>
            );
          })}
        </code>
      </pre>
    </figure>
  );
}

export interface TranscriptMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool' | 'unknown';
  author: string;
  timestamp?: string;
  body: ComponentChildren;
  evidence?: EvidenceState;
}

export function TranscriptBlock({ messages, label = 'Transcript' }: { messages: ReadonlyArray<TranscriptMessage>; label?: string }) {
  const headingId = useId();
  return (
    <section class="pl-transcript" aria-labelledby={headingId}>
      <h2 id={headingId}>{label}</h2>
      <ol>
        {messages.map((message) => (
          <li key={message.id} class="pl-transcript-message" data-role={message.role}>
            <header>
              <Stack space={1}>
                <span class="pl-transcript-author">{message.author}</span>
                {message.timestamp ? <time dateTime={message.timestamp}>{message.timestamp}</time> : null}
              </Stack>
              {message.evidence ? <EvidenceStateBadge state={message.evidence} /> : null}
            </header>
            <div class="pl-transcript-body">{message.body}</div>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function Disclosure({ summary, children, open = false }: { summary: string; children: ComponentChildren; open?: boolean }) {
  return (
    <details class="pl-disclosure" open={open}>
      <summary>{summary}</summary>
      <div class="pl-disclosure__body">{children}</div>
    </details>
  );
}
