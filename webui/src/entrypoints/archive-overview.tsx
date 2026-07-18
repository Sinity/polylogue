import { hydrate, render } from 'preact';
import { ArchiveOverviewIsland } from '../islands/archive-overview';
import '../styles.css';

interface ArchiveOverviewBootstrap {
  readonly continuation: string | null;
}

function readBootstrap(): ArchiveOverviewBootstrap | undefined {
  const node = document.getElementById('archive-overview-bootstrap');
  if (node === null) {
    return undefined;
  }
  try {
    const value: unknown = JSON.parse(node.textContent ?? '');
    if (
      typeof value === 'object' &&
      value !== null &&
      'continuation' in value &&
      (typeof (value as { continuation: unknown }).continuation === 'string' ||
        (value as { continuation: unknown }).continuation === null)
    ) {
      return { continuation: (value as { continuation: string | null }).continuation };
    }
  } catch {
    return undefined;
  }
  return undefined;
}

const root = document.getElementById('archive-overview-island');
if (root !== null) {
  const bootstrap = readBootstrap();
  const island =
    bootstrap === undefined ? (
      <ArchiveOverviewIsland />
    ) : (
      <ArchiveOverviewIsland initialContinuation={bootstrap.continuation} />
    );
  if (bootstrap !== undefined && root.childNodes.length > 0) {
    hydrate(island, root);
  } else {
    render(island, root);
  }
}
