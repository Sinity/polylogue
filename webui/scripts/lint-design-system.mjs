import { readdir, readFile } from 'node:fs/promises';
import { extname, join, relative } from 'node:path';
import ts from 'typescript';

const root = new URL('..', import.meta.url).pathname;
const sourceRoot = join(root, 'src');
const restrictedFixtureTags = new Set([
  'button',
  'code',
  'details',
  'pre',
  'summary',
  'svg',
  'table',
  'tbody',
  'td',
  'th',
  'thead',
  'tr',
]);
const failures = [];

async function filesUnder(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  }));
  return nested.flat();
}

function lineAndColumn(sourceFile, position) {
  const point = sourceFile.getLineAndCharacterOfPosition(position);
  return `${point.line + 1}:${point.character + 1}`;
}

for (const path of await filesUnder(sourceRoot)) {
  const extension = extname(path);
  if (!['.ts', '.tsx', '.css'].includes(extension)) continue;
  const name = relative(root, path);
  const text = await readFile(path, 'utf8');

  if (extension === '.css') {
    if (!name.includes('src/generated/') && /#[0-9a-fA-F]{3,8}\b/.test(text)) {
      failures.push(`${name}: raw hex color; add a generated semantic token instead`);
    }
    continue;
  }

  const sourceFile = ts.createSourceFile(
    name,
    text,
    ts.ScriptTarget.Latest,
    true,
    extension === '.tsx' ? ts.ScriptKind.TSX : ts.ScriptKind.TS,
  );
  const isVertical = name.includes('src/fixture/') || name.includes('src/verticals/');

  function visit(node) {
    if (ts.isJsxAttribute(node) && node.name.text === 'dangerouslySetInnerHTML') {
      failures.push(`${name}:${lineAndColumn(sourceFile, node.pos)}: dangerouslySetInnerHTML is forbidden`);
    }
    if (isVertical && (ts.isJsxOpeningElement(node) || ts.isJsxSelfClosingElement(node))) {
      const tag = node.tagName.getText(sourceFile);
      if (restrictedFixtureTags.has(tag)) {
        failures.push(
          `${name}:${lineAndColumn(sourceFile, node.pos)}: raw <${tag}> bypasses the design-system component`,
        );
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
}

if (failures.length > 0) {
  console.error('WebUI design-system lint failed:');
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exitCode = 1;
} else {
  console.log('WebUI design-system lint: OK');
}
