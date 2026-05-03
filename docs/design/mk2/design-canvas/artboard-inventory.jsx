// Component inventory artboard — every concrete component the implementation will need.

const Inventory = () => {
  const groups = [
    { title: 'shared kernel · models', items: [
      ['ConversationQuerySpec',   'filter set produced from CLI flags or web facets'],
      ['RootModeRequest',          'parsed query + verb + outputs (CLI parity contract)'],
      ['ConversationHeader',       'header-only read · #570'],
      ['MessagePage',              '{ id, limit, offset, role?, type? } → MessageRow[]'],
      ['RawArtifactPage',          'raw provider payload page (CLI verb `raw`)'],
      ['DaemonStatus',             'pid · uptime · components[] · overall'],
      ['ComponentHealth',          'name · state · lag_s · detail'],
      ['QueryDescriptor',          'name · cli flag · validator · completion source'],
    ]},
    { title: 'CLI · interactive', items: [
      ['QueryFirstGroup',          'click group · intercepts argv'],
      ['query_verbs/*',            'list count stats show open messages raw bulk-export delete'],
      ['shell_completion_values',  'archive-backed: providers · ids · repos · cwd · tags · tools'],
      ['filter_picker',            'evolved into `polylogue select` · descriptor-driven'],
      ['doctor (target=daemon)',   'reads daemon /api/health · prints overall + components'],
    ]},
    { title: 'daemon · service', items: [
      ['polylogued/serve.py',      'uvicorn launcher · pid file · 127.0.0.1 only'],
      ['daemon/app.py',            'ASGI · query/read/status/derived endpoints'],
      ['daemon/web/',              'vendored static reader bundle'],
      ['daemon/sources/live.py',   'wraps existing claude-code/codex live ingest'],
      ['daemon/sources/receiver.py','browser-capture receiver (slice 2)'],
      ['daemon/status.py',         'aggregates ComponentHealth from all subsystems'],
    ]},
    { title: 'web reader · UI', items: [
      ['<Facets>',                 'provider · repo · has · saved views'],
      ['<ResultList>',             'paginated · keyboard nav · multi-select'],
      ['<Reader>',                 'header + messages stream + actions bar'],
      ['<SidePanel>',              'tabs: messages outline · raw · provenance'],
      ['<StatusStrip>',            'reuses DaemonStatus (1Hz poll)'],
      ['<Palette>',                'cmd-K · runs the same verbs as the CLI'],
      ['<RecallBuilder>',          'multi-select → bundle (slice 3)'],
    ]},
    { title: 'TUI · operator cockpit', items: [
      ['Dashboard panel',          'archive · live · capture · diagnostics · next'],
      ['Browser panel',            'fuzzy result list · open in web'],
      ['Search panel',             'live query · same descriptors'],
      ['StatCard widget',          'value + delta + sparkline'],
      ['ProviderBar widget',       'distribution bar · 16-cell'],
    ]},
    { title: 'mcp · adapter', items: [
      ['polylogue-mcp executable', 'console_script · separate from `polylogue`'],
      ['mcp.tools.search',         'wraps ConversationQuerySpec'],
      ['mcp.tools.read',           'wraps ConversationHeader + MessagePage'],
      ['role gate',                'read | write | admin · per #570 sanitized errors'],
    ]},
  ];
  return (
    <div className="inventory">
      <header><Mono>COMPONENT INVENTORY</Mono><h2>What ships, where it lives, what it consumes</h2></header>
      <div className="inv__grid">
        {groups.map(g => (
          <section key={g.title} className="inv__col">
            <header><Mono>{g.title}</Mono></header>
            <ul>
              {g.items.map(([name, desc]) => (
                <li key={name}>
                  <Mono className="inv__name">{name}</Mono>
                  <span className="inv__desc">{desc}</span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
};

window.Inventory = Inventory;
