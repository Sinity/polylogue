// Personas + jobs — who this is for, what they hire it to do.

const Personas = () => {
  const people = [
    {
      id: 'maintainer', label: 'open-source maintainer',
      sub: 'works across 12 repos in claude-code + codex',
      jobs: [
        'find the conversation where I worked through that bug last month',
        'audit what tools the agent actually called on a PR',
        'turn three threads into a recall pack for the design doc',
      ],
      pains: [
        'browser tabs forget · CLI history lies · provider exports differ',
        'no single place to search across providers',
      ],
      shape: 'CLI-first · TUI cockpit · web reader for sharing',
    },
    {
      id: 'researcher', label: 'AI researcher',
      sub: 'compares model behavior across Claude / GPT / open weights',
      jobs: [
        'pull every conversation tagged eval=v3 and diff outputs',
        'export a recall pack to a paper appendix',
        'feed an MCP agent over the archive without re-implementing search',
      ],
      pains: [
        'each provider has its own export shape',
        'no per-tool / per-repo facets in any vendor UI',
      ],
      shape: 'web reader for visual diff · MCP adapter · CLI for batch',
    },
    {
      id: 'writer', label: 'writer · second-brain user',
      sub: 'turns long conversations into essays',
      jobs: [
        'find the thread where we landed on the right framing',
        'annotate a specific message and revisit it next week',
        'pull the cited messages into a draft without copy-paste hell',
      ],
      pains: [
        'losing the thread literally — which conversation was it in?',
        'no durable annotation layer over chat',
      ],
      shape: 'web reader · annotations · saved views',
    },
    {
      id: 'team', label: 'small team lead',
      sub: '4-person team, mostly individual conversations',
      jobs: [
        'ship a recall pack that captures how we made a decision',
        'audit which agent runs touched production paths',
        'keep all of this on the team\u2019s laptops, not in someone\u2019s vendor account',
      ],
      pains: [
        'compliance pressure on cloud-hosted chat history',
        'institutional memory evaporates with people',
      ],
      shape: 'recall packs · doctor · MCP for review tooling',
    },
  ];

  return (
    <div className="personas">
      <header className="personas__head">
        <Mono>PERSONAS · JOBS-TO-BE-DONE</Mono>
        <h2>Who hires this, and what for</h2>
        <p>One archive, many surfaces — because the same person uses different surfaces for different jobs in the same week.</p>
      </header>
      <div className="personas__grid">
        {people.map(p => (
          <article key={p.id} className="persona">
            <header>
              <div className="persona__sigil"><Mono>{p.id.slice(0,2).toUpperCase()}</Mono></div>
              <div>
                <h3>{p.label}</h3>
                <Mono>{p.sub}</Mono>
              </div>
            </header>
            <section>
              <h4><Mono>jobs</Mono></h4>
              <ul>{p.jobs.map((j,i) => <li key={i}>{j}</li>)}</ul>
            </section>
            <section>
              <h4><Mono>pains</Mono></h4>
              <ul>{p.pains.map((j,i) => <li key={i}>{j}</li>)}</ul>
            </section>
            <footer><Mono>shape</Mono> <span>{p.shape}</span></footer>
          </article>
        ))}
      </div>
    </div>
  );
};

window.Personas = Personas;
