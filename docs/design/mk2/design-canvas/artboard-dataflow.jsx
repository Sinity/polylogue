// Data-flow artboard — how a single conversation moves from provider to surfaces.

const Dataflow = () => (
  <div className="flow">
    <header className="flow__head">
      <Mono>DATA FLOW</Mono>
      <h2>One conversation, end to end</h2>
      <p>From provider event to the pixel a user reads. Each box is a real module; each arrow names the wire format.</p>
    </header>

    <div className="flow__diagram">
      <Lane title="sources">
        <FBox kind="src" name="claude-code tail"  sub="JSONL on disk"/>
        <FBox kind="src" name="codex tail"        sub="WebSocket frames"/>
        <FBox kind="src" name="browser capture"   sub="HAR-shaped POST"/>
        <FBox kind="src" name="manual import"     sub="zip · folder · url"/>
      </Lane>

      <Arrow label="RawArtifact" />

      <Lane title="ingest">
        <FBox kind="ing" name="normalizer"        sub="provider-specific"/>
        <FBox kind="ing" name="dedupe + hash"     sub="content_addr_id"/>
        <FBox kind="ing" name="enrich"            sub="repo · cwd · tool runs"/>
      </Lane>

      <Arrow label="Conversation + MessageRow*" />

      <Lane title="archive">
        <FBox kind="arc" name="sqlite · headers"  sub="ConversationHeader"/>
        <FBox kind="arc" name="sqlite · pages"    sub="MessagePage"/>
        <FBox kind="arc" name="raw blobs"         sub="zstd-compressed"/>
        <FBox kind="arc" name="annotations.db"    sub="user-anchored notes"/>
      </Lane>

      <Arrow label="ConversationQuerySpec → rows" />

      <Lane title="daemon · 127.0.0.1:8765">
        <FBox kind="dmn" name="/api/query"        sub="descriptor table"/>
        <FBox kind="dmn" name="/api/read/:id"     sub="header + pages"/>
        <FBox kind="dmn" name="/api/health"       sub="DaemonStatus"/>
        <FBox kind="dmn" name="/api/derived"      sub="recall · views"/>
      </Lane>

      <Arrow label="JSON over loopback" />

      <Lane title="surfaces">
        <FBox kind="srf" name="web reader"        sub="React · vendored bundle"/>
        <FBox kind="srf" name="CLI · query verbs" sub="parity guaranteed"/>
        <FBox kind="srf" name="TUI cockpit"       sub="textual"/>
        <FBox kind="srf" name="MCP adapter"       sub="separate console_script"/>
      </Lane>
    </div>

    <footer className="flow__notes">
      <span><Mono>invariant</Mono> any surface can render any conversation by id alone — no surface holds state the others can't reconstruct.</span>
      <span><Mono>invariant</Mono> the descriptor table is the only place where filter syntax is defined (#621).</span>
      <span><Mono>invariant</Mono> nothing leaves loopback unless the user explicitly exports.</span>
    </footer>
  </div>
);

const Lane = ({ title, children }) => (
  <div className="lane">
    <header><Mono>{title}</Mono></header>
    <div className="lane__cells">{children}</div>
  </div>
);

const FBox = ({ kind, name, sub }) => (
  <div className={`fbox fbox--${kind}`}>
    <div className="fbox__name">{name}</div>
    <div className="fbox__sub">{sub}</div>
  </div>
);

const Arrow = ({ label }) => (
  <div className="arrow">
    <div className="arrow__line"></div>
    <div className="arrow__head"></div>
    <div className="arrow__label"><Mono>{label}</Mono></div>
  </div>
);

window.Dataflow = Dataflow;
