// Top-level app — assembles the design canvas with five artboards.

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "tone": "graphite",
  "accent": "slate-cyan"
}/*EDITMODE-END*/;

const App = () => {
  const [tweaks, setTweak] = window.useTweaks(TWEAK_DEFAULTS);

  React.useEffect(() => {
    const r = document.documentElement;
    const tones = {
      graphite: { '--bg-0': '#0b0c0e', '--bg-1': '#111317', '--bg-2': '#161a1f' },
      slate:    { '--bg-0': '#0d1014', '--bg-1': '#141922', '--bg-2': '#1a2030' },
      ink:      { '--bg-0': '#08090b', '--bg-1': '#0e1015', '--bg-2': '#13161c' },
    };
    const accents = {
      'slate-cyan':  { '--accent': '#7fa3b5', '--accent-soft': 'rgba(127,163,181,.14)', '--accent-line': 'rgba(127,163,181,.32)' },
      'desat-green': { '--accent': '#6fae8f', '--accent-soft': 'rgba(111,174,143,.14)', '--accent-line': 'rgba(111,174,143,.32)' },
      'amber':       { '--accent': '#caa15a', '--accent-soft': 'rgba(202,161,90,.14)',  '--accent-line': 'rgba(202,161,90,.32)' },
    };
    Object.entries(tones[tweaks.tone] || tones.graphite).forEach(([k,v]) => r.style.setProperty(k,v));
    Object.entries(accents[tweaks.accent] || accents['slate-cyan']).forEach(([k,v]) => r.style.setProperty(k,v));
  }, [tweaks]);

  const { DesignCanvas, DCSection, DCArtboard, TweaksPanel, TweakSection, TweakRadio } = window;

  return (
    <>
      <DesignCanvas>
        <DCSection id="surfaces" title="Surfaces" subtitle="local-first AI archive · CLI + polylogued + web reader + TUI">
          <DCArtboard id="web"      label="local web reader · http://127.0.0.1:8765"  width={1480} height={920}>
            <window.WebReader />
          </DCArtboard>
          <DCArtboard id="cli"      label="polylogue · query-first CLI"               width={900}  height={920}>
            <window.TerminalCLI />
          </DCArtboard>
          <DCArtboard id="fuzzy"    label="polylogue select conversation · fuzzy"      width={1100} height={520}>
            <window.FuzzySelect />
          </DCArtboard>
          <DCArtboard id="tui"      label="polylogue dashboard · operator cockpit"     width={1300} height={780}>
            <window.TUICockpit />
          </DCArtboard>
        </DCSection>

        <DCSection id="architecture" title="Architecture">
          <DCArtboard id="boundary" label="surface boundaries — what lives where"      width={1300} height={720}>
            <window.Boundary />
          </DCArtboard>
          <DCArtboard id="states"   label="states — empty · loading · degraded · failure · privacy"  width={1700} height={920}>
            <window.States />
          </DCArtboard>
          <DCArtboard id="handoff"  label="implementation handoff · vertical slice 1"  width={1500} height={1500}>
            <window.Handoff />
          </DCArtboard>
        </DCSection>

        <DCSection id="reference" title="Reference">
          <DCArtboard id="keymap"    label="keymap · one muscle memory across surfaces" width={1500} height={780}>
            <window.Keymap />
          </DCArtboard>
          <DCArtboard id="inventory" label="component inventory · models · services · widgets" width={1500} height={1100}>
            <window.Inventory />
          </DCArtboard>
          <DCArtboard id="roadmap"   label="roadmap · slices through the issue stack"   width={1500} height={900}>
            <window.Roadmap />
          </DCArtboard>
          <DCArtboard id="telemetry" label="local telemetry · what the daemon counts"   width={1500} height={780}>
            <window.Telemetry />
          </DCArtboard>
          <DCArtboard id="dataflow"   label="data flow · provider → archive → surfaces" width={1700} height={760}>
            <window.Dataflow />
          </DCArtboard>
          <DCArtboard id="personas"   label="personas · who hires this and for what"    width={1500} height={1100}>
            <window.Personas />
          </DCArtboard>
          <DCArtboard id="comparison" label="honest comparison · vs the alternatives"   width={1700} height={900}>
            <window.Comparison />
          </DCArtboard>
        </DCSection>
      </DesignCanvas>

      <TweaksPanel title="Tweaks">
        <TweakSection label="palette tone">
          <TweakRadio
            label="tone"
            value={tweaks.tone}
            onChange={(v) => setTweak('tone', v)}
            options={['graphite','slate','ink']}
          />
        </TweakSection>
        <TweakSection label="accent">
          <TweakRadio
            label="accent"
            value={tweaks.accent}
            onChange={(v) => setTweak('accent', v)}
            options={['slate-cyan','desat-green','amber']}
          />
        </TweakSection>
      </TweaksPanel>
    </>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
