// Keymap artboard — full keyboard map across web reader / TUI / fuzzy / CLI palette.

const Keymap = () => (
  <div className="keymap">
    <header className="keymap__head">
      <Mono>KEYMAP</Mono>
      <h2>One muscle memory across surfaces</h2>
      <p>Same verbs, same letters. Web reader, TUI, fuzzy selector, and the <Mono>:</Mono> palette agree.</p>
    </header>

    <div className="keymap__grid">
      <KCol title="navigation">
        <KRow keys={['/', 'g s']} act="search" note="focus query box · facets compose"/>
        <KRow keys={[':']}        act="command palette" note="fuzzy over every action"/>
        <KRow keys={['g a']}      act="annotations" note="my notes across the archive"/>
        <KRow keys={['g r']}      act="recall packs"/>
        <KRow keys={['g d']}      act="dashboard / TUI"/>
        <KRow keys={['j','k']}    act="next / prev result"/>
        <KRow keys={['↵']}        act="open conversation"/>
        <KRow keys={['gg','G']}   act="top / bottom"/>
        <KRow keys={['esc']}      act="back · clear filter"/>
      </KCol>

      <KCol title="reader">
        <KRow keys={['m']} act="messages tab"/>
        <KRow keys={['r']} act="raw artifacts"/>
        <KRow keys={['p']} act="provenance"/>
        <KRow keys={['s']} act="star"/>
        <KRow keys={['P']} act="pin"/>
        <KRow keys={['a']} act="annotate at message"/>
        <KRow keys={['o']} act="open in editor (cwd)"/>
        <KRow keys={['y']} act="yank id"/>
        <KRow keys={['Y']} act="yank message link"/>
        <KRow keys={['n','N']} act="next / prev match"/>
      </KCol>

      <KCol title="selection · multi">
        <KRow keys={['x']}   act="toggle select"/>
        <KRow keys={['*']}   act="select all visible"/>
        <KRow keys={['+']}   act="add to recall pack"/>
        <KRow keys={['t']}   act="add tag…"/>
        <KRow keys={['T']}   act="remove tag…"/>
        <KRow keys={['e']}   act="bulk export…"/>
        <KRow keys={['⌫']}   act="delete (confirm)"/>
      </KCol>

      <KCol title="status · daemon">
        <KRow keys={['?']}      act="help overlay"/>
        <KRow keys={['F2']}     act="cycle panel focus (TUI)"/>
        <KRow keys={['^r']}     act="refresh status"/>
        <KRow keys={['^l']}     act="redraw"/>
        <KRow keys={['1','2','3','4','5','6']} act="jump to TUI panel"/>
        <KRow keys={['q']}      act="quit TUI"/>
      </KCol>

      <KCol title="fuzzy selector">
        <KRow keys={['↑','↓']}  act="move"/>
        <KRow keys={['tab']}    act="multi-select"/>
        <KRow keys={['↵']}      act="accept"/>
        <KRow keys={['^p']}     act="print id to stdout"/>
        <KRow keys={['^o']}     act="open in browser/editor"/>
        <KRow keys={['^f']}     act="filter mode (live spec)"/>
        <KRow keys={['esc']}    act="cancel"/>
      </KCol>

      <KCol title="palette · :">
        <KRow keys={[': open']}      act="open conversation"/>
        <KRow keys={[': doctor']}    act="run doctor target"/>
        <KRow keys={[': export']}    act="bulk export"/>
        <KRow keys={[': recall new']} act="new recall pack"/>
        <KRow keys={[': view save']} act="save current filter"/>
        <KRow keys={[': daemon restart']} act="restart polylogued"/>
      </KCol>
    </div>
  </div>
);

const KCol = ({ title, children }) => (
  <section className="kcol">
    <header><Mono>{title}</Mono></header>
    <div>{children}</div>
  </section>
);

const KRow = ({ keys, act, note }) => (
  <div className="krow">
    <span className="krow__keys">{keys.map((k,i) => <Key key={i}>{k}</Key>)}</span>
    <span className="krow__act">{act}</span>
    {note && <span className="krow__note">{note}</span>}
  </div>
);

window.Keymap = Keymap;
