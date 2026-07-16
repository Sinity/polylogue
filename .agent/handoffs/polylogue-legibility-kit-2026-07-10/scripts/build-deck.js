const path = require('path');
const pptxgen = require('pptxgenjs');
const {
  imageSizingContain,
  imageSizingCrop,
  safeOuterShadow,
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require(process.env.PPTXGENJS_HELPERS_PATH || 'pptxgenjs_helpers');

const ROOT = path.resolve(__dirname, '..');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Polylogue + Sinex external-legibility working session';
pptx.subject = 'External legibility and launch plan for Polylogue and Sinex';
pptx.title = 'Polylogue + Sinex: from sophisticated substrates to legible products';
pptx.company = '';
pptx.lang = 'en-US';
pptx.theme = {
  headFontFace: 'Aptos Display',
  bodyFontFace: 'Aptos',
  lang: 'en-US',
};
pptx.defineSlideMaster({
  title: 'DARK',
  background: { color: '0A0D12' },
  objects: [
    { rect: { x: 0.55, y: 7.13, w: 12.25, h: 0.01, line: { color: '273141', width: 1 } } },
    { text: { text: 'POLYLOGUE × SINEX · EXTERNAL LEGIBILITY KIT', options: { x: 0.62, y: 7.17, w: 5.8, h: 0.18, fontFace: 'Aptos', fontSize: 7.5, color: '718094', charSpacing: 1.1, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.14, color: '718094', fontFace: 'Aptos', fontSize: 8 },
});
pptx.defineSlideMaster({
  title: 'LIGHT',
  background: { color: 'F3F0E9' },
  objects: [
    { rect: { x: 0.55, y: 7.13, w: 12.25, h: 0.01, line: { color: 'C8C4BA', width: 1 } } },
    { text: { text: 'POLYLOGUE × SINEX · EXTERNAL LEGIBILITY KIT', options: { x: 0.62, y: 7.17, w: 5.8, h: 0.18, fontFace: 'Aptos', fontSize: 7.5, color: '747970', charSpacing: 1.1, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.14, color: '747970', fontFace: 'Aptos', fontSize: 8 },
});

const C = {
  dark: '0A0D12', panel: '111721', line: '293443', text: 'EFF4FA', muted: '9CA9B8',
  poly: '65E2C1', sinex: 'E99E66', bead: '93A7FF', blue: '73A6FF', red: 'FF7F88',
  green: '76E18B', amber: 'F2C66D', light: 'F3F0E9', ink: '171A17', lmuted: '626961', white: 'FFFFFF'
};
const shadow = safeOuterShadow('000000', 0.22, 45, 3, 1.4);

function addTitle(slide, kicker, title, subtitle, light=false) {
  const ink = light ? C.ink : C.text;
  const muted = light ? C.lmuted : C.muted;
  slide.addText(kicker.toUpperCase(), { x:0.65, y:0.48, w:5.8, h:0.22, margin:0, fontSize:9.5, bold:true, charSpacing:1.6, color: light ? '1E6C55' : C.poly });
  slide.addText(title, { x:0.65, y:0.82, w:12.0, h:0.78, margin:0, fontSize:31, bold:true, breakLine:false, color:ink, fit:'shrink' });
  if (subtitle) slide.addText(subtitle, { x:0.67, y:1.62, w:11.75, h:0.5, margin:0, fontSize:14.5, color:muted, breakLine:false, fit:'shrink' });
}
function addPill(slide, text, x, y, w, color=C.poly, light=false) {
  slide.addShape(pptx.ShapeType.roundRect, { x,y,w,h:0.32, rectRadius:0.06, fill:{color: light?'FFFFFF':'111A22', transparency: light?8:0}, line:{color,width:1} });
  slide.addText(text, {x:x+0.07,y:y+0.075,w:w-0.14,h:0.13,fontSize:8.5,bold:true,color: light?C.ink:color,margin:0,align:'center',fit:'shrink'});
}
function addCard(slide, x,y,w,h, title, body, accent=C.poly, light=false) {
  slide.addShape(pptx.ShapeType.roundRect,{x,y,w,h,rectRadius:0.08,fill:{color:light?'FBFAF7':C.panel},line:{color:light?'D2CEC3':C.line,width:1},shadow: light?undefined:shadow});
  slide.addShape(pptx.ShapeType.line,{x:x+0.22,y:y+0.22,w:0.34,h:0,line:{color:accent,width:2.5}});
  slide.addText(title,{x:x+0.22,y:y+0.36,w:w-0.44,h:0.35,fontSize:15.2,bold:true,color:light?C.ink:C.text,margin:0,fit:'shrink'});
  slide.addText(body,{x:x+0.22,y:y+0.82,w:w-0.44,h:h-1.02,fontSize:11.2,color:light?C.lmuted:C.muted,margin:0.01,breakLine:false,fit:'shrink',valign:'top'});
}
function addFooterClaim(slide, text, light=false) {
  slide.addShape(pptx.ShapeType.roundRect,{x:0.65,y:6.45,w:12.0,h:0.45,rectRadius:0.05,fill:{color:light?'E9E5DC':'0F151E'},line:{color:light?'CCC7BC':C.line,width:1}});
  slide.addText(text,{x:0.87,y:6.58,w:11.55,h:0.16,fontSize:9.5,color:light?C.ink:C.muted,margin:0,align:'center',fit:'shrink'});
}
function clean(slide) {
  warnIfSlideHasOverlaps(slide, pptx); // diagnostics only
  warnIfSlideElementsOutOfBounds(slide, pptx);
}

// 1 — title
{
  const s=pptx.addSlide('DARK');
  s.addText('POLYLOGUE', {x:0.75,y:0.55,w:1.62,h:0.3,fontSize:11,bold:true,charSpacing:1.8,color:C.poly,margin:0});
  s.addText('×', {x:2.43,y:0.49,w:0.28,h:0.3,fontSize:18,color:'526176',margin:0,align:'center'});
  s.addText('SINEX', {x:2.78,y:0.55,w:1.25,h:0.3,fontSize:11,bold:true,charSpacing:1.8,color:C.sinex,margin:0});
  s.addText('From sophisticated substrates\nto legible products', {x:0.75,y:1.45,w:8.2,h:1.75,fontSize:42,bold:true,color:C.text,margin:0,breakLine:false,fit:'shrink'});
  s.addText('A concrete external-legibility package: repository patches, a rebuilt demo doctrine, an executable Beads launch cut, a single-machine swarm plan, visual prototypes, and sixteen parallel fork missions.', {x:0.78,y:3.55,w:7.4,h:1.05,fontSize:17,color:C.muted,margin:0,breakLine:false,fit:'shrink'});
  s.addShape(pptx.ShapeType.roundRect,{x:9.12,y:1.18,w:3.35,h:4.75,rectRadius:0.12,fill:{color:'101721'},line:{color:'334153',width:1},shadow});
  const items=[['01','A story a stranger remembers',C.poly],['02','Proofs with independent oracles',C.blue],['03','A launch cut agents can execute',C.bead],['04','Sinex-backed Polylogue direction',C.sinex]];
  items.forEach((it,i)=>{const y=1.62+i*1.02;s.addText(it[0],{x:9.47,y,w:0.4,h:0.2,fontSize:9,bold:true,color:it[2],margin:0});s.addText(it[1],{x:9.95,y:y-0.03,w:2.15,h:0.38,fontSize:13.2,bold:true,color:C.text,margin:0,fit:'shrink'});if(i<3)s.addShape(pptx.ShapeType.line,{x:9.46,y:y+0.55,w:2.48,h:0,line:{color:C.line,width:1}})});
  s.addText('Prepared from static repository access · 10 July 2026', {x:0.78,y:6.38,w:6.2,h:0.25,fontSize:9,color:'718094',margin:0});
  clean(s);
}

// 2 — diagnosis
{
  const s=pptx.addSlide('LIGHT'); addTitle(s,'The core diagnosis','The products are deeper than the story available to outsiders.','Capability is not the limiting factor. Category, narrative order, proof design, and first-run experience are.',true);
  addCard(s,0.7,2.35,3.75,2.95,'Polylogue today','A sophisticated evidence model is often encountered as a broad transcript archive. The generated site drifts toward “AI memory,” and the current tour proves construct breadth before showing a memorable user problem.',C.poly,true);
  addCard(s,4.78,2.35,3.75,2.95,'Sinex today','Its most distinctive ideas—material versus interpretation, three clocks, replay, explicit gaps, and authority—sit behind service topology. The deterministic demo is a DB/API smoke path, not a thesis proof.',C.sinex,true);
  addCard(s,8.86,2.35,3.75,2.95,'Joint risk','A metadata-only bridge wastes the fit; a generic-event merger destroys Polylogue’s domain value. The integration needs a strong authority matrix and stable identity/revision contract.',C.bead,true);
  addFooterClaim(s,'External legibility is a product surface: category → story → evidence → caveat → drill-down → next action.',true); clean(s);
}

// 3 — category pair
{
  const s=pptx.addSlide('DARK'); addTitle(s,'Positioning','One stable category sentence per project.','Everything else should reinforce these statements, not compete with them.');
  s.addShape(pptx.ShapeType.roundRect,{x:0.75,y:2.35,w:5.95,h:3.1,rectRadius:0.12,fill:{color:'101821'},line:{color:'2D6F62',width:1.3},shadow});
  s.addText('POLYLOGUE',{x:1.05,y:2.72,w:2.2,h:0.25,fontSize:10,bold:true,charSpacing:1.5,color:C.poly,margin:0});
  s.addText('The local flight recorder\nand system of record\nfor AI work.',{x:1.05,y:3.17,w:5.0,h:1.25,fontSize:27,bold:true,color:C.text,margin:0,fit:'shrink'});
  s.addText('Claims have receipts. History has topology. Memory requires judgment.',{x:1.07,y:4.63,w:4.95,h:0.48,fontSize:12.2,color:C.muted,margin:0,fit:'shrink'});
  s.addShape(pptx.ShapeType.roundRect,{x:6.95,y:2.35,w:5.6,h:3.1,rectRadius:0.12,fill:{color:'171713'},line:{color:'9B6846',width:1.3},shadow});
  s.addText('SINEX',{x:7.25,y:2.72,w:2.2,h:0.25,fontSize:10,bold:true,charSpacing:1.5,color:C.sinex,margin:0});
  s.addText('The local evidence substrate\nfor digital life\nand agent work.',{x:7.25,y:3.17,w:4.75,h:1.25,fontSize:27,bold:true,color:C.text,margin:0,fit:'shrink'});
  s.addText('Preserve source material. Reinterpret honestly. Expose what is missing.',{x:7.27,y:4.63,w:4.55,h:0.48,fontSize:12.2,color:C.muted,margin:0,fit:'shrink'});
  addFooterClaim(s,'Together: Polylogue explains AI work; Sinex preserves the wider evidentiary world in which it happened.'); clean(s);
}

// 4 — Polylogue mockup
{
  const s=pptx.addSlide('DARK'); addTitle(s,'Polylogue first impression','Lead with one evidence chain, not an inventory.','The public-safe landing prototype makes the structural tool result the hero.');
  s.addImage({path:path.join(ROOT, 'mockups/polylogue-home.png'),...imageSizingContain(path.join(ROOT, 'mockups/polylogue-home.png'),0.75,2.22,8.0,4.18)});
  addCard(s,9.05,2.35,3.55,1.12,'Receipts, not rhetoric','Failure is grounded in exit_code=4 and a typed tool result.',C.poly,false);
  addCard(s,9.05,3.7,3.55,1.12,'Topology, not folders','Forked history composes without erasing physical evidence.',C.blue,false);
  addCard(s,9.05,5.05,3.55,1.12,'Judged memory','Candidates and accepted assertions remain different states.',C.bead,false);
  clean(s);
}

// 5 — Sinex mockup
{
  const s=pptx.addSlide('LIGHT'); addTitle(s,'Sinex first impression','Lead with epistemic payoff before infrastructure.','The strongest story is reconstruction plus an honest missing-source boundary.',true);
  s.addImage({path:path.join(ROOT, 'mockups/sinex-home.png'),...imageSizingContain(path.join(ROOT, 'mockups/sinex-home.png'),0.72,2.22,8.15,4.18)});
  addCard(s,9.15,2.32,3.35,0.92,'Material ≠ interpretation','Parser mistakes remain reversible.',C.sinex,true);
  addCard(s,9.15,3.42,3.35,0.92,'Three clocks','Occurrence, coining, and persistence do not collapse.',C.blue,true);
  addCard(s,9.15,4.52,3.35,0.92,'Confidence ≠ authority','Models propose; judgment promotes.',C.green,true);
  addCard(s,9.15,5.62,3.35,0.92,'Silence ≠ absence','Disabled, stale, partial, and empty are distinct.',C.red,true);
  clean(s);
}

// 6 — architecture
{
  const s=pptx.addSlide('DARK'); addTitle(s,'Maximal joint architecture','Sinex is the durable backend; Polylogue remains the domain kernel.','The target stores transcripts in Sinex without flattening Polylogue into generic events.');
  const cols=[0.72,3.22,5.72,8.22,10.72];
  const labels=[['Sources','exports · hooks\nbrowser · files',C.muted],['Polylogue','acquire · parse\nnormalize',C.poly],['Sinex','material · history\njudgment · effects',C.sinex],['Projections','transcripts · search\nwork packets',C.blue],['Product','CLI · web · MCP\ncontext',C.bead]];
  labels.forEach((d,i)=>{s.addShape(pptx.ShapeType.roundRect,{x:cols[i],y:2.65,w:2.0,h:1.45,rectRadius:0.09,fill:{color:i===2?'18150F':'111821'},line:{color:d[2],width:1.2},shadow});s.addText(d[0],{x:cols[i]+0.16,y:2.94,w:1.68,h:0.28,fontSize:14.3,bold:true,color:C.text,align:'center',margin:0});s.addText(d[1],{x:cols[i]+0.13,y:3.36,w:1.74,h:0.48,fontSize:10,color:C.muted,align:'center',margin:0,fit:'shrink'});if(i<4){s.addShape(pptx.ShapeType.chevron,{x:cols[i]+2.06,y:3.19,w:0.34,h:0.35,fill:{color:'3B485A'},line:{color:'3B485A'}})}});
  addCard(s,0.78,4.65,3.78,1.18,'Sinex owns durability','Provider-native + normalized material, replay-specific interpretations, judgments, lifecycle, coverage, settlement, and model effects.',C.sinex,false);
  addCard(s,4.79,4.65,3.78,1.18,'Polylogue owns meaning','Sessions, messages, tools, lineage, compaction, physical/logical accounting, memory policy, query behavior, and UX.',C.poly,false);
  addCard(s,8.8,4.65,3.78,1.18,'SQLite remains valuable','Standalone mode, local/offline projection, FTS/vector acceleration, UI state, cache, watermarks, and outbox.',C.bead,false);
  addFooterClaim(s,'Decisive proof: drop rebuildable SQLite tiers and reconstruct them from Sinex-held material and history.'); clean(s);
}

// 7 — demo doctrine
{
  const s=pptx.addSlide('LIGHT'); addTitle(s,'Demo doctrine','A demo is a bounded experiment, not a pretty transcript.','Every promoted story needs an oracle, negative controls, caveats, and a product-boundary receipt.',true);
  const steps=[['QUESTION','What decision or doubt does the viewer have?'],['CLAIM','What exactly will the artifact establish?'],['CONSTRUCT','Which property is being measured?'],['ORACLE','What independently determines the expected result?'],['FALSIFIERS','What outcome would make the claim fail?'],['RECEIPTS','Which refs, commands, packets, and hashes reproduce it?']];
  steps.forEach((d,i)=>{const x=0.72+(i%3)*4.15;const y=2.32+Math.floor(i/3)*1.68;s.addShape(pptx.ShapeType.roundRect,{x,y,w:3.78,h:1.35,rectRadius:0.07,fill:{color:'FBFAF7'},line:{color:'D2CEC4',width:1}});s.addText(String(i+1).padStart(2,'0'),{x:x+0.18,y:y+0.17,w:0.35,h:0.22,fontSize:9,bold:true,color:i<3?'1E6C55':'B55D2E',margin:0});s.addText(d[0],{x:x+0.62,y:y+0.16,w:2.7,h:0.22,fontSize:10,bold:true,charSpacing:1.1,color:C.ink,margin:0});s.addText(d[1],{x:x+0.18,y:y+0.54,w:3.38,h:0.54,fontSize:11.2,color:C.lmuted,margin:0,fit:'shrink'});});
  addFooterClaim(s,'Presentation order: payoff → evidence → honest limitation → drill-down → architecture reveal.',true); clean(s);
}

// 8 — demo portfolio
{
  const s=pptx.addSlide('DARK'); addTitle(s,'Reconsidered demo portfolio','A small public arc, backed by deeper proof cards and torture tests.','The hero stories stop competing with subsystem verification.');
  const rows=[
    ['HERO','Polylogue · The Receipts','assistant claim → typed failure → repair → safe resume',C.poly],
    ['HERO','Sinex · Reconstruct Tuesday','multi-source moment + deliberate coverage hole',C.sinex],
    ['HERO','Joint · Resume This Bead','intent + agent work + machine effects + reviewed context',C.bead],
    ['PROOF','History Has Branches','physical artifacts ≠ logical unique work',C.blue],
    ['PROOF','The System Changes Its Mind','parser-v1 → replay → parser-v2 with history intact',C.amber],
    ['TORTURE','Crash, resume, no duplicate occurrence','interrupted import with settlement and oracle',C.red],
  ];
  rows.forEach((r,i)=>{const y=2.2+i*0.68;s.addShape(pptx.ShapeType.roundRect,{x:0.78,y,w:11.85,h:0.52,rectRadius:0.04,fill:{color:i%2?'10151D':'121923'},line:{color:C.line,width:0.7}});s.addText(r[0],{x:1.0,y:y+0.17,w:0.7,h:0.14,fontSize:8.2,bold:true,charSpacing:0.8,color:r[3],margin:0});s.addText(r[1],{x:1.95,y:y+0.12,w:3.35,h:0.22,fontSize:12,bold:true,color:C.text,margin:0,fit:'shrink'});s.addText(r[2],{x:5.5,y:y+0.13,w:6.5,h:0.22,fontSize:10.8,color:C.muted,margin:0,fit:'shrink'});});
  addFooterClaim(s,'Keep the existing broad demo catalogs—but classify them as proof cards, experiments, or operational campaigns.'); clean(s);
}

// 9 — launch cut
{
  const s=pptx.addSlide('LIGHT'); addTitle(s,'Polylogue launch cut','A deliberately narrow sequence drawn from existing Beads.','Trust and web reliability first; one flagship story and one anti-grep proof; then packaging and media.',true);
  const waves=[
    ['0','TRUST FLOOR','0hqs · bby.1','Bound queries; truthful timeout/degraded UI',C.red],
    ['1','LANDING','3tl.12 · 3tl.15 · 3tl.16 · 3tl.8','README, anti-grep, claims ledger, repository surface','1E6C55'],
    ['2','VISIBLE PRODUCT','ap7 · 212.2','Semantic tool cards + The Receipts flagship',C.blue],
    ['3','EVIDENCE SHELF','3tl.4 · reuse 212.8','Bounded findings + honesty anti-demo',C.amber],
    ['4','RELEASE','3tl.7 · 3tl.9 · 3tl.10','Clean install matrix, media drift gate, launch packet',C.bead],
  ];
  waves.forEach((w,i)=>{const y=2.12+i*0.82;s.addShape(pptx.ShapeType.roundRect,{x:0.77,y,w:11.86,h:0.62,rectRadius:0.05,fill:{color:i===0?'FFF3F2':'FBFAF7'},line:{color:i===0?'E1AAA7':'D2CEC3',width:1}});s.addShape(pptx.ShapeType.ellipse,{x:0.96,y:y+0.13,w:0.34,h:0.34,fill:{color:w[4]},line:{color:w[4]}});s.addText(w[0],{x:1.04,y:y+0.22,w:0.18,h:0.1,fontSize:7.5,bold:true,color:'FFFFFF',align:'center',margin:0});s.addText(w[1],{x:1.52,y:y+0.13,w:1.55,h:0.18,fontSize:9,bold:true,charSpacing:1,color:C.ink,margin:0});s.addText(w[2],{x:3.15,y:y+0.13,w:2.4,h:0.18,fontSize:9.5,bold:true,color:'566170',margin:0});s.addText(w[3],{x:5.63,y:y+0.13,w:6.45,h:0.25,fontSize:11,color:C.lmuted,margin:0,fit:'shrink'});});
  addFooterClaim(s,'Do not block the launch on the full Sinex backend, all semantic card types, or publishable memory uplift.',true); clean(s);
}

// 10 — swarm
{
  const s=pptx.addSlide('DARK'); addTitle(s,'Single-machine swarm','Parallelize by file ownership and resource class—not by issue count.','Sacrifice Git elegance for throughput; retain semantic authority and validation gates.');
  const lanes=[
    ['CHEAP / PARALLEL','Narrative · docs · claims · ADR','6–10 agents',C.poly],
    ['MEDIUM','fixtures · tour · renderer contracts','3–5 agents',C.blue],
    ['HEAVY TOKEN 1','full Polylogue verify / scale run','one at a time',C.bead],
    ['HEAVY TOKEN 2','Rust build / Sinex infra demo','one at a time',C.sinex],
    ['BROWSER TOKEN','E2E / screenshots / media','one at a time',C.amber],
  ];
  lanes.forEach((l,i)=>{const y=2.25+i*0.71;s.addShape(pptx.ShapeType.roundRect,{x:0.78,y,w:7.05,h:0.56,rectRadius:0.04,fill:{color:i<2?'111923':'15161A'},line:{color:C.line,width:1}});s.addText(l[0],{x:1.0,y:y+0.18,w:1.45,h:0.14,fontSize:8.2,bold:true,color:l[3],charSpacing:0.8,margin:0});s.addText(l[1],{x:2.55,y:y+0.14,w:3.25,h:0.22,fontSize:11.2,bold:true,color:C.text,margin:0,fit:'shrink'});s.addText(l[2],{x:6.1,y:y+0.17,w:1.35,h:0.16,fontSize:9.2,color:C.muted,align:'right',margin:0});});
  s.addShape(pptx.ShapeType.roundRect,{x:8.25,y:2.25,w:4.3,h:3.7,rectRadius:0.09,fill:{color:'111721'},line:{color:'354153',width:1},shadow});
  s.addText('CONDUCTOR LOOP',{x:8.55,y:2.52,w:2.0,h:0.22,fontSize:9.5,bold:true,color:C.bead,charSpacing:1.2,margin:0});
  const loop=['Assign exclusive paths','Require a proof packet','Cherry-pick or apply patch','Run narrow gate','Integrate by wave','Run cold-reader + privacy audit'];
  loop.forEach((t,i)=>{s.addShape(pptx.ShapeType.ellipse,{x:8.58,y:2.92+i*0.46,w:0.22,h:0.22,fill:{color:i<3?C.poly:C.bead},line:{color:i<3?C.poly:C.bead}});s.addText(String(i+1),{x:8.63,y:2.985+i*0.46,w:0.11,h:0.08,fontSize:6.7,bold:true,color:C.dark,align:'center',margin:0});s.addText(t,{x:8.98,y:2.92+i*0.46,w:2.95,h:0.22,fontSize:10.7,color:C.text,margin:0,fit:'shrink'});});
  addFooterClaim(s,'Use worktrees for isolation; use a shared artifact shelf and merge packet for coordination.'); clean(s);
}

// 11 — prompts
{
  const s=pptx.addSlide('LIGHT'); addTitle(s,'Sixteen fork missions','The package can be executed as a conversation swarm.','Each prompt owns a bounded surface and returns a patch plus evidence.',true);
  const groups=[
    ['PUBLIC STORY','01 landing · 05 claims · 06 install/media',C.poly],
    ['POLYLOGUE PRODUCT','02 semantic cards · 03 Receipts · 04 anti-grep · 07 web · 08 resume · 09 lineage',C.blue],
    ['SINEX PRODUCT','10 narrative · 11 moment · 12 replay · 13 outage',C.sinex],
    ['JOINT ARCHITECTURE','14 backend ADR · 15 Agent Work Packet',C.bead],
    ['INTEGRATION','16 conductor / cold-reader / launch audit',C.red],
  ];
  groups.forEach((g,i)=>{const y=2.16+i*0.79;s.addShape(pptx.ShapeType.roundRect,{x:0.78,y,w:11.86,h:0.6,rectRadius:0.05,fill:{color:'FBFAF7'},line:{color:'D2CEC3',width:1}});s.addShape(pptx.ShapeType.rect,{x:0.78,y,w:0.11,h:0.6,fill:{color:g[2]},line:{color:g[2]}});s.addText(g[0],{x:1.08,y:y+0.19,w:2.05,h:0.17,fontSize:9,bold:true,charSpacing:0.9,color:C.ink,margin:0});s.addText(g[1],{x:3.15,y:y+0.15,w:8.8,h:0.26,fontSize:11.2,color:C.lmuted,margin:0,fit:'shrink'});});
  addFooterClaim(s,'Prompts are self-contained: scope, design constraints, negative controls, validation, and deliverables.',true); clean(s);
}

// 12 — patches
{
  const s=pptx.addSlide('DARK'); addTitle(s,'Concrete work already prepared','The package includes bounded repository patches, not just recommendations.','They are static-access improvements designed to be applied or mined into the active branches.');
  addCard(s,0.77,2.2,5.85,2.65,'Polylogue patch','• category-aligned README and package metadata\n• new Why / Demos / Public Claims / Sinex backend docs\n• revised docs navigation and landing page\n• evidence-first deterministic tour\n• focused unit tests and generated-site checks',C.poly,false);
  addCard(s,6.88,2.2,5.7,2.65,'Sinex patch','• payoff-first README\n• Concepts / For Agents / Demos / Public Claims docs\n• transcript-complete Polylogue backend direction\n• current bridge explicitly labeled transitional\n• metadata-only doctrine corrected at the design layer',C.sinex,false);
  addCard(s,0.77,5.1,3.75,1.15,'Visual prototypes','Polylogue landing · Sinex landing · Resume This Bead dashboard',C.blue,false);
  addCard(s,4.78,5.1,3.75,1.15,'Machine-readable plans','claims ledger · demo portfolio · Beads launch cut · worktree lanes',C.bead,false);
  addCard(s,8.79,5.1,3.79,1.15,'Proof artifact','A passing, private-data-free Polylogue tour with 30/30 declared constructs.',C.green,false);
  clean(s);
}

// 13 — end
{
  const s=pptx.addSlide('LIGHT');
  s.addText('The shortest credible path', {x:0.78,y:0.72,w:7.8,h:0.45,fontSize:10,bold:true,charSpacing:1.5,color:'1E6C55',margin:0});
  s.addText('One memorable incident.\nOne anti-grep proof.\nOne honest limitation.\nOne reviewed resume.', {x:0.78,y:1.35,w:7.35,h:2.35,fontSize:36,bold:true,color:C.ink,margin:0,fit:'shrink'});
  s.addText('Then let the architecture reveal why the result is trustworthy.', {x:0.82,y:4.04,w:6.9,h:0.58,fontSize:17,color:C.lmuted,margin:0,fit:'shrink'});
  s.addShape(pptx.ShapeType.roundRect,{x:8.45,y:1.18,w:4.05,h:4.8,rectRadius:0.1,fill:{color:'FBFAF7'},line:{color:'CDC8BD',width:1},shadow});
  s.addText('SHIP BAR', {x:8.82,y:1.55,w:1.5,h:0.22,fontSize:9.5,bold:true,charSpacing:1.2,color:C.sinex,margin:0});
  const checklist=[['Bounded web failure states','must pass'],['Flagship deterministic packet','must pass'],['Claims resolve to receipts','must pass'],['Clean install path','must pass'],['Cold-reader category recall','must pass'],['Full Sinex backend','may follow'],['General memory uplift','experiment later']];
  checklist.forEach((r,i)=>{const y=2.02+i*0.48;const later=i>=5;s.addShape(pptx.ShapeType.ellipse,{x:8.83,y:y+0.02,w:0.2,h:0.2,fill:{color:later?'C9C5BA':C.green},line:{color:later?'C9C5BA':C.green}});s.addText(later?'–':'✓',{x:8.87,y:y+0.055,w:0.11,h:0.08,fontSize:7,bold:true,color:later?C.ink:C.dark,margin:0,align:'center'});s.addText(r[0],{x:9.18,y,w:1.95,h:0.19,fontSize:10.5,bold:!later,color:C.ink,margin:0,fit:'shrink'});s.addText(r[1],{x:11.45,y:y+0.01,w:0.68,h:0.16,fontSize:8.3,color:later?'817F77':'1E6C55',align:'right',margin:0,fit:'shrink'});});
  s.addText('Artifacts: plans · patches · prompts · mockups · proof packet', {x:0.82,y:6.48,w:7.4,h:0.22,fontSize:10,color:'747970',margin:0});
  clean(s);
}

pptx.writeFile({ fileName: path.join(ROOT, 'Polylogue-Sinex-external-legibility.pptx') });
