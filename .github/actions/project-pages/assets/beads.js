(() => {
  const pageSize = 100;
  const state = {issues: [], visible: pageSize, byId: new Map()};
  const ids = ['stats','search','status','priority','type','result-count','issues','load-more'];
  const el = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const linkify = value => esc(value).replace(/(https?:[/][/][^ <]+)/g, '<a href="$1">$1</a>');
  const closed = issue => ['closed','done','resolved'].includes(issue.status);
  const blockers = issue => (issue.dependencies || []).filter(dep => dep.type === 'blocks' && !closed(state.byId.get(dep.depends_on_id) || {}));
  const bucket = issue => closed(issue) ? 'closed' : issue.status === 'in_progress' ? 'in_progress' : blockers(issue).length ? 'blocked' : 'ready';
  const field = (title, value) => value ? `<section class="issue-section"><h3>${title}</h3><div class="issue-prose">${linkify(value)}</div></section>` : '';
  const statusBadge = issue => { const value=bucket(issue); const color=value==='closed'?'green':value==='blocked'?'red':value==='in_progress'?'yellow':''; return `<span class="badge ${color}">${esc(value.replace('_',' '))}</span>`; };
  function relationships(issue) {
    const deps=issue.dependencies || [];
    if (!deps.length) return '';
    return `<section class="issue-section"><h3>Relationships</h3><div class="relations">${deps.map(dep => `<button type="button" class="relation" data-issue="${esc(dep.depends_on_id)}">${esc(dep.type)} · ${esc(dep.depends_on_id)}</button>`).join('')}</div></section>`;
  }
  function card(issue) {
    const labels=(issue.labels || []).map(label => `<span class="badge">${esc(label)}</span>`).join('');
    const date=issue.updated_at ? new Date(issue.updated_at).toLocaleDateString(undefined,{year:'numeric',month:'short',day:'numeric'}) : '';
    return `<details class="issue" id="${esc(issue.id)}"><summary><span class="priority p${esc(issue.priority)}">P${esc(issue.priority)}</span><span><span class="issue-title"><span class="issue-id">${esc(issue.id)}</span>${esc(issue.title)}</span><span class="subline">${statusBadge(issue)}<span class="badge">${esc(issue.issue_type || 'task')}</span>${labels}</span></span><span class="updated">${esc(date)}</span></summary><div class="issue-body">${field('Description',issue.description)}${field('Design',issue.design)}${field('Acceptance criteria',issue.acceptance_criteria)}${field('Notes',issue.notes)}${relationships(issue)}${field('Closure',issue.close_reason)}</div></details>`;
  }
  function matches(issue) {
    const query=el.search.value.trim().toLowerCase();
    const haystack=[issue.id,issue.title,issue.description,issue.design,issue.acceptance_criteria,issue.notes,issue.close_reason,...(issue.labels || [])].join(String.fromCharCode(10)).toLowerCase();
    const wanted=el.status.value, actual=bucket(issue);
    const statusOk=wanted==='all' || (wanted==='active' && actual!=='closed') || wanted===actual;
    return statusOk && (el.priority.value==='all' || String(issue.priority)===el.priority.value) && (el.type.value==='all' || issue.issue_type===el.type.value) && (!query || query.split(' ').filter(Boolean).every(term => haystack.includes(term)));
  }
  function syncUrl() {
    const url=new URL(location.href);
    [['q','search',''],['status','status','active'],['priority','priority','all'],['type','type','all']].forEach(([key,id,base]) => { const value=el[id].value; if (value && value!==base) url.searchParams.set(key,value); else url.searchParams.delete(key); });
    history.replaceState(null,'',url);
  }
  function render() {
    const filtered=state.issues.filter(matches).sort((a,b)=>(a.priority-b.priority) || String(b.updated_at).localeCompare(String(a.updated_at)));
    el.issues.innerHTML=filtered.slice(0,state.visible).map(card).join('') || '<div class="empty">No issues match these filters.</div>';
    el['result-count'].textContent=`${filtered.length.toLocaleString()} matching issue${filtered.length===1?'':'s'}`;
    el['load-more'].hidden=filtered.length<=state.visible;
    syncUrl();
    if(location.hash){const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(target){target.open=true;requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));}}
  }
  function openIssue(id){if(!state.byId.has(id))return;el.search.value=id;el.status.value='all';state.visible=pageSize;location.hash=encodeURIComponent(id);render();}
  Promise.all([fetch('../project.json').then(r=>r.json()),fetch('issues.jsonl').then(r=>{if(!r.ok)throw new Error(`HTTP ${r.status}`);return r.text();})]).then(([project,raw])=>{
    document.title=`${project.name} roadmap`;
    document.getElementById('project-brand').innerHTML=`${esc(project.name)}<span>.</span>`;
    document.getElementById('github-link').href=`https://github.com/${project.repository}`;
    document.getElementById('source-link').href=`https://github.com/${project.repository}/blob/${project.branch || 'master'}/.beads/issues.jsonl`;
    state.issues=raw.split(String.fromCharCode(10)).map(line=>line.endsWith(String.fromCharCode(13))?line.slice(0,-1):line).filter(Boolean).map(JSON.parse).filter(row=>row._type==='issue');
    state.byId=new Map(state.issues.map(issue=>[issue.id,issue]));
    const params=new URLSearchParams(location.search);el.search.value=params.get('q')||'';el.status.value=params.get('status')||'active';el.priority.value=params.get('priority')||'all';
    const types=[...new Set(state.issues.map(issue=>issue.issue_type).filter(Boolean))].sort();el.type.insertAdjacentHTML('beforeend',types.map(type=>`<option value="${esc(type)}">${esc(type)}</option>`).join(''));el.type.value=params.get('type')||'all';
    const counts={active:0,ready:0,blocked:0,in_progress:0,closed:0};state.issues.forEach(issue=>{const value=bucket(issue);counts[value]=(counts[value]||0)+1;if(value!=='closed')counts.active++;});
    el.stats.innerHTML=[['Active',counts.active],['Ready',counts.ready],['Blocked',counts.blocked],['In progress',counts.in_progress],['Delivered',counts.closed]].map(([label,value])=>`<div class="stat"><strong>${value.toLocaleString()}</strong><span>${label}</span></div>`).join('');render();
  }).catch(error=>{el.issues.innerHTML=`<div class="empty">Could not load the committed Beads records: ${esc(error.message)}</div>`;el['result-count'].textContent='Board unavailable';});
  ['search','status','priority','type'].forEach(id=>el[id].addEventListener(id==='search'?'input':'change',()=>{state.visible=pageSize;render();}));
  el['load-more'].addEventListener('click',()=>{state.visible+=pageSize;render();});
  el.issues.addEventListener('click',event=>{const button=event.target.closest('[data-issue]');if(button)openIssue(button.dataset.issue);});
})();
