# Changelog

All notable user-visible changes to Polylogue are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

User-visible: anything an operator running `polylogue` would notice — new
flags, removed or renamed commands, output changes, breaking schema
migrations, security fixes. Internal refactors, test additions, and
documentation polish do not require an entry.

## [0.2.0](https://github.com/Sinity/polylogue/compare/v0.1.0...v0.2.0) (2026-07-11)


### Added

* **analytics:** delegation enabling primitive + view (1vpm.1 phases 1-2) ([#2607](https://github.com/Sinity/polylogue/issues/2607)) ([09dc46c](https://github.com/Sinity/polylogue/commit/09dc46c26648e7b90de3acf03e3efbf82f10dafc))
* **analyze:** add corpus-wide sanitized portfolio report ([#2437](https://github.com/Sinity/polylogue/issues/2437)) ([#2451](https://github.com/Sinity/polylogue/issues/2451)) ([d101a37](https://github.com/Sinity/polylogue/commit/d101a37c989177dc9d8a2b49a241a7a4818697ac))
* **archive:** preserve structured session evidence ([#2496](https://github.com/Sinity/polylogue/issues/2496)) ([e55c675](https://github.com/Sinity/polylogue/commit/e55c675aaeee8c565c7cd1bb7276af7fccc2866d))
* **browser-capture:** add gated post commands ([#2494](https://github.com/Sinity/polylogue/issues/2494)) ([79a8ac0](https://github.com/Sinity/polylogue/commit/79a8ac097aa68304bdb5b3e53ef1c1a7d6556afd))
* **browser-extension:** capture retry queue, receiver health probe, popup outcomes ([#2671](https://github.com/Sinity/polylogue/issues/2671)) ([67f522b](https://github.com/Sinity/polylogue/commit/67f522bd54762ad9a387ab4eadde1850806b4059))
* **capture:** acquire assistant-produced file bytes at capture time ([#2669](https://github.com/Sinity/polylogue/issues/2669)) ([8db8d60](https://github.com/Sinity/polylogue/commit/8db8d60b34565f23ced411b786b252b6a4473b00))
* **cli:** add full message read export ([0d63cd6](https://github.com/Sinity/polylogue/commit/0d63cd6cc2c3aac5055535c7f9e30fe8a273a356))
* converge dogfood branch state ([#2504](https://github.com/Sinity/polylogue/issues/2504)) ([d40d00a](https://github.com/Sinity/polylogue/commit/d40d00a312e5530a76e3e1ba29f3ddc37ce89d82))
* **coordination:** render subagent exchanges ([#2545](https://github.com/Sinity/polylogue/issues/2545)) ([a35bd6d](https://github.com/Sinity/polylogue/commit/a35bd6d97eafef647cf016998437c49d5aca98bc))
* **cost:** price current Opus 4.7/4.8 flagships ([#2445](https://github.com/Sinity/polylogue/issues/2445)) ([fbb8f09](https://github.com/Sinity/polylogue/commit/fbb8f09187ae013b6a27db01e2961440f507cacb))
* **demo:** enrich demo cost and fix profile token-lane round-trip ([#2446](https://github.com/Sinity/polylogue/issues/2446)) ([05f8c98](https://github.com/Sinity/polylogue/commit/05f8c980c5c516bec358abcdd841edbf742c1e07))
* **demo:** give demo session a canonical repo for repos_touched ([#2447](https://github.com/Sinity/polylogue/issues/2447)) ([b73f01d](https://github.com/Sinity/polylogue/commit/b73f01db8e2f008bee5801f17a297d80867a105d))
* **demo:** Incident 14:32 constructs — source outage, cross-material duplicate, compaction omission ([#2674](https://github.com/Sinity/polylogue/issues/2674)) ([8c15e86](https://github.com/Sinity/polylogue/commit/8c15e86127ba15454c667ccae0c7c1318d0050cc))
* **demo:** materialize session insights in no-daemon seed ([#2439](https://github.com/Sinity/polylogue/issues/2439)) ([21951de](https://github.com/Sinity/polylogue/commit/21951de986200babb056d1f9c77ad17758ec4fad))
* **demo:** one-command receipts proof, enforced public-claims gate, corpus v2 ([#2662](https://github.com/Sinity/polylogue/issues/2662)) ([529fc43](https://github.com/Sinity/polylogue/commit/529fc43979d7e3292252529925ba2009a4ddfa23))
* **demo:** package claim evidence reproduction ([678f3d6](https://github.com/Sinity/polylogue/commit/678f3d6805b664ef47b1d8350ed81549a01bb9e0))
* **devtools:** accept bead: owners in coverage-manifest gap records ([#2611](https://github.com/Sinity/polylogue/issues/2611)) ([10a1212](https://github.com/Sinity/polylogue/commit/10a1212f4b17b9359494949dccb032456f26c4ef))
* **devtools:** add bench ingest-throughput wall-clock benchmark ([#2490](https://github.com/Sinity/polylogue/issues/2490)) ([d415412](https://github.com/Sinity/polylogue/commit/d415412d7d68008a92296f6f6dd29f53b4cb7a10))
* **devtools:** add claim-vs-evidence demo report ([af4915d](https://github.com/Sinity/polylogue/commit/af4915d11bea6c81eda0da5521932a9c7d150890))
* **devtools:** add cost reconciliation probe ([153dab3](https://github.com/Sinity/polylogue/commit/153dab3eaf721b82ffce37ac14174d12e91abc1d))
* **devtools:** add full-body read package projection ([0e83712](https://github.com/Sinity/polylogue/commit/0e837125aa425a8c6f8015cbeeb1b5231a01ebb6))
* **devtools:** catch subcommand/flag drift in verify doc-commands ([#2442](https://github.com/Sinity/polylogue/issues/2442)) ([0d5a3bb](https://github.com/Sinity/polylogue/commit/0d5a3bb6ee3c18e1abc4690a249aa465d0ce5e9d))
* **devtools:** classify affordance surface usage ([fb4eab8](https://github.com/Sinity/polylogue/commit/fb4eab84989d6eee4aaab86f8eb0e05d7dcfba95))
* **devtools:** compare Claude usage grains ([2abe147](https://github.com/Sinity/polylogue/commit/2abe1478401ffe117df71e0c019a62f7a2152945))
* **devtools:** Demo Finding Packet contract + registry lint ([#2589](https://github.com/Sinity/polylogue/issues/2589)) ([4e49b6c](https://github.com/Sinity/polylogue/commit/4e49b6ccd03a63ffed2b50a878a882131f79801e))
* **devtools:** enforce production writer ownership ([#2682](https://github.com/Sinity/polylogue/issues/2682)) ([f314e81](https://github.com/Sinity/polylogue/commit/f314e812ec64b43bd50be04a8f825fc9402f7c14))
* **devtools:** expose cost probe residual metadata ([a6ecbaf](https://github.com/Sinity/polylogue/commit/a6ecbaf0a6d3aa7eba2da525cad14dca056a6a4c))
* **devtools:** instrument ingest with cpu/memory/io/stage metrics ([#2491](https://github.com/Sinity/polylogue/issues/2491)) ([52b5bc0](https://github.com/Sinity/polylogue/commit/52b5bc038a0bd959563cf61db4a1e0fd15001ea3))
* **devtools:** show Claude lineage in cost residuals ([76f56a4](https://github.com/Sinity/polylogue/commit/76f56a46263278ec089d389dae54769e5854c702))
* **devtools:** show logical usage grain in cost probe ([77543d4](https://github.com/Sinity/polylogue/commit/77543d449515b036954216cb6349b5c6ed5eb7ee))
* **devtools:** summarize Codex state outlier flags ([ab72066](https://github.com/Sinity/polylogue/commit/ab72066dc528bedc06b1e1b6da01554ac0dec629))
* **devtools:** timestamp-doctrine lint for durable-tier DDL ([#2601](https://github.com/Sinity/polylogue/issues/2601)) ([6c12e92](https://github.com/Sinity/polylogue/commit/6c12e92348d6e67db212462573ff939110996395))
* **export:** add fail-closed sanitized shareable export ([#2381](https://github.com/Sinity/polylogue/issues/2381)) ([#2433](https://github.com/Sinity/polylogue/issues/2433)) ([f5ab219](https://github.com/Sinity/polylogue/commit/f5ab2195558823fd01d6db432a32e1768d6ba930))
* **export:** add sanitized spans-jsonl bundle leg ([#2434](https://github.com/Sinity/polylogue/issues/2434)) ([#2455](https://github.com/Sinity/polylogue/issues/2455)) ([c4b6771](https://github.com/Sinity/polylogue/commit/c4b677181847375d9636cb9448e3accbf4688595))
* **export:** redact Windows paths and emails in sanitized export ([#2444](https://github.com/Sinity/polylogue/issues/2444)) ([c05e83e](https://github.com/Sinity/polylogue/commit/c05e83ee9d1c2a8605f48f2d640749c0ddae62db))
* **forensics:** price origin-reported usage ([f859356](https://github.com/Sinity/polylogue/commit/f859356a56dde0229e846c6aa838b21e60ecd74c))
* **insights:** add usage timeline read model ([56f38ef](https://github.com/Sinity/polylogue/commit/56f38ef6653f9e2937983d9554270188ef1adae8))
* **insights:** deterministic agent-workflow pathology detectors ([#2448](https://github.com/Sinity/polylogue/issues/2448)) ([72d0623](https://github.com/Sinity/polylogue/commit/72d0623ea331f5348ceb7c6fde61e7c6ce41d200))
* **insights:** expose pathology distribution via API + MCP tool ([#2383](https://github.com/Sinity/polylogue/issues/2383)) ([#2449](https://github.com/Sinity/polylogue/issues/2449)) ([e7a39b3](https://github.com/Sinity/polylogue/commit/e7a39b37fb0de39ba103bb4633bce8d4ba90da23))
* install and monitor harness hooks ([#2644](https://github.com/Sinity/polylogue/issues/2644)) ([7e6123c](https://github.com/Sinity/polylogue/commit/7e6123cac3e5ed3c365ff9abb4fafe04deec3b9c))
* integrate archive convergence and capture platform ([#2534](https://github.com/Sinity/polylogue/issues/2534)) ([3498981](https://github.com/Sinity/polylogue/commit/349898106783020294e52e30d3a2296021094940))
* **lineage:** surface a completeness signal on composed reads ([#2603](https://github.com/Sinity/polylogue/issues/2603)) ([c06ca60](https://github.com/Sinity/polylogue/commit/c06ca601c3e35eae8408425c3c6b27092aab3ea4))
* **maintenance:** bound index replay selections ([d98f126](https://github.com/Sinity/polylogue/commit/d98f1262c6df6b3c274d723a80850fe56594cffa))
* **maintenance:** replay source rows into index tier ([a549285](https://github.com/Sinity/polylogue/commit/a549285140be52fde995ff826cd9d8b1073d9589))
* **mcp:** add filtered message tail paging ([f64920f](https://github.com/Sinity/polylogue/commit/f64920f22bc8234af2bb506130c6bf5c418f0da9))
* **mcp:** expose postmortem bundle and sanitized export as agent tools ([#2443](https://github.com/Sinity/polylogue/issues/2443)) ([66fb40a](https://github.com/Sinity/polylogue/commit/66fb40af5c4a7aeb8cde44b6b605414602e0d60c))
* normalize session lineage and correct cross-provider cost accounting ([#2469](https://github.com/Sinity/polylogue/issues/2469)) ([7412b69](https://github.com/Sinity/polylogue/commit/7412b69a9b64863a1bf4df11d7feeb99f61262ff))
* **parsers:** capture Claude server_tool_use web request counts ([#2486](https://github.com/Sinity/polylogue/issues/2486)) ([badd731](https://github.com/Sinity/polylogue/commit/badd73180f1e33db3bcf26db02800f4648e4a965))
* **query:** add Terminal pipeline stage; route verbs through one executor ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2463](https://github.com/Sinity/polylogue/issues/2463)) ([fb9073c](https://github.com/Sinity/polylogue/commit/fb9073cfc801a6b4d8150e6998c96b966e39047c))
* **query:** attach evidence units to session results ([f76e700](https://github.com/Sinity/polylogue/commit/f76e7006f3d6fcda7206ff7bc67b572421f15e59))
* **query:** attach related units to session queries ([#2495](https://github.com/Sinity/polylogue/issues/2495)) ([4a9c7a0](https://github.com/Sinity/polylogue/commit/4a9c7a07a12787a2b71a13f588b1b4dc835747a5))
* **query:** expose action followup classification ([c68c278](https://github.com/Sinity/polylogue/commit/c68c278ef9c5f212ca49c4a9d70215910aca1f4c))
* **query:** expose action outcome predicates ([8b0595d](https://github.com/Sinity/polylogue/commit/8b0595dcc29f88c04679f8da307a66d5a43b462c))
* **query:** expose projection unit completions ([03f9ffb](https://github.com/Sinity/polylogue/commit/03f9ffb727072ed87fd6b576dcf41784134f0682))
* **query:** select fields for attached projection units ([867b1d0](https://github.com/Sinity/polylogue/commit/867b1d0482aca870053b188648b2a1d92170efc0))
* **read-package:** expose handoff freshness metadata ([d7aa9e8](https://github.com/Sinity/polylogue/commit/d7aa9e81a5a662236f86d938a06f750f9e9a7b57))
* **read:** add projection render specs ([#2503](https://github.com/Sinity/polylogue/issues/2503)) ([ce2f62a](https://github.com/Sinity/polylogue/commit/ce2f62ac65e6e7c29558a74f1dd10ba78d0999e2))
* **storage:** block content-hash citation anchors with typed resolver ([#2588](https://github.com/Sinity/polylogue/issues/2588)) ([6974706](https://github.com/Sinity/polylogue/commit/6974706611cb18695a1b85924193724c8b3f93dd))
* **storage:** classify attachment acquisition debt separately from source blobs ([#2586](https://github.com/Sinity/polylogue/issues/2586)) ([b536fc3](https://github.com/Sinity/polylogue/commit/b536fc3aa57ed92b3b6ec2300e97d7bb3b0af247))
* **storage:** persist raw revision authority evidence ([#2681](https://github.com/Sinity/polylogue/issues/2681)) ([2032b2c](https://github.com/Sinity/polylogue/commit/2032b2cb26c6000b2f7db7d07efb4a2176546bf1))
* **usage:** expose all-origin logical token rollups ([8ca374e](https://github.com/Sinity/polylogue/commit/8ca374e59b7863e7414b473a85bb73886e98eb92))
* **usage:** label logical model rollup grain ([a336b86](https://github.com/Sinity/polylogue/commit/a336b864c670397b941a757ede80ac1df430692d))
* **usage:** report pricing provenance lanes ([57263f9](https://github.com/Sinity/polylogue/commit/57263f9c8831221ad44842259b68f9a5ab012292))
* **web:** evidence-cockpit IA — four-verb navigation, landing view, session evidence strip ([#2675](https://github.com/Sinity/polylogue/issues/2675)) ([79fb50d](https://github.com/Sinity/polylogue/commit/79fb50d9450f24a52cb84c7d060265c6ca70f2cd))


### Fixed

* **agent:** align quick raw-debt status ([28db963](https://github.com/Sinity/polylogue/commit/28db9636e2843cdd6ebcd43de73259a2c233e096))
* **agent:** allow active devloop backlog ([5816a75](https://github.com/Sinity/polylogue/commit/5816a7599067cb5347cedf16b25a9d801c3cf039))
* **agent:** include raw debt in quick devloop status ([d5589c4](https://github.com/Sinity/polylogue/commit/d5589c418635a8cfcf3525419b54857e0d6ba722))
* **agent:** keep devloop probes on quick status ([7bd0440](https://github.com/Sinity/polylogue/commit/7bd044094b22e6d98eeb4bc0ec4fc1d20383f8ee))
* **agent:** keep quick status off raw debt scans ([2743c46](https://github.com/Sinity/polylogue/commit/2743c46fd680349060154e26b1b008eb51afac4b))
* **agent:** resolve devloop repo roots dynamically ([#2505](https://github.com/Sinity/polylogue/issues/2505)) ([79290e6](https://github.com/Sinity/polylogue/commit/79290e6e6ea718aad5de0bb5d015eeb91c42699a))
* **archive:** report evidence and convergence honestly ([#2502](https://github.com/Sinity/polylogue/issues/2502)) ([7120bde](https://github.com/Sinity/polylogue/commit/7120bde619a1ffaaef6a82afcba13de904adf80f))
* **attachments:** acquire Claude extracted payloads ([60d93b6](https://github.com/Sinity/polylogue/commit/60d93b618e2d8cbd73dce8176ee2c9267fdfb8af))
* batch deep-read defects — convergence probes, conn leaks, reset gate, parser ([#2543](https://github.com/Sinity/polylogue/issues/2543)) ([9ed21a6](https://github.com/Sinity/polylogue/commit/9ed21a613d3f5c036434bc44a7f953321c240a17))
* **beads:** add timestamp-aware guard against post-checkout/post-merge revert ([209c9db](https://github.com/Sinity/polylogue/commit/209c9db223286a70667c6a1fc64063d9d704df28))
* bound daemon HTTP archive-query concurrency, add per-request timeout ([#2628](https://github.com/Sinity/polylogue/issues/2628)) ([f6c1da9](https://github.com/Sinity/polylogue/commit/f6c1da997bea64bc6cd9670d9cbb8f7e7439ec51))
* **browser-capture:** bound the outbound post-command queue (gnie [#4](https://github.com/Sinity/polylogue/issues/4)) ([#2568](https://github.com/Sinity/polylogue/issues/2568)) ([28a263f](https://github.com/Sinity/polylogue/commit/28a263fcd217a4b611824edab6649d6705e4be56))
* **browser-capture:** report stale archived captures ([c972c84](https://github.com/Sinity/polylogue/commit/c972c843400b3fde9504f409fbebe9295bd64123))
* **browser-capture:** require an auto-minted bearer token by default ([#2571](https://github.com/Sinity/polylogue/issues/2571)) ([f39a7fb](https://github.com/Sinity/polylogue/commit/f39a7fb6c19899a8a134c8db846723fc276c0861))
* **browser-extension:** bound asset acquisition by wall-clock and failure streaks ([#2672](https://github.com/Sinity/polylogue/issues/2672)) ([73a0654](https://github.com/Sinity/polylogue/commit/73a0654aa115ae6e6af597f122dfbaa35c950507))
* **chatgpt:** strip inline citation markers; keep injected context messages ([#2668](https://github.com/Sinity/polylogue/issues/2668)) ([9f54f7f](https://github.com/Sinity/polylogue/commit/9f54f7faf6e06d4da95e387f69588629fbb90056))
* classify blockless thinking messages as reasoning ([#2634](https://github.com/Sinity/polylogue/issues/2634)) ([bdc7b73](https://github.com/Sinity/polylogue/commit/bdc7b73ad8500319cf2c8c9084856be9a3beb840))
* **cli:** bound daemon fast-path probes ([f458002](https://github.com/Sinity/polylogue/commit/f458002985838c379074cd6f1e321395f581eddb))
* **cli:** bound root find row output ([54dd217](https://github.com/Sinity/polylogue/commit/54dd217a0b9733d8c050a5b306c578eae09143c4))
* **cli:** compact ops status JSON by default ([62862df](https://github.com/Sinity/polylogue/commit/62862dfaabbd4b3ef1c425a6a963332932b9a8cd))
* **cli:** count queried analyze results from search scope ([2726f5a](https://github.com/Sinity/polylogue/commit/2726f5a5a186299b336d3bb49cbc5c6fbcc6ae47))
* **cli:** honor JSON format for select ([#2517](https://github.com/Sinity/polylogue/issues/2517)) ([3baa238](https://github.com/Sinity/polylogue/commit/3baa238f87b2e2f12be1f5aaa24093dcd208470a))
* **cli:** mark analyze totals as partial during convergence ([37e192b](https://github.com/Sinity/polylogue/commit/37e192b346e5ce45f1d2e00f73a4580b5ffabf54))
* **cli:** resolve exact refs before root search ([08d3a5b](https://github.com/Sinity/polylogue/commit/08d3a5b8867f70f7e4acd16a9ece60fc69eb122e))
* **cli:** route structured queries without FTS ([c58f93e](https://github.com/Sinity/polylogue/commit/c58f93e2b7b63612d0766585277ae933ae32436e))
* **cli:** warn when empty results may be partial ([41b7e74](https://github.com/Sinity/polylogue/commit/41b7e7409573cdb5fbe11b303e082a6b1aff856a))
* **codex:** omit inline image data from tool text ([6d3871f](https://github.com/Sinity/polylogue/commit/6d3871f871d9b592981ad758bbaec57cb7c675c3))
* **context:** coerce non-user assertion authors to candidate at one chokepoint ([#2572](https://github.com/Sinity/polylogue/issues/2572)) ([39f0765](https://github.com/Sinity/polylogue/commit/39f0765a536a1268f64890aae49d5c5576641423))
* **context:** enforce assertion guidance trust boundary ([#2677](https://github.com/Sinity/polylogue/issues/2677)) ([ce65396](https://github.com/Sinity/polylogue/commit/ce65396c2c80da8a652feaf71748cec6f5a15549))
* **convergence:** classify stale raw parse failures ([884efb5](https://github.com/Sinity/polylogue/commit/884efb5f92d76b065bedc1d22cf8dc836fe31201))
* **convergence:** restore append-only raw blob spans ([cbb27fb](https://github.com/Sinity/polylogue/commit/cbb27fbd030ca78ff03b7947860ed106243ac081))
* **coordination:** compact agent status projections ([#2656](https://github.com/Sinity/polylogue/issues/2656)) ([#2656](https://github.com/Sinity/polylogue/issues/2656)) ([de7f2b9](https://github.com/Sinity/polylogue/commit/de7f2b90960f6fc9af2733c2625ed6af81280aa8))
* **coordination:** resolve logical caller identity ([#2665](https://github.com/Sinity/polylogue/issues/2665)) ([#2665](https://github.com/Sinity/polylogue/issues/2665)) ([cfa0839](https://github.com/Sinity/polylogue/commit/cfa0839d2b7d34dc7673f7d0cd777b1f2523d37e))
* **core:** consolidate 6 divergent _parse_archive_datetime copies ([#2567](https://github.com/Sinity/polylogue/issues/2567)) ([f54e87e](https://github.com/Sinity/polylogue/commit/f54e87ea123bf89ae7a9c52be0fefc04d338ee7e))
* **cost:** bill subscription output credits at 5x input rate ([#2484](https://github.com/Sinity/polylogue/issues/2484)) ([3c1bbb3](https://github.com/Sinity/polylogue/commit/3c1bbb3f3248a2013edb3d8933539add2587723c))
* **cost:** count Codex cumulative tokens per session, not per model ([#2489](https://github.com/Sinity/polylogue/issues/2489)) ([22d8bb5](https://github.com/Sinity/polylogue/commit/22d8bb50789d6ebec0fe59ae26eec16920c506d0))
* **cost:** flag paid models missing a catalog cache rate instead of $0 ([#2487](https://github.com/Sinity/polylogue/issues/2487)) ([e851b33](https://github.com/Sinity/polylogue/commit/e851b33deed3a79f560704c00104349098fcd6e1))
* **cost:** keep Codex message token lanes disjoint ([#2647](https://github.com/Sinity/polylogue/issues/2647)) ([8c9dfbb](https://github.com/Sinity/polylogue/commit/8c9dfbb0044ea73a1629731b14347d2ac56d3388))
* **daemon:** align browser capture spool watching ([3b90e9e](https://github.com/Sinity/polylogue/commit/3b90e9e33eb1e66f50e019a822a18622bc4f57ca))
* **daemon:** back off filtered retry paths ([c92ecd5](https://github.com/Sinity/polylogue/commit/c92ecd5f99588d3d4ac0ddf780d71343dbde759c))
* **daemon:** back off no-op catch-up chunks ([95fe147](https://github.com/Sinity/polylogue/commit/95fe1475b1012ac5b301e7a728b2cd2cb081a94e))
* **daemon:** back off no-op retry batches ([c982135](https://github.com/Sinity/polylogue/commit/c982135f96f41169651aac7f29fa8a5c9cf673fa))
* **daemon:** bound ordinary revision replay ([#2689](https://github.com/Sinity/polylogue/issues/2689)) ([8030555](https://github.com/Sinity/polylogue/commit/803055541011a3057477d9a4ac9c6396b06890c8))
* **daemon:** close two stored-XSS bug classes in the web shell (2n39) ([#2569](https://github.com/Sinity/polylogue/issues/2569)) ([7002c9f](https://github.com/Sinity/polylogue/commit/7002c9f2e818a96b82b5681bcf90bc23c8b84b41))
* **daemon:** decode session route ids ([42c3ed6](https://github.com/Sinity/polylogue/commit/42c3ed6fd4f01cfa1c8962d310a19ba2131647e2))
* **daemon:** default /api/provider-usage to headline detail, not full ([#2560](https://github.com/Sinity/polylogue/issues/2560)) ([3306866](https://github.com/Sinity/polylogue/commit/33068663ab6c4e6ce10ada7f94881bb017e989e6))
* **daemon:** enable authority-safe raw convergence ([#2688](https://github.com/Sinity/polylogue/issues/2688)) ([df559d8](https://github.com/Sinity/polylogue/commit/df559d8b3047f3db096a01e5c6b0ee6d7a2937ba))
* **daemon:** Host-admission gate, receiver hardening, spool governor ([#2559](https://github.com/Sinity/polylogue/issues/2559)) ([bfd247d](https://github.com/Sinity/polylogue/commit/bfd247d5646c66786eacb8351874625eb2ba9b3a))
* **daemon:** mark active index rebuilds not ready ([cc4698a](https://github.com/Sinity/polylogue/commit/cc4698a885c84be2956e2ede650c0dd4870232e3))
* **daemon:** route web facets through facade ([2e02aa8](https://github.com/Sinity/polylogue/commit/2e02aa8e1fe794219b72b3257989e3fc9cab9138))
* **daemon:** serialize archive writers across runtime loops ([#2676](https://github.com/Sinity/polylogue/issues/2676)) ([29e5b45](https://github.com/Sinity/polylogue/commit/29e5b45524b850d4bbdb28e1ba9731ab82377317))
* **daemon:** share bounded embedding readiness ([19e493b](https://github.com/Sinity/polylogue/commit/19e493b5785f62b570d71c8c3f8ae456abd6eadc))
* **debt:** classify empty Claude raw artifacts ([16c42ef](https://github.com/Sinity/polylogue/commit/16c42efb0d5ec2062749feabd0f1dd5c665cbf9c))
* **debt:** mark parsed raw materialization repairable ([3afc6e1](https://github.com/Sinity/polylogue/commit/3afc6e15ffe10c62ce65f30ff34637e5a8bd6f3b))
* **demo:** add acknowledgment-window sensitivity ([8a12256](https://github.com/Sinity/polylogue/commit/8a12256ad9f22939e6d86740a8a2b927f8f9bc98))
* **demo:** calibrate failure acknowledgment markers ([ca76f2d](https://github.com/Sinity/polylogue/commit/ca76f2df1071c71d93cce0a63e35195f35cc9695))
* **demo:** explain claim classifier decisions ([b4bb351](https://github.com/Sinity/polylogue/commit/b4bb351147880f7a650b1e8a168e6e05b6f3183b))
* **demo:** expose claim evidence samples by origin ([c75dc47](https://github.com/Sinity/polylogue/commit/c75dc4737e8dddcf863f81701ccba08c9ef2e62b))
* **demos:** close generated shelf over repository inputs ([#2651](https://github.com/Sinity/polylogue/issues/2651)) ([#2651](https://github.com/Sinity/polylogue/issues/2651)) ([c6aa6a0](https://github.com/Sinity/polylogue/commit/c6aa6a05d2a64309a2dbed52bbdda75d495e1441))
* **demo:** show claim evidence follow-up previews ([aef35bf](https://github.com/Sinity/polylogue/commit/aef35bfbb5340d015c9eae555f2adcf306d2978d))
* **demo:** split ambiguous failure follow-ups ([1d9187d](https://github.com/Sinity/polylogue/commit/1d9187d532b6fcafdfe33bfe41113e20f4eb24b8))
* **demo:** split failure outcomes by handler class ([6e35892](https://github.com/Sinity/polylogue/commit/6e35892a1d5b32f50b00cfdc0ad0d1cf73b5e042))
* **demo:** state claim sample frames explicitly ([4640501](https://github.com/Sinity/polylogue/commit/46405011f667290b3b2a5cdcb719805453f457d8))
* **demo:** stratify claim evidence samples by origin ([2ce5e2f](https://github.com/Sinity/polylogue/commit/2ce5e2f433bcfc677eb64053cfb5b9b0e0ffb6dd))
* **devloop:** enforce demo shelf currentness ([943aebd](https://github.com/Sinity/polylogue/commit/943aebd8945d6d5928c45c9397f4e0687173d618))
* **devtools:** align Claude cost probe to cache horizon ([e5d39c0](https://github.com/Sinity/polylogue/commit/e5d39c08f6b38e4edcda0ad4d82de1e85cd52e14))
* **devtools:** detect squash-equivalent worktrees ([#2501](https://github.com/Sinity/polylogue/issues/2501)) ([f41e078](https://github.com/Sinity/polylogue/commit/f41e078903edd6327459a609364aeb9755f6a6ac))
* **devtools:** generate demo shelf navigation ([7aba1c8](https://github.com/Sinity/polylogue/commit/7aba1c8bf9d8765b948d198d20b30624e6ae93de))
* **devtools:** keep dev-loop daemon source-only ([c37cb0d](https://github.com/Sinity/polylogue/commit/c37cb0dc390c539e0a121993d1e970d7977c9587))
* **devtools:** key verify's stall detector off test-event progress ([#2581](https://github.com/Sinity/polylogue/issues/2581)) ([85fa450](https://github.com/Sinity/polylogue/commit/85fa4505585b46c37b6c03c10121babf201e0e77))
* **docs:** repair generated pages links ([#2500](https://github.com/Sinity/polylogue/issues/2500)) ([a9985f4](https://github.com/Sinity/polylogue/commit/a9985f4115d057c7de0842c1a23414492b3c1efe))
* **embed:** account for skipped catchup sessions ([f316d4b](https://github.com/Sinity/polylogue/commit/f316d4b10f9cd2ecd08b37a0128587dd61fea13f))
* **embed:** align status text fields ([6a4b4ca](https://github.com/Sinity/polylogue/commit/6a4b4ca4bba681f9177e72bc580e0f4eddb4e9e3))
* **embed:** avoid false zero cost in status detail ([510fe18](https://github.com/Sinity/polylogue/commit/510fe188f8ecf90cb77c1b1084c4a8303a692f68))
* **embed:** bound archive backfill windows ([1963a01](https://github.com/Sinity/polylogue/commit/1963a0129022162b1ea519fcb4e48e52483f2017))
* **embed:** bound archive pending windows after eligibility ([47cb6f9](https://github.com/Sinity/polylogue/commit/47cb6f9fcd1e509b03ab16d77104596ded6d49e8))
* **embed:** bound default archive status counts ([2562cd4](https://github.com/Sinity/polylogue/commit/2562cd4dc343bbe9091ac6811ea4d8ceffa080d2))
* **embed:** bound exact archive detail status ([d7403e2](https://github.com/Sinity/polylogue/commit/d7403e2ad6e1735a193035337be872dc90437e0d))
* **embed:** count only prose in embedding stats ([#2519](https://github.com/Sinity/polylogue/issues/2519)) ([4601922](https://github.com/Sinity/polylogue/commit/46019227d125ae113d5bdad8fc7496d28ace082d))
* **embeddings:** count exact prose in pending windows ([6ae15a9](https://github.com/Sinity/polylogue/commit/6ae15a909ed107bb1d75f251f05d0c6a27f335b9))
* **embeddings:** detect same-count message edits ([d07627e](https://github.com/Sinity/polylogue/commit/d07627ebeffb41100d4da82a1390dd8645d4dcae))
* **embeddings:** preserve unknown status counts ([#2661](https://github.com/Sinity/polylogue/issues/2661)) ([69990dc](https://github.com/Sinity/polylogue/commit/69990dcc873c2fc0a9c900861bb10db94b75f434))
* **embeddings:** report manual backfill progress ([3eb5efa](https://github.com/Sinity/polylogue/commit/3eb5efaa4fa7f53960a1fd50386c505f6bde1275))
* **embeddings:** reselect stale clean archive sessions ([b998ec4](https://github.com/Sinity/polylogue/commit/b998ec4cfc28cde2606c89631a451b3b0133fd47))
* **embeddings:** stop retrying terminal provider errors ([108d71f](https://github.com/Sinity/polylogue/commit/108d71f3e5abe826fd2f937b3010a3679ff7b002))
* **embed:** distinguish session and prose coverage ([c1c1b1a](https://github.com/Sinity/polylogue/commit/c1c1b1a4144cc6c27164e8ef56884fe0cec4897c))
* **embed:** emit structured backfill results ([#2518](https://github.com/Sinity/polylogue/issues/2518)) ([2993141](https://github.com/Sinity/polylogue/commit/299314105b0747a3d16a9d1b187ab3b6cc4b3df4))
* **embed:** expose preflight min-message floor ([#2506](https://github.com/Sinity/polylogue/issues/2506)) ([12e7cdf](https://github.com/Sinity/polylogue/commit/12e7cdfd2fa877b8c094d5a4560c68e3ceb7dc94))
* **embed:** keep backfill windows under message cap ([#2516](https://github.com/Sinity/polylogue/issues/2516)) ([18df4c4](https://github.com/Sinity/polylogue/commit/18df4c457445c016d29c7a1ceecbcfcfa1c4624c))
* **embed:** keep bounded backfill windows cheap ([3441570](https://github.com/Sinity/polylogue/commit/344157058cab0eb23a64b1a74324d2b9e9cfee6b))
* **embed:** keep partial vectors retrieval-ready ([#2515](https://github.com/Sinity/polylogue/issues/2515)) ([aa13e1f](https://github.com/Sinity/polylogue/commit/aa13e1f73f85eefe347d9f8869d0a2e98825c25b))
* **embed:** keep status responsive during backfill ([33dae38](https://github.com/Sinity/polylogue/commit/33dae38e0e39a86a4b7ab01f0c097a8f3fe940f3))
* **embed:** make status backfill actions executable ([9ec46c8](https://github.com/Sinity/polylogue/commit/9ec46c89dcaccecff10892961472e3ab77d23891))
* **embed:** persist archive backfill runs in ops ([093fd81](https://github.com/Sinity/polylogue/commit/093fd814455174299f0a7718b7485b30bcf3bf68))
* **embed:** report archive catchup session counts ([7eb514e](https://github.com/Sinity/polylogue/commit/7eb514e955006da5f9ead6db307f20338542dec4))
* **embed:** report archive embedding metadata fast ([b187f5d](https://github.com/Sinity/polylogue/commit/b187f5d1f65c14d969506853cc7ee501909416ba))
* **embed:** report archive embedding timestamp bounds ([1f0d6b8](https://github.com/Sinity/polylogue/commit/1f0d6b80f75f882da6ce56378770af4b0c69ad7c))
* **embed:** summarize uniform archive metadata fast ([6d3a84d](https://github.com/Sinity/polylogue/commit/6d3a84d3069247e6ca9bc79023ea98be73a4284c))
* **embed:** surface material catchup progress ([fc24f58](https://github.com/Sinity/polylogue/commit/fc24f58b53560c21074d6b40dc766244ea6a7c27))
* **embed:** treat completed sessions as embedded ([#2514](https://github.com/Sinity/polylogue/issues/2514)) ([6cd054d](https://github.com/Sinity/polylogue/commit/6cd054d652cffcdf27860e769e9a15451cddd08b))
* **extension:** make capture popup readable ([1bfb0a3](https://github.com/Sinity/polylogue/commit/1bfb0a3138da769fc62a7c274321c29331afa62e))
* **extension:** stop automatic page capture ([a35d521](https://github.com/Sinity/polylogue/commit/a35d5210110b6df6a5adfdbee93ce35c4f79387f))
* **fts:** chunk missing-row repair ([6406558](https://github.com/Sinity/polylogue/commit/6406558618b2a0e216b5e131bfbc5f83a1b8f3d1))
* harden cloud bootstrap for sandbox lanes ([#2631](https://github.com/Sinity/polylogue/issues/2631)) ([c68585b](https://github.com/Sinity/polylogue/commit/c68585b8bf3a8e83f57faa364fff108a23ed504a))
* **hermes:** make state database imports reproducible ([#2639](https://github.com/Sinity/polylogue/issues/2639)) ([#2639](https://github.com/Sinity/polylogue/issues/2639)) ([9e92b6b](https://github.com/Sinity/polylogue/commit/9e92b6b6d7656f315dd491b8f5f59049d104e868))
* **import:** keep newer session body on stale reimport ([9beac9a](https://github.com/Sinity/polylogue/commit/9beac9a5300125eb0f4362d8c8ce41ab033db6ac))
* **ingest:** finalize live raw parse state ([#2663](https://github.com/Sinity/polylogue/issues/2663)) ([938c671](https://github.com/Sinity/polylogue/commit/938c671e5c5c68454b77592d84c0c726f53819e7))
* **ingest:** preserve richer browser captures ([c907759](https://github.com/Sinity/polylogue/commit/c9077590a279037c530dd309f37b886f9052c594))
* **ingest:** publish full results after raw completion ([#2664](https://github.com/Sinity/polylogue/issues/2664)) ([0cccef1](https://github.com/Sinity/polylogue/commit/0cccef1df3b8618c6e6dbe3e3a4b6860ab8cebc3))
* **insights:** add text_derived provenance markers to forensic payloads ([#2579](https://github.com/Sinity/polylogue/issues/2579)) ([90d3e31](https://github.com/Sinity/polylogue/commit/90d3e31747de58f3a7cd354866504ca4ec49ea2c))
* **insights:** audit every registered product, not just contracted ones ([#2573](https://github.com/Sinity/polylogue/issues/2573)) ([7b5f846](https://github.com/Sinity/polylogue/commit/7b5f846f57073d931737962dcabc17d5ef962dc0))
* **insights:** render None, not 0.0, for averages over empty backing rows ([#2585](https://github.com/Sinity/polylogue/issues/2585)) ([f8f3e40](https://github.com/Sinity/polylogue/commit/f8f3e40a5b7f548e6a9530d3e9f48c9964345e79))
* **insights:** scope run-projection row refs ([#2539](https://github.com/Sinity/polylogue/issues/2539)) ([b9d01d7](https://github.com/Sinity/polylogue/commit/b9d01d7aa34a9ee106c5122dbb8f4d2f1a8bd457))
* **insights:** stop laundering weak temporal provenance into provider_ts ([#2558](https://github.com/Sinity/polylogue/issues/2558)) ([89b14e5](https://github.com/Sinity/polylogue/commit/89b14e5876749fdc66f26745ce507ba76ebc788a))
* **insights:** treat run projections as optional cache ([40f22dd](https://github.com/Sinity/polylogue/commit/40f22dd804ab9dbfa610a8111ce0c5cd981b3dd0))
* **insights:** upsert run-projection rows on cross-session ref collisions ([#2464](https://github.com/Sinity/polylogue/issues/2464)) ([15f4f21](https://github.com/Sinity/polylogue/commit/15f4f21b63f662f13fb4f16b5379d7dfaad17627))
* **lineage:** hold one read transaction across composition ([#2594](https://github.com/Sinity/polylogue/issues/2594)) ([0861717](https://github.com/Sinity/polylogue/commit/0861717015066fe602d363df2440afadec7deea3))
* **lineage:** iterative transcript composition, remove effective depth cap ([#2542](https://github.com/Sinity/polylogue/issues/2542)) ([1e4a694](https://github.com/Sinity/polylogue/commit/1e4a694382a7a76ec7b2391f9309d02ab1a595cf))
* **logging:** forward exc_info/extra in stdlib logger shim ([#2548](https://github.com/Sinity/polylogue/issues/2548)) ([ee1a51c](https://github.com/Sinity/polylogue/commit/ee1a51cb69bb9bfcba7fd577a843c655a42e1e7a))
* **maintenance:** clear session insight debt ([5ab2352](https://github.com/Sinity/polylogue/commit/5ab23525d17df4ef5f5e295b90aeae3fcfa1f982))
* **maintenance:** ignore parsed raw sidecars ([#2511](https://github.com/Sinity/polylogue/issues/2511)) ([1d5d112](https://github.com/Sinity/polylogue/commit/1d5d1126943648d18bfcb2c55520942e0f807347))
* **maintenance:** make index rebuilds plannable and FK-clean ([#2522](https://github.com/Sinity/polylogue/issues/2522)) ([41d4bb4](https://github.com/Sinity/polylogue/commit/41d4bb4a7be28376c1ace1e46e3ca31f0962c879))
* **maintenance:** replay parsed raw rows after index reset ([#2508](https://github.com/Sinity/polylogue/issues/2508)) ([d1bbb3e](https://github.com/Sinity/polylogue/commit/d1bbb3e02bfb09d55ba0e2ef6a1f69be5db37408))
* **maintenance:** resume interrupted raw materialization ([#2510](https://github.com/Sinity/polylogue/issues/2510)) ([b55ea52](https://github.com/Sinity/polylogue/commit/b55ea528e9c3d856c9ef2bf144b8ccba5a1e01b3))
* make partial resume profiles explicit ([#2635](https://github.com/Sinity/polylogue/issues/2635)) ([186daf2](https://github.com/Sinity/polylogue/commit/186daf287ae2b04994f7f6afe5b0564b6f947584))
* **mcp:** report split-tier embedding stats ([c70081b](https://github.com/Sinity/polylogue/commit/c70081b69728b678a66371bf43e24d00eb6c267b))
* **mcp:** stop silently capping aggregate totals ([29c020d](https://github.com/Sinity/polylogue/commit/29c020d4b2f3dd58f3135c8ce5ff810a45d3a020))
* **ops:** point index resets at source replay ([e84c2ae](https://github.com/Sinity/polylogue/commit/e84c2aef8a75f8ae196643b2364e869f234350f6))
* parse ChatGPT recipient-addressed tool calls as TOOL_USE, not raw text ([#2629](https://github.com/Sinity/polylogue/issues/2629)) ([7688d10](https://github.com/Sinity/polylogue/commit/7688d103d5393e9bc3d6b249bcd6daa656eee58a))
* **parsers:** keep asset-only chat turns ([607d79b](https://github.com/Sinity/polylogue/commit/607d79bddcff4f95eb046f0f80857d2ca656a515))
* **read:** degrade bounded reads to content ([4eb337b](https://github.com/Sinity/polylogue/commit/4eb337b36d139e806ccd1feacba904b6eb60141f))
* **readiness:** classify explained raw join gaps ([d9bc786](https://github.com/Sinity/polylogue/commit/d9bc786b8619e2d7a0ea3c80fdecbbb4a838cda3))
* **readiness:** close verifier failures ([#2547](https://github.com/Sinity/polylogue/issues/2547)) ([4b9389d](https://github.com/Sinity/polylogue/commit/4b9389d752b53742fc70df5364d221b16a292f9d))
* **readiness:** expose fast archive convergence state ([c9ded08](https://github.com/Sinity/polylogue/commit/c9ded089ca53d61baa883e5915a3b2248c621250))
* **readiness:** report materialization progress counts ([f5d848e](https://github.com/Sinity/polylogue/commit/f5d848e87cf8946e3d981cede9ff58401cdf7cb8))
* **read:** use session links for streaming lineage checks ([29ed6ea](https://github.com/Sinity/polylogue/commit/29ed6ea87a9f3df390b5f9987c0dbfcd43f0c4e9))
* repair 3 flagship CLI query bugs found in prod smoke test ([#2626](https://github.com/Sinity/polylogue/issues/2626)) ([5d5edaf](https://github.com/Sinity/polylogue/commit/5d5edaf496d70b9372c7c2123c2e70a6ed4d34e6))
* restore deterministic archive contracts ([#2641](https://github.com/Sinity/polylogue/issues/2641)) ([f6b396b](https://github.com/Sinity/polylogue/commit/f6b396bf63cbe43b6e7645e94b8aafa88a2fd0b0))
* route ops reset --session/--source through the mutation contract ([#2627](https://github.com/Sinity/polylogue/issues/2627)) ([a075ce2](https://github.com/Sinity/polylogue/commit/a075ce290837f68af678f093b02f39fe42286d11))
* **search:** keep default find lexical ([faba5f3](https://github.com/Sinity/polylogue/commit/faba5f3950b308658dbbbfe9067488f65b606fcc))
* **search:** make FTS readiness cheap in fresh indexes ([a46be16](https://github.com/Sinity/polylogue/commit/a46be16d2b3a74241fe9f063f1052f4743493198))
* **security:** harden blob hash validation, drop misleading symlink check ([#2599](https://github.com/Sinity/polylogue/issues/2599)) ([b6b9fef](https://github.com/Sinity/polylogue/commit/b6b9fef2a05ed7d5a1272188ce852f74a133b637))
* **sources:** close CursorStore get-&gt;modify-&gt;put lost-update race ([#2615](https://github.com/Sinity/polylogue/issues/2615)) ([fac493c](https://github.com/Sinity/polylogue/commit/fac493cb3cc37edfa58e939b367c78e069730304))
* **sources:** merge Claude Code stream sessions ([#2520](https://github.com/Sinity/polylogue/issues/2520)) ([a651bb9](https://github.com/Sinity/polylogue/commit/a651bb95f1fc3595cff8f545ec3768831b858daf))
* **sources:** restrict append planning to JSONL streams ([#2691](https://github.com/Sinity/polylogue/issues/2691)) ([fae9e0b](https://github.com/Sinity/polylogue/commit/fae9e0bb580caaac18f94b28674f104c5b9ca6b8))
* stamp focused test runs with git head ([#2632](https://github.com/Sinity/polylogue/issues/2632)) ([4dabc85](https://github.com/Sinity/polylogue/commit/4dabc85dd05f6c0d710e6755853a2bf70b4b0a2f))
* **status:** gate readiness on raw materialization ([91bf2d7](https://github.com/Sinity/polylogue/commit/91bf2d7fb8a43601f5b046ccddb974f29c6b2beb))
* **status:** ignore stale rebuild attempts for readiness ([f2f1a21](https://github.com/Sinity/polylogue/commit/f2f1a2156dbcfc1bc6e4f23ed449dfb16d37cac5))
* **status:** preserve unknown embedding pending messages ([36446ec](https://github.com/Sinity/polylogue/commit/36446ecbf0c79497712f3149509d1e8ae1e65b03))
* **status:** project durable raw authority into readiness ([#2695](https://github.com/Sinity/polylogue/issues/2695)) ([93af1a5](https://github.com/Sinity/polylogue/commit/93af1a53a8b68d6500c69ddfc6060d542c6d0d8d))
* **status:** show SQLite maintenance state ([f7d2539](https://github.com/Sinity/polylogue/commit/f7d2539a4380c8e7b3d86499f6eb9d248af70726))
* **status:** use fast materialization readiness ([80dd748](https://github.com/Sinity/polylogue/commit/80dd748cce2c4b0e86927605f1cf605fc57ad490))
* **status:** validate actions view readiness ([#2687](https://github.com/Sinity/polylogue/issues/2687)) ([9018d58](https://github.com/Sinity/polylogue/commit/9018d5861a699d595da42a56d9df95ad5d4fdd24))
* **storage:** add offline index generation promotion ([#2685](https://github.com/Sinity/polylogue/issues/2685)) ([a2bbd25](https://github.com/Sinity/polylogue/commit/a2bbd25d6a315791430e11191c3b2070cdfdbe2f))
* **storage:** compose forks on paginated, batch, and streaming reads ([#2485](https://github.com/Sinity/polylogue/issues/2485)) ([8664ee8](https://github.com/Sinity/polylogue/commit/8664ee872286811f5094e125cbcf78559c6b7499)), closes [#2470](https://github.com/Sinity/polylogue/issues/2470)
* **storage:** contain authority-ambiguous raw replay ([#2670](https://github.com/Sinity/polylogue/issues/2670)) ([#2670](https://github.com/Sinity/polylogue/issues/2670)) ([202a09c](https://github.com/Sinity/polylogue/commit/202a09c240bc56a74c7d078663aafc7c7f97e59c))
* **storage:** contain conflicting archive index identities ([#2680](https://github.com/Sinity/polylogue/issues/2680)) ([36147f2](https://github.com/Sinity/polylogue/commit/36147f29c7255474071bce869e6942c02ce3e348))
* **storage:** enforce monotonic raw revision replay ([#2684](https://github.com/Sinity/polylogue/issues/2684)) ([6a579d0](https://github.com/Sinity/polylogue/commit/6a579d09021c9b0ed128287ace51cdff077cd5b7))
* **storage:** gate embedding success write against config-change race ([#2616](https://github.com/Sinity/polylogue/issues/2616)) ([16fdb0f](https://github.com/Sinity/polylogue/commit/16fdb0fcacba55f9640222a3e0e52984f6c414aa))
* **storage:** include timeless sessions in search --since filters ([#2578](https://github.com/Sinity/polylogue/issues/2578)) ([97459e6](https://github.com/Sinity/polylogue/commit/97459e61d4fae716594eb871381f865a4f79593e))
* **storage:** include timeless work-events/phases in since/until windows ([#2577](https://github.com/Sinity/polylogue/issues/2577)) ([1b29fa5](https://github.com/Sinity/polylogue/commit/1b29fa52460cbeca78656be4536531cf66cb6d1f))
* **storage:** pair actions view by transcript rank, not tool_id equality ([#2597](https://github.com/Sinity/polylogue/issues/2597)) ([7b5a5aa](https://github.com/Sinity/polylogue/commit/7b5a5aa0589772ed9d7eb9632af1591cf22cdd33))
* **storage:** preserve semantic replay frontiers ([#2692](https://github.com/Sinity/polylogue/issues/2692)) ([7d300a5](https://github.com/Sinity/polylogue/commit/7d300a59628742cf739706bd5ebe225daf9b4825))
* **storage:** preserve source evidence during reset ([#2537](https://github.com/Sinity/polylogue/issues/2537)) ([1f629c9](https://github.com/Sinity/polylogue/commit/1f629c9d2d923789f0dbf7167ffb021b13b3707d))
* **storage:** remove unreachable blob-GC lease mechanism ([#2617](https://github.com/Sinity/polylogue/issues/2617)) ([a263612](https://github.com/Sinity/polylogue/commit/a2636120495d1b87a8e5170c7d108144c3bc7f4b))
* **storage:** reserve blobs until durable references commit ([#2660](https://github.com/Sinity/polylogue/issues/2660)) ([a0ef2fa](https://github.com/Sinity/polylogue/commit/a0ef2fa8d479a0168db36fe09f3752de6311b26e))
* **storage:** retire governed bundle raws ([#2693](https://github.com/Sinity/polylogue/issues/2693)) ([304b840](https://github.com/Sinity/polylogue/commit/304b84019d7dfa0474a8db4685166849af16d297))
* **storage:** route ops-doctor orphan-blob repair through the safe GC planner ([#2564](https://github.com/Sinity/polylogue/issues/2564)) ([8e4efb7](https://github.com/Sinity/polylogue/commit/8e4efb72379f458d23524c00fb2de29346ce0e0b))
* **storage:** single-source FTS freshness DDL ([c769ea7](https://github.com/Sinity/polylogue/commit/c769ea7b7cd6b04da1d54314809d5a093a62471c))
* **storage:** slice provider usage on prefix-sharing forks ([4628cd3](https://github.com/Sinity/polylogue/commit/4628cd30f24ea72546ed2d53512d093e2aea8501))
* **storage:** stop epoch-pinning timeless rows in the query CLI unit engine ([#2576](https://github.com/Sinity/polylogue/issues/2576)) ([c67fd45](https://github.com/Sinity/polylogue/commit/c67fd45392a693428d504c95359c7e4bdfcd9a53))
* **storage:** stop replaying terminal raw revisions ([#2690](https://github.com/Sinity/polylogue/issues/2690)) ([90cf639](https://github.com/Sinity/polylogue/commit/90cf639b151c0c16dbb74ff70f7132c09297f6e7))
* **storage:** stop scheduling governed replay debt ([#2694](https://github.com/Sinity/polylogue/issues/2694)) ([63a6c75](https://github.com/Sinity/polylogue/commit/63a6c756326ba43238bee48b8de24eab657bfb17))
* **storage:** stop usage_timeline silently dropping timeless-session usage ([#2575](https://github.com/Sinity/polylogue/issues/2575)) ([a31d096](https://github.com/Sinity/polylogue/commit/a31d0968581d108fba5a6261d41e939e3e4907ab))
* **storage:** surface SQLite maintenance drift ([e50b923](https://github.com/Sinity/polylogue/commit/e50b923168de1e2ec8a7c81a4f11f30cc0c724de))
* **storage:** trigram-index the affordance-usage CLI-detection substring scan ([#2622](https://github.com/Sinity/polylogue/issues/2622)) ([f5c35e7](https://github.com/Sinity/polylogue/commit/f5c35e7021dceebfb2836e4c84377fe8e0bbd457))
* **storage:** trust stale FTS readiness rows ([05a30c0](https://github.com/Sinity/polylogue/commit/05a30c0a8c652669305675fc29f0bd4cf55ea808))
* **test:** repair full-verify baseline drift from 2026-07-05 commit batch ([#2556](https://github.com/Sinity/polylogue/issues/2556)) ([a836e4b](https://github.com/Sinity/polylogue/commit/a836e4bfc9d8830d30b0995efbee4e2ada71e3cd))
* **topology:** integrate hook service placement ([#2658](https://github.com/Sinity/polylogue/issues/2658)) ([#2658](https://github.com/Sinity/polylogue/issues/2658)) ([a8eb1bf](https://github.com/Sinity/polylogue/commit/a8eb1bf1a8af6fdf3257f022b49935b07ad5dc73))
* **types:** restore quick verification gate ([87ce471](https://github.com/Sinity/polylogue/commit/87ce4718768abe247bc54bc03fda7f0db7a78752))
* **usage:** reconcile Codex tokens from disjoint rollups ([cae61ed](https://github.com/Sinity/polylogue/commit/cae61ed8ad5006e83374b72bfde7a252d82889bb))
* **verify:** restore quick gate currency ([3216c02](https://github.com/Sinity/polylogue/commit/3216c020660acef86db68f51239c343816c35a93))
* **web-shell:** correct total_sessions path, clear stuck status notice ([#2562](https://github.com/Sinity/polylogue/issues/2562)) ([8506792](https://github.com/Sinity/polylogue/commit/850679209c6a551106d31104b640c728f70295f1))
* **web:** render truthful loading/stale/error states, not false emptiness ([#2673](https://github.com/Sinity/polylogue/issues/2673)) ([5479bea](https://github.com/Sinity/polylogue/commit/5479beabcb300049d9a2bbb50859d70b3742639d))
* word-boundary anchor the work-event text-signal keyword matcher ([#2630](https://github.com/Sinity/polylogue/issues/2630)) ([7806b26](https://github.com/Sinity/polylogue/commit/7806b26e76f9307931d0150ca2bed35bd1d34566))


### Changed

* **cli:** avoid select setup on bounded output ([01e592e](https://github.com/Sinity/polylogue/commit/01e592e3db300e1420b7d209e68e9744c60d9f7b))
* **cli:** delegate session pages to daemon ([01847a3](https://github.com/Sinity/polylogue/commit/01847a36d52879bfef79d404413bf4d1a09b0102))
* **demo:** bound claim evidence report scans ([9045330](https://github.com/Sinity/polylogue/commit/90453304c1cf7e737459e1a83ea4ec8240d2d7b4))
* **demo:** bound claim evidence sampling before pairing ([8112fa2](https://github.com/Sinity/polylogue/commit/8112fa24373c5c8e8bbf64c69affd6686fa4b5d7))
* **devtools:** bound affordance detail reports ([898a6d1](https://github.com/Sinity/polylogue/commit/898a6d1868925a44a603fede6dc73e28379705a9))
* **embed:** avoid exact scans during window selection ([#2512](https://github.com/Sinity/polylogue/issues/2512)) ([ba4655a](https://github.com/Sinity/polylogue/commit/ba4655afc2040bf0f59e015331032a487511cba5))
* **embed:** constrain archive block text lookup ([#2513](https://github.com/Sinity/polylogue/issues/2513)) ([da18955](https://github.com/Sinity/polylogue/commit/da18955963ac32bc0c4f88443a0926d7a4abdb82))
* **embeddings:** batch archive vector writes ([660d01d](https://github.com/Sinity/polylogue/commit/660d01dbb9ceef95daff4b4613a05aecad1c7385))
* **embeddings:** stream pending prose windows ([ff6c4f9](https://github.com/Sinity/polylogue/commit/ff6c4f950960ee1848033e9faea1c99f5e3432a5))
* **embed:** index authored prose selection ([#2507](https://github.com/Sinity/polylogue/issues/2507)) ([cdd0081](https://github.com/Sinity/polylogue/commit/cdd008167b4cfe03d08231b0200420b4353df957))
* **ingest:** batch commits by accumulated message count ([#2492](https://github.com/Sinity/polylogue/issues/2492)) ([eaa4ad7](https://github.com/Sinity/polylogue/commit/eaa4ad7e23ac9238291b9d1902f9c13ec7963432)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* **ingest:** batch session event writes ([6e681ff](https://github.com/Sinity/polylogue/commit/6e681ff40534daaac2479641d665c380cf560910))
* **ingest:** log slow write stage timings ([5da0be3](https://github.com/Sinity/polylogue/commit/5da0be374074fe6ad1fd46d911f3c5f1c38c549c))
* **ingest:** skip current root graph refreshes ([452f051](https://github.com/Sinity/polylogue/commit/452f051db880ad4d17156c981644b8b08f0241bd))
* **ingest:** skip unchanged thread refreshes ([cf6b41e](https://github.com/Sinity/polylogue/commit/cf6b41ef7a344e4577d37ba7158dadef1ddc1f33))
* **insights:** aggregate cost rollups from usage rows ([bb2f84f](https://github.com/Sinity/polylogue/commit/bb2f84ff871fb6717c94c733ddd3b963570509b1))
* **insights:** bound async giant-session rebuilds ([#2540](https://github.com/Sinity/polylogue/issues/2540)) ([3de0079](https://github.com/Sinity/polylogue/commit/3de00794ef058d68ec8707c6a977f391504b8f95))
* **insights:** bound insight-rebuild WAL via per-chunk commits (Ref [#2458](https://github.com/Sinity/polylogue/issues/2458)) ([#2466](https://github.com/Sinity/polylogue/issues/2466)) ([2eee22a](https://github.com/Sinity/polylogue/commit/2eee22a9fbfe9f177bc895f64c3271917044a91b))
* **insights:** bound usage timeline event scans ([b9f4253](https://github.com/Sinity/polylogue/commit/b9f4253cae00dfdd621281ef5ce3e1ebedbb8862))
* **maintenance:** batch raw materialization replay ([#2509](https://github.com/Sinity/polylogue/issues/2509)) ([4a39c0c](https://github.com/Sinity/polylogue/commit/4a39c0c31fa07d76e314773cf69a7e191ff474d9))
* **maintenance:** replay rebuild rows source-first ([#2524](https://github.com/Sinity/polylogue/issues/2524)) ([4a0ab58](https://github.com/Sinity/polylogue/commit/4a0ab5883a8fbcda0b558779d8066af04fc318d6))
* **maintenance:** report weighted rebuild run size ([#2529](https://github.com/Sinity/polylogue/issues/2529)) ([ad1bf4e](https://github.com/Sinity/polylogue/commit/ad1bf4e177eafde709f0bc63f627b2d01bec2a90))
* **maintenance:** scope targeted index materialization ([90f1c5a](https://github.com/Sinity/polylogue/commit/90f1c5a493e1f583149fbf60ed78cdd1467e8532))
* **maintenance:** weight rebuild planning by raw payload ([#2528](https://github.com/Sinity/polylogue/issues/2528)) ([3954534](https://github.com/Sinity/polylogue/commit/39545347c89f66e9e97ab63f59cbc206f625a185))
* **pipeline:** report raw ingest batch starts ([#2530](https://github.com/Sinity/polylogue/issues/2530)) ([e15eda2](https://github.com/Sinity/polylogue/commit/e15eda24e6118ce7c1631da3b618c07666f1205a))
* **read-package:** run artifacts in process ([ed7cafb](https://github.com/Sinity/polylogue/commit/ed7cafbf4e8bff88a94089bf4629355a3af7651f))
* **read:** avoid full-session pagination loads ([a17e3af](https://github.com/Sinity/polylogue/commit/a17e3af95dd601f07ca0aa2b946f8efaa750b2c5))
* **read:** bound exact-session temporal handoffs ([478d6a7](https://github.com/Sinity/polylogue/commit/478d6a77cf1cea1d63c180701d49099ad2cfacc3))
* **read:** stream exact markdown file exports ([a9dc3f2](https://github.com/Sinity/polylogue/commit/a9dc3f274abb92c10830b5d2dcd82f5e291cb905))
* **read:** stream messages file exports ([177523a](https://github.com/Sinity/polylogue/commit/177523aab8c9407f319eed77361fad965f0ac56b))
* **status:** keep full status off exact scans ([4399db6](https://github.com/Sinity/polylogue/commit/4399db6360c27cb1e87e316fac6604470588f576))
* **storage:** append thread membership suffixes ([adc2541](https://github.com/Sinity/polylogue/commit/adc25412e5d57865e5325fc4cef9274e100bac2f))
* **storage:** avoid redundant thread refresh churn ([af0c1cc](https://github.com/Sinity/polylogue/commit/af0c1cc1c48c37f2ba37f80fd525ad934ff3142e))
* **storage:** bound revision census memory ([#2686](https://github.com/Sinity/polylogue/issues/2686)) ([3fe7837](https://github.com/Sinity/polylogue/commit/3fe7837c83377017341958d544c62f79587bff0b))
* **storage:** cache composed lineage during graph repair ([51ddf75](https://github.com/Sinity/polylogue/commit/51ddf75a82adf1b248fdda287c12b4f5eb962043))
* **storage:** clean empty lineage tails by session ([#2525](https://github.com/Sinity/polylogue/issues/2525)) ([b2936ba](https://github.com/Sinity/polylogue/commit/b2936ba8dddd90b94c5c708f8be4569e16171344))
* **storage:** reduce graph resolve refresh churn ([#2526](https://github.com/Sinity/polylogue/issues/2526)) ([7b9f4e0](https://github.com/Sinity/polylogue/commit/7b9f4e0419875801d7cc26653b6f4858e7e3a2d4))
* **storage:** time delayed prefix re-extraction ([#2527](https://github.com/Sinity/polylogue/issues/2527)) ([04ecfe0](https://github.com/Sinity/polylogue/commit/04ecfe0ec26910fa6abb3221aa4607ede15d81e9))
* **storage:** use session-scoped block lookup for signatures ([#2532](https://github.com/Sinity/polylogue/issues/2532)) ([89b20c3](https://github.com/Sinity/polylogue/commit/89b20c33584a2a8aebef8798b1a425a1ce4d5ad9))
* **usage:** add headline detail for usage reports ([c1067fb](https://github.com/Sinity/polylogue/commit/c1067fb45cc1cf93fb6a9dd5ccd6bc03c79477a7))

## [Unreleased]

### Added

- Maintenance replay failures now surface in `polylogue status` and the
  daemon raw-failure health check (#1198). Per-record failures from
  `repair_session_insights`, `repair_action_event_read_model`, and
  `source_replay` are routed through
  `polylogue.maintenance.failure_routing.route_failure_sample` into an
  append-only JSONL file at
  `<archive_root>/.maintenance-state/failures.jsonl`, then merged into
  the existing `_raw_failure_info()` payload with `source="maintenance"`
  and the originating `operation_id`. The `raw_failures` health alert
  escalates through the existing WARNING / ERROR / CRITICAL ladder and
  cites a representative `op=<short_id>` so operators can pull the
  failing replay's resume state file directly. Absolute paths in
  routed messages and locators are redacted at write time.

- Non-log notification backends for the daemon health loop (#1233).
  `notification_backend` now accepts `"webhook"`, `"journald"`,
  `"email"`, `"apprise"`, and lists/comma-separated fan-out specs
  (e.g. `notification_backend = ["log", "apprise"]`). The webhook
  backend signs the JSON envelope with HMAC-SHA256 in
  `X-Polylogue-Signature` when `notification_webhook_secret` is set,
  and uses exponential backoff across up to three attempts. The email
  backend (`[notifications.email]` TOML table or
  `POLYLOGUE_NOTIFICATION_EMAIL_*` env vars) speaks SMTP with
  STARTTLS / implicit-TLS on port 465 and rate-limits to N
  messages/hour. The Apprise backend reads
  `notification_apprise_urls` and dispatches to 100+ services
  (Pushover, Discord, Slack, ntfy, Matrix, Telegram, …) through one
  adapter. Per-backend failures in fan-out mode are isolated so one
  broken destination does not silence the rest.

- Cycle outlook reaches the CLI and MCP surfaces (#1138). New
  `polylogue cost outlook --plan <name>` subcommand renders the typed
  `CycleOutlook` payload from #1137 — cycle window, burn rate,
  projected total, quota pressure, overage rows, coverage, and
  confidence — in JSON or plain mode. Plain mode labels USD totals as
  API-equivalent, quota figures by basis, and surfaces "quota not
  configured" explicitly so subscription-equivalent and
  API-equivalent numbers never collapse into a single unlabelled
  `cost`. The legacy flat surface is preserved as
  `polylogue cost rollup`. New MCP tool `cost_outlook(plan, method)`
  returns the same typed payload; the matching async facade method
  `Polylogue.cost_outlook(plan_name, now=None, method=...)` resolves
  plans against `[[cost.subscription.plans]]` user overrides merged
  with the curated seed.
- Learning feedback loop — corrections as deterministic rebuild signal
  (#1131). New `polylogue feedback` command group (`record`, `list`,
  `clear`) and matching MCP tools (`record_correction`,
  `list_corrections`, `clear_corrections`) let users override the
  heuristic classifier, accept/reject auto-tags, and replace generated
  summaries. Corrections live in the new `user_corrections` table (schema
  v15) outside the content-hash boundary: applying or removing a
  correction never alters `conversations.content_hash`. The
  `classify_session` insight path now consults corrections after the
  heuristic so rebuilds always produce the same merged verdict across
  runs.

### Schema

- Bump `SCHEMA_VERSION` from 14 to 15 to introduce the `user_corrections`
  table. Existing databases are rejected with the usual fresh-first
  message; an explicit upgrade script is required.

- Tool usage analytics with explicit per-provider coverage (#1133). New
  insight type `tool_usage` rolls up per-(provider, tool, action_kind)
  call counts, conversation counts, distinct tool ids, and affected-path
  / output-text density over canonical `action_events`. The same
  envelope carries a per-provider coverage map distinguishing
  "data unavailable" (provider exposes no tool events) from "zero
  observed" so coverage gaps are never collapsed into silent zeros.
  Available as `polylogue insights tool-usage` (`--tool`,
  `--mcp-server`, `--action-kind` filters), MCP tool `tool_usage`, and
  facade `list_tool_usage_insights(query)`. MCP tool names of the form
  `mcp__<server>__<tool>` have their server segment extracted as a
  first-class `mcp_server` field on each entry.

- Resume brief is now a first-class durable insight (#1129). `ResumeBrief`
  carries a typed `provenance` payload citing the session, message,
  work-event, phase, and work-thread IDs it composed from, with a
  `materializer_version` consumers can use to invalidate cached renderings.
  New MCP tool `get_resume_brief(conversation_id, related_limit)` returns the
  typed brief on the shared single-object envelope; the existing
  `polylogue resume <session-id>` CLI continues to render the brief and now
  surfaces the provenance fields via `--format json`.

- Typed maintenance planner contract (#1144): `BackfillOperation`
  envelopes now carry a typed `scope`, `reason`
  (`InvalidationReason`), `resume_cursor`, bounded `failure_samples`
  with a `truncated` flag, and `metrics` alongside the existing
  fields. `polylogue maintenance run --target` exposes a new
  `message_embeddings` target (no-op until the dormant embedding
  pipeline is wired in, #828) so the planner's invalidation-key
  vocabulary covers messages FTS, action-event read model, session
  insights, and embeddings/vector index.
- Realtime update channel for the daemon web reader (#957): new
  `GET /api/events` Server-Sent Events endpoint streams daemon-event
  notifications (`ingestion_batch`, `ingest`, `reset`, `operation`) so the
  reader updates without a manual refresh. `?poll=1` returns the same
  payload as a JSON snapshot for `EventSource`-less fallbacks. `GET
  /api/status` now advertises a monotonic `last_event_id` field and a
  weak ETag, and returns `304 Not Modified` when the client's
  `If-None-Match` matches.

### Security

- FTS5 query escaping now treats `.`, `/`, and `?` as special characters so
  search inputs containing path-like or single-character-wildcard
  punctuation are quoted rather than surfaced to SQLite as syntax errors.
  Previously, inputs such as `../etc/passwd`, `foo.bar`, or `test?` from
  CLI/MCP search raised `sqlite3.OperationalError` and leaked the underlying
  FTS5 grammar to callers.
- Webhook delivery now connects to the IP validated against the SSRF
  denylist rather than re-resolving the hostname, closing a DNS-rebinding
  TOCTOU window. Hostname is preserved for SNI/cert verification.
- Path sanitization (`safe_path_component`) NFC-normalizes input before
  applying the ASCII allowlist; visually-identical NFC and NFD forms now
  hash to the same path.
- `pip-audit` runs in CI on every PR that touches `pyproject.toml` or
  `uv.lock`, on master pushes, and weekly. Bumped four CVE-affected deps
  (`pygments`, `pytest`, `cryptography`, `python-multipart`) and pinned
  the latter two as direct constraints.

### Added

- Broad-distribution packaging readiness (#953): the
  `release verify-distribution` gate now runs as a `distribution` job in
  `ci.yml` on every push, a new `release.yml` workflow publishes wheel
  and sdist to PyPI via OIDC Trusted Publishing on `vX.Y.Z` tag push,
  and a multi-stage `Containerfile` produces an OCI image that
  `release.yml` builds and pushes to `ghcr.io/sinity/polylogue` with
  semver tags. See [`docs/release.md`](docs/release.md) for the cut-time
  checklist.
- Local source discovery and parsing for Gemini CLI `~/.gemini/tmp` sessions
  and Hermes `~/.hermes/sessions` session documents, with distinct
  `gemini-cli` / `hermes` source identities.
- Antigravity source ingestion through its local language-server Markdown
  export surface, with parseable brain artifacts retained as auxiliary
  documents and raw protobuf state classified as non-directly-parseable
  sidecar storage.
- `polylogue insights work-events` now accepts `--session-date-since` and
  `--session-date-until`, exposing canonical-date bounded work-event reads
  through the public insight facade.
- `polylogue select` as a query-backed selector that prints one matched
  conversation field for shell pipelines, with interactive `fzf`/prompt
  selection when attached to a terminal.
- `polylogued` as the daemon/service executable; live source watching now runs
  as `polylogued watch`.
- `polylogued run` to run live watching and the browser-capture receiver
  together as daemon-owned components.
- `dependabot.yml` for weekly Python and GitHub Actions updates with
  patch grouping.
- `actionlint` workflow validating workflow YAML on PRs that touch
  `.github/`.
- MCP serving now accepts `--role read|write|admin`; mutation tools require
  `write` and maintenance tools require `admin`.
- `devtools verify-schema-roundtrip`, diff-shaped `devtools proof-pack`,
  Markdown `devtools obligation-diff`, and a PR proof-comment workflow expose
  proof coverage, known gaps, and schema package round-trips as reusable gates.
- Codex ingestion now materializes `turn_context.cwd`, token/function events,
  and tool call/result messages so cwd filters and diagnostics work beyond
  Claude Code.
- Expression index `idx_raw_conv_effective_provider` so raw-conversation
  provider-filter queries no longer scan the full table.
- Partial indexes scoped to the gating `WHERE` of each `session_profiles`
  search-text backfill, so repeated bootstraps drain an empty index
  instead of scanning the table.
- Schema v5: tags M2M tables (`tags`, `conversation_tags`), blob GC lease
  tables (`pending_blob_refs`, `gc_generations`), and `insights.db` extraction
  with its own schema lifecycle. Wipe-and-rebuild required.
- `ArchiveWriteGateway` as the single canonical write path (daemon RPC stub for
  slice G). Post-ingest side effects (FTS repair, cache invalidation) now route
  through `polylogue.archive.write_effects`.
- `run_blob_gc()` for safe garbage collection of unreferenced blobs with lease,
  generation, and MIN_AGE guards.
- `devtools run-benchmark-campaigns` now includes a
  `daemon-live-convergence` synthetic campaign that reports live-ingest file
  counts, read/write byte shape, append-tail byte shape, stage timings, and
  archive row counts, plus process and cgroup memory peaks.
- Durable reader workspaces are now persisted across the archive API, daemon
  `/api/user/workspaces`, `polylogue user-state workspaces`, and MCP mutation
  tools, with resolved/degraded target evidence for tabs, stack, compare, and
  timeline modes.
- The local reader daemon now serves `/w/stack` and `/w/compare` shell routes
  plus `/api/stack` and `/api/compare` envelopes for multi-conversation reader
  workflows with explicit missing-target evidence.

### Changed

- Live daemon convergence now drains watcher batches single-flight, records
  stale cursor-write counters, uses bounded tail hashes for same-size rewrite
  detection, and exposes recent source-path churn in the daemon workload probe.
- Session work-event and phase evidence payloads now expose timing/date
  provenance, and work-event insight reads order by event time instead of
  conversation recency.
- The root query `--tail` mode and tail-overlay JSON provenance were removed;
  daemon-owned live ingestion is now the supported path for fresh session state.
- `--resource-mode` and Polylogue CLI self-demotion were removed from
  foreground maintenance commands; workstation resource policy belongs in the
  host environment and daemon supervision, not in product-level CLI flags.
- Query/file-reference filters now use `referenced_path` / `--referenced-path`
  consistently, and MCP conversation reads return headers while message bodies
  live behind paginated `get_messages` / `messages` reads.
- Browser capture now calls its local artifact directory a `spool`; the stale
  inbox helper/export was removed from public paths.
- `schema list`, `schema explain`, and `schema compare` use canonical
  `--format json` output while retaining `--json` as a strict alias.
- `polylogue audit` was removed from the product CLI; verification-lab audit
  workflows live under `devtools`.
- Daemon status now caps live cursor file samples while preserving exact
  counts, and full-ingest attempts heartbeat during long storage-write phases.
- Schema v9 adds indexes for message foreign keys on `provider_events` and
  `attachment_refs`, avoiding full child-table scans during conversation
  replacement.
- `polylogued status` now reports recent live-ingest attempts with durable
  phase, file-count, byte-read, timing, RSS, cgroup memory, and stale-heartbeat
  snapshots so interrupted convergence work leaves diagnosable state in the
  archive DB.
- Codex JSONL ingestion now parses hot streams directly from raw records,
  skips validation-off pre-sampling for known stream providers, and reuses
  message hash payloads during materialization, reducing daemon live-ingest
  parse overhead for large Codex sessions.
- Daemon live convergence refreshes affected insight rows in batches and avoids
  process-pool startup for tiny ingest batches, reducing convergence and parse
  overhead for live JSONL workloads.
- Daemon live catch-up no longer pre-hashes uncursored files, reads cursor
  state in bulk, and reports the max ingest worker count in live benchmark
  metrics so many-file convergence throughput is observable.
- Daemon live ingest bounds small-file convergence groups, offloads sync
  parse/write work from the event loop, and drains process-pool results without
  retaining the whole parsed batch in memory.
- Daemon live convergence avoids duplicate message-FTS repair, chunks sync
  insight rebuilds by message budget, samples each JSONL once before raw
  storage, and streams raw blobs in 1 MiB chunks for large catch-up workloads.
- Daemon convergence now scopes embedding work to changed conversations and
  avoids starting an async embedding runner from the synchronous convergence
  stage.
- Daemon live ingest now excludes relationship-index JSONL sidecars before raw
  storage instead of treating scalar `conversation`/`parent`/`child` records as
  provider conversation streams.
- Live watching is no longer exposed through root `polylogue watch` or
  `polylogue run --watch`; use `polylogued watch` for the long-running source
  watcher.
- Root `polylogue run` and its stage subcommands were removed; ingestion is
  daemon-owned through `polylogued run` and explicit `polylogue ingest PATH`
  requests.
- Legacy batch-run state, JSON run artifacts, run observers, and the `runs`
  schema table were removed; schema v8 archives are fresh-only.
- Browser-capture receiver serving/status moved from root `polylogue
  browser-capture` to `polylogued browser-capture`.
- `polylogued status` now reports configured daemon components, including live
  watch roots and the browser-capture receiver target.
- `polylogue doctor --daemon` now includes the same daemon component status in
  the interactive health surface.
- Live watcher cursor and failure state now live in the archive database and
  failed ingests remain retryable after backoff instead of being recorded as
  successful cursor progress.
- `polylogued status` and daemon ingestion events now expose live cursor
  backlog, retry state, batch counters, byte deltas, and convergence timings.
- Live daemon ingestion now uses cursor offsets for append-only JSONL growth so
  completed tails can be read and merged without re-reading unchanged source
  prefixes.
- Daemon convergence now uses the live watcher's batched ingest path as the
  only source-ingest path; post-ingest convergence stages only repair FTS,
  embeddings, and insights.
- `Config` rejects relative `archive_root`, `render_root`, or `db_path`
  with `ConfigError` at construction.
- `_privacy_level_value` raises `ValueError` on unknown level strings
  instead of silently returning `"standard"`.
- `get_stats_by` raises `ValueError` on unknown `group_by` instead of
  silently falling back to provider grouping.
- Top-level boundary-catchable errors: `ArchiveOperations` config/repository
  init, parsing-service backend uninitialized, and parser state-machine
  phase violations now raise from the `PolylogueError` hierarchy
  (previously `RuntimeError`).
- Search-path degradation visibility: `search_action_results`,
  `search_hybrid_results`, and `helper_summary` log at `WARNING` (not
  `DEBUG`) when falling back from a failed primary path.

### Fixed

- Schema v2 archives now upgrade in place to the additive schema v3
  `messages.message_type` column/index, preserving existing archive rows
  instead of failing the next run after an older binary rewrites
  `PRAGMA user_version`.
- Schema v2 archives that already have `messages.message_type` skip the
  message-type backfill scan and only repair the missing version/index state.
- Schema v3 archives now upgrade to v4 by rebuilding action-event FTS rows
  with base-table rowids, enabling targeted incremental FTS repairs instead
  of archive-wide FTS scans.
- `sanitize_path` symlink probe narrowed to `OSError` and treats
  uncertainty as suspicious (previously a `PermissionError` on an
  unreadable directory could mask a traversal attempt).
- `_clamp_limit` already enforced an upper bound (1000); confirmed
  resolved.
- `resolve_id` already supports `strict=True` and every MCP destructive
  call site already uses it; confirmed resolved.

## [0.1.0] — Unreleased

Initial development snapshot. Versioned releases begin once the install
path lands (#416).
