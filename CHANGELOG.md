# Changelog

All notable user-visible changes to Polylogue are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

User-visible: anything an operator running `polylogue` would notice — new
flags, removed or renamed commands, output changes, breaking schema
migrations, security fixes. Internal refactors, test additions, and
documentation polish do not require an entry.

## [0.2.0](https://github.com/Sinity/polylogue/compare/v0.1.0...v0.2.0) (2026-06-13)


### Added

* add benchmark infra, fix FTS5 prefix, fix hooks health, add schema v21, lazy structlog ([#1742](https://github.com/Sinity/polylogue/issues/1742)) ([bcba9d1](https://github.com/Sinity/polylogue/commit/bcba9d118b03dd283f240f30556d987cd6b83066))
* add blob integrity probe ([#1480](https://github.com/Sinity/polylogue/issues/1480)) ([07c6def](https://github.com/Sinity/polylogue/commit/07c6def9ce0a1c3efbe88485fee790e9144d4680)), closes [#1231](https://github.com/Sinity/polylogue/issues/1231)
* add bounded embedding preflight ([#1531](https://github.com/Sinity/polylogue/issues/1531)) ([6c2b86d](https://github.com/Sinity/polylogue/commit/6c2b86d2c8c43664b9bdb1e42ce5447651f67ace))
* add reader target references ([#1045](https://github.com/Sinity/polylogue/issues/1045)) ([a4f7cda](https://github.com/Sinity/polylogue/commit/a4f7cda1de9daba2364baa8646c36e5634db8176))
* add shared query-expression compiler (CLI slice) ([#1861](https://github.com/Sinity/polylogue/issues/1861)) ([c854dfd](https://github.com/Sinity/polylogue/commit/c854dfd975ad65e9b0bce1d583427bf550984034))
* add target-aware reader user state ([#1053](https://github.com/Sinity/polylogue/issues/1053)) ([3633308](https://github.com/Sinity/polylogue/commit/3633308c7731b8d9a64ac93735153efef8b0df1b))
* add typed recall-pack target evidence ([#1054](https://github.com/Sinity/polylogue/issues/1054)) ([9555507](https://github.com/Sinity/polylogue/commit/95555079d2626afa11611243fa91e535230b2e38))
* **api:** expose embedding readiness helpers ([#1575](https://github.com/Sinity/polylogue/issues/1575)) ([723b808](https://github.com/Sinity/polylogue/commit/723b808fa7e337fba5c64be95164d9eb6917bc60)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **assembly:** log unmatched/ambiguous history.jsonl rows ([#1700](https://github.com/Sinity/polylogue/issues/1700)) ([8d9195a](https://github.com/Sinity/polylogue/commit/8d9195a3c426f31eb184843c7939c2490a5acd52))
* **benchmarks+devtools:** campaigns, SLO, evidence-report ([#998](https://github.com/Sinity/polylogue/issues/998), [#1063](https://github.com/Sinity/polylogue/issues/1063), [#997](https://github.com/Sinity/polylogue/issues/997)) ([#1072](https://github.com/Sinity/polylogue/issues/1072)) ([3cbf610](https://github.com/Sinity/polylogue/commit/3cbf61054913fe7de319cd16f2cbf84fb93b23e9))
* **blackboard:** persistent agent-addressable notes ([#1697](https://github.com/Sinity/polylogue/issues/1697)) ([cd52d93](https://github.com/Sinity/polylogue/commit/cd52d93cd3524057b76987ce4dfaf572c2af2ff5))
* **cli:** add --json flag ([#1689](https://github.com/Sinity/polylogue/issues/1689)) ([7cd0b79](https://github.com/Sinity/polylogue/commit/7cd0b7929433a20c2c1bfd85256c099f6df4ddb0))
* **cli:** add context compose command ([#1494](https://github.com/Sinity/polylogue/issues/1494)) ([7755656](https://github.com/Sinity/polylogue/commit/77556562c13e3da68007b2f03fc61dec02ed4c5b))
* **cli:** add polylogue commands listing ([#1681](https://github.com/Sinity/polylogue/issues/1681)) ([f5146ee](https://github.com/Sinity/polylogue/commit/f5146ee162f3b4ee1131bf4475b868c3e78ccf1e))
* **cli:** add polylogue paths command and defensive archive.db symlink ([#1627](https://github.com/Sinity/polylogue/issues/1627)) ([bb419eb](https://github.com/Sinity/polylogue/commit/bb419eb360ee38333ccf7f18aa146e58165eed94))
* **cli:** add polylogue recent command ([#1701](https://github.com/Sinity/polylogue/issues/1701)) ([d12fb95](https://github.com/Sinity/polylogue/commit/d12fb95529be2f9daf0024fbf831e239814e2556))
* **cli:** add systematic --json output test and snapshot gate ([#1689](https://github.com/Sinity/polylogue/issues/1689)) ([06de763](https://github.com/Sinity/polylogue/commit/06de76340e2676e0415f88e05072d8785d2c6f75))
* **cli:** correlate sessions with git commits ([#1690](https://github.com/Sinity/polylogue/issues/1690)) ([292628e](https://github.com/Sinity/polylogue/commit/292628ebbad010266391e1ff3489fb8a8d86106b))
* **cli:** dynamic completion coverage matrix bash/zsh/fish ([#1271](https://github.com/Sinity/polylogue/issues/1271)) ([#1324](https://github.com/Sinity/polylogue/issues/1324)) ([31ae6cc](https://github.com/Sinity/polylogue/commit/31ae6cc5e76c482b38a324d18fd0f3f1b135ab5c))
* **cli:** error/help discipline + --diagnose + actionable usage errors ([#1273](https://github.com/Sinity/polylogue/issues/1273)) ([#1350](https://github.com/Sinity/polylogue/issues/1350)) ([4bd105f](https://github.com/Sinity/polylogue/commit/4bd105fe53d82330401f1a3a29ba42bae7cff5b6))
* **cli:** first-run status actionable text + tutorial ([#1263](https://github.com/Sinity/polylogue/issues/1263)) ([1d1a2cc](https://github.com/Sinity/polylogue/commit/1d1a2ccdbd925695b36e90ef54398de035341e51))
* **cli:** first-run status actionable text + tutorial ([#1263](https://github.com/Sinity/polylogue/issues/1263)) ([#1376](https://github.com/Sinity/polylogue/issues/1376)) ([1d1a2cc](https://github.com/Sinity/polylogue/commit/1d1a2ccdbd925695b36e90ef54398de035341e51))
* **cli:** harmonize list output fields with MCP shape ([#1699](https://github.com/Sinity/polylogue/issues/1699)) ([d0173a4](https://github.com/Sinity/polylogue/commit/d0173a4971f8010b6358551db5051d63cdefec77))
* **cli:** output assurance contract + JSON schemas + NDJSON streaming ([#1272](https://github.com/Sinity/polylogue/issues/1272)) ([#1340](https://github.com/Sinity/polylogue/issues/1340)) ([7d8dae1](https://github.com/Sinity/polylogue/commit/7d8dae124d246a8a3832565170fa974b708396b4))
* **cli:** polish color, completions, narrow-terminal output ([#958](https://github.com/Sinity/polylogue/issues/958)) ([#1115](https://github.com/Sinity/polylogue/issues/1115)) ([109ccfd](https://github.com/Sinity/polylogue/commit/109ccfd78d629aa7e07d01fc530c49690a5ca586))
* **cli:** polylogue ingest truthful or rejects ([#1264](https://github.com/Sinity/polylogue/issues/1264)) ([#1383](https://github.com/Sinity/polylogue/issues/1383)) ([8382b18](https://github.com/Sinity/polylogue/commit/8382b180419d8d06ec5e0a75edccc287891092df)), closes [#869](https://github.com/Sinity/polylogue/issues/869)
* **cli:** theme tokens + flag-alias consistency + --help-markdown ([#1274](https://github.com/Sinity/polylogue/issues/1274)) ([#1364](https://github.com/Sinity/polylogue/issues/1364)) ([117b53d](https://github.com/Sinity/polylogue/commit/117b53d1c32a9eee92d39f6062a5f3e3c9db0651))
* **config:** unified polylogue.toml with explicit layer precedence ([#829](https://github.com/Sinity/polylogue/issues/829)) ([#1143](https://github.com/Sinity/polylogue/issues/1143)) ([3617391](https://github.com/Sinity/polylogue/commit/36173910d3910a1ff63c3f1cb427d4b22daeac8e))
* **cost:** cycle outlook engine ([#1137](https://github.com/Sinity/polylogue/issues/1137)) ([#1160](https://github.com/Sinity/polylogue/issues/1160)) ([e843c16](https://github.com/Sinity/polylogue/commit/e843c16cc9e32ba8462759be5d01ce26c75bca51))
* **cost:** expose cycle outlook through CLI and MCP ([#1138](https://github.com/Sinity/polylogue/issues/1138)) ([#1168](https://github.com/Sinity/polylogue/issues/1168)) ([f2330c1](https://github.com/Sinity/polylogue/commit/f2330c177beec5c4380fa75ccce5e20db8a1bbcf))
* **cost:** per-basis cost rollup with per-model breakdown ([#1136](https://github.com/Sinity/polylogue/issues/1136)) ([#1166](https://github.com/Sinity/polylogue/issues/1166)) ([7261aeb](https://github.com/Sinity/polylogue/commit/7261aeb15844e0052547d08f3d06f0eee2f3b9d5))
* **cost:** typed subscription plan config ([#1132](https://github.com/Sinity/polylogue/issues/1132)) ([#1153](https://github.com/Sinity/polylogue/issues/1153)) ([b6e3a27](https://github.com/Sinity/polylogue/commit/b6e3a27dd0c91d03a4cfd2738e5ff03c457fcfc3))
* **daemon:** /metrics Prometheus endpoint ([#1321](https://github.com/Sinity/polylogue/issues/1321)) ([#1407](https://github.com/Sinity/polylogue/issues/1407)) ([cae4602](https://github.com/Sinity/polylogue/commit/cae4602f2d6df55fee9280b06ab094bb48dc8d84))
* **daemon:** add OTLP HTTP receiver and rich Prometheus metrics ([#1321](https://github.com/Sinity/polylogue/issues/1321)) ([10e2a78](https://github.com/Sinity/polylogue/commit/10e2a78657d9fed7f8111bdb9ded42ef073bde12))
* **daemon:** auto-calibrating cursor-lag anomaly band ([#1349](https://github.com/Sinity/polylogue/issues/1349)) ([#1372](https://github.com/Sinity/polylogue/issues/1372)) ([be5acac](https://github.com/Sinity/polylogue/commit/be5acac97cd177fbf2ffbab1480ec4c8b2195b11))
* **daemon:** convergence-debt alert with per-source-family thresholds ([#1226](https://github.com/Sinity/polylogue/issues/1226)) ([#1335](https://github.com/Sinity/polylogue/issues/1335)) ([aaed9be](https://github.com/Sinity/polylogue/commit/aaed9bec2f69703604bdf4e8182ffa5ad7d8df63))
* **daemon:** distinguish slow-but-progressing from stuck attempts ([#1246](https://github.com/Sinity/polylogue/issues/1246)) ([#1399](https://github.com/Sinity/polylogue/issues/1399)) ([2c0901e](https://github.com/Sinity/polylogue/commit/2c0901e2a85ca4acb649d5d5e4d6dd8e91bc630c))
* **daemon:** FTS-trigger drift detection + auto-repair ([#1229](https://github.com/Sinity/polylogue/issues/1229)) ([#1342](https://github.com/Sinity/polylogue/issues/1342)) ([a9b21aa](https://github.com/Sinity/polylogue/commit/a9b21aa15ead9256d5b13343c724e0dca40c10d4))
* **daemon:** health endpoint contracts + k8s ready/live probes ([#1224](https://github.com/Sinity/polylogue/issues/1224)) ([#1320](https://github.com/Sinity/polylogue/issues/1320)) ([fe03370](https://github.com/Sinity/polylogue/commit/fe0337084cf3abd64d411fa0cb701c8efa22bc63))
* **daemon:** notification backends — webhook/journald/email/Apprise ([#1233](https://github.com/Sinity/polylogue/issues/1233)) ([#1348](https://github.com/Sinity/polylogue/issues/1348)) ([2d82700](https://github.com/Sinity/polylogue/commit/2d82700943d381250e26034099043adf12bf20b6))
* **daemon:** per-source cursor-lag SLO + escalation ([#1232](https://github.com/Sinity/polylogue/issues/1232)) ([#1346](https://github.com/Sinity/polylogue/issues/1346)) ([62ac10e](https://github.com/Sinity/polylogue/commit/62ac10e625510a04d3b171b156a266d4d379b36a))
* **daemon:** webhook notifications for health alerts ([#1150](https://github.com/Sinity/polylogue/issues/1150)) ([#1159](https://github.com/Sinity/polylogue/issues/1159)) ([96c6b99](https://github.com/Sinity/polylogue/commit/96c6b998bc0e5fdffa193c1ca230ed3852752df9))
* **devtools:** add `devtools test` focused runner so agents avoid raw pytest ([#1786](https://github.com/Sinity/polylogue/issues/1786)) ([15af28c](https://github.com/Sinity/polylogue/commit/15af28cde2d753bbd8566c9ea3d0551d1a5ed48d))
* **devtools:** add archive space report ([#1572](https://github.com/Sinity/polylogue/issues/1572)) ([46aa9f5](https://github.com/Sinity/polylogue/commit/46aa9f5c5d29764eb751043ca6ae005ef9fe7e7e)), closes [#1486](https://github.com/Sinity/polylogue/issues/1486) [#1552](https://github.com/Sinity/polylogue/issues/1552)
* **devtools:** add safe worktree gc ([#1222](https://github.com/Sinity/polylogue/issues/1222)) ([68ae333](https://github.com/Sinity/polylogue/commit/68ae3330b7313455c455bece8b78ce124d17fb6a))
* **devtools:** comparable daemon-workload-probe reports for convergence proofs ([#845](https://github.com/Sinity/polylogue/issues/845)) ([#1100](https://github.com/Sinity/polylogue/issues/1100)) ([dc87105](https://github.com/Sinity/polylogue/commit/dc87105927ece866938e2a63b933d5ca2b90aebe))
* **devtools:** evidence dashboard and changed-path traceability ([#998](https://github.com/Sinity/polylogue/issues/998)) ([#1091](https://github.com/Sinity/polylogue/issues/1091)) ([2a61e20](https://github.com/Sinity/polylogue/commit/2a61e205f14dd069afcf5852c16e4b4ed2ce0f35))
* **devtools:** failure-context command for agent inner-loop debugging ([#1285](https://github.com/Sinity/polylogue/issues/1285)) ([812c56e](https://github.com/Sinity/polylogue/commit/812c56e1f769bfd426baa38a919d02ca24d15ad1))
* **devtools:** render use_when/examples into Click --help ([#1286](https://github.com/Sinity/polylogue/issues/1286)) ([2d09d63](https://github.com/Sinity/polylogue/commit/2d09d63e413788dc6a1fb15702c5d79579744679)), closes [#1283](https://github.com/Sinity/polylogue/issues/1283)
* **devtools:** verification-impact --paths for speculative routing ([#1312](https://github.com/Sinity/polylogue/issues/1312)) ([#1330](https://github.com/Sinity/polylogue/issues/1330)) ([9d04126](https://github.com/Sinity/polylogue/commit/9d041264769cd3f6adb5f78ff84d332fd4f3b776))
* **devtools:** wire nightly scale workflow into benchmark-campaign index ([#1220](https://github.com/Sinity/polylogue/issues/1220)) ([ee60a7b](https://github.com/Sinity/polylogue/commit/ee60a7b143a3dad4aee2d68e9129726daae83eae))
* **devtools:** xtask auto-log + replay + budget + prune ([#1289](https://github.com/Sinity/polylogue/issues/1289)) ([#1315](https://github.com/Sinity/polylogue/issues/1315)) ([602a922](https://github.com/Sinity/polylogue/commit/602a922ae108cf45665942c06482d2e3e6817e54)), closes [#1283](https://github.com/Sinity/polylogue/issues/1283)
* **dist:** browser extension release artifacts + install docs ([#1238](https://github.com/Sinity/polylogue/issues/1238)) ([#1345](https://github.com/Sinity/polylogue/issues/1345)) ([b5dccee](https://github.com/Sinity/polylogue/commit/b5dccee00abaa0f79cfccf5ac11f5ff2d3860bea))
* **dist:** Homebrew formula + auto-bump action ([#1237](https://github.com/Sinity/polylogue/issues/1237)) ([#1341](https://github.com/Sinity/polylogue/issues/1341)) ([babc820](https://github.com/Sinity/polylogue/commit/babc820a7f077ab1d50e0384569950f266ffdc28))
* **dist:** Nix flake module + service + cachix + FlakeHub ([#1235](https://github.com/Sinity/polylogue/issues/1235)) ([#1336](https://github.com/Sinity/polylogue/issues/1336)) ([0474138](https://github.com/Sinity/polylogue/commit/04741388be75f2ab5cec345fa6c954db33400ffa))
* **dist:** OCI container with multi-arch + healthcheck + compose ([#1236](https://github.com/Sinity/polylogue/issues/1236)) ([#1319](https://github.com/Sinity/polylogue/issues/1319)) ([7cf4a78](https://github.com/Sinity/polylogue/commit/7cf4a78d2d04d354f80dc67b016dfa33c84e7810))
* **dist:** publish polylogue-mcp and polylogue-hooks as separate PyPI packages ([#1309](https://github.com/Sinity/polylogue/issues/1309)) ([#1423](https://github.com/Sinity/polylogue/issues/1423)) ([873146f](https://github.com/Sinity/polylogue/commit/873146f47dd0f8114d52da1bf7d18fd7ff43c174))
* **dist:** PyPI trusted-publisher workflow with installed smoke ([#1234](https://github.com/Sinity/polylogue/issues/1234)) ([#1311](https://github.com/Sinity/polylogue/issues/1311)) ([de83b59](https://github.com/Sinity/polylogue/commit/de83b590cbbccce428d1d636a6841f645f7f7edd))
* **distribution:** PyPI/container packaging readiness ([#953](https://github.com/Sinity/polylogue/issues/953)) ([#1141](https://github.com/Sinity/polylogue/issues/1141)) ([e7bf42f](https://github.com/Sinity/polylogue/commit/e7bf42f9c8d2e84924780e6f8676022e0c2cdd01))
* **embeddings:** activation flow + hybrid default + local backend support ([#1217](https://github.com/Sinity/polylogue/issues/1217)) ([#1327](https://github.com/Sinity/polylogue/issues/1327)) ([9f6978b](https://github.com/Sinity/polylogue/commit/9f6978b831801a0a00d21c36882186bb2b2a6126))
* **embeddings:** bound catch-up windows by cost ([#1543](https://github.com/Sinity/polylogue/issues/1543)) ([3775bee](https://github.com/Sinity/polylogue/commit/3775bee4afb9cd2014e974a5056abbc73adafc1f))
* **embeddings:** emit preflight json plan ([#1561](https://github.com/Sinity/polylogue/issues/1561)) ([1c39aaf](https://github.com/Sinity/polylogue/commit/1c39aaf21eaa36818470feeb90929ab7118f0f92))
* **embeddings:** expose status next actions ([#1560](https://github.com/Sinity/polylogue/issues/1560)) ([a6af6bb](https://github.com/Sinity/polylogue/commit/a6af6bbf090890fb64bb0492d686ea73391851de)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* expose embedding catch-up metrics ([#1524](https://github.com/Sinity/polylogue/issues/1524)) ([f42f053](https://github.com/Sinity/polylogue/commit/f42f053d1aaf9cc479fbcb59631cc9903f615f82))
* expose reader user state through MCP ([#1052](https://github.com/Sinity/polylogue/issues/1052)) ([2face55](https://github.com/Sinity/polylogue/commit/2face55b7750f1cde14ebcbdbbb6fb7ca91c11dd))
* expose reader user-state APIs ([#1050](https://github.com/Sinity/polylogue/issues/1050)) ([b8c513f](https://github.com/Sinity/polylogue/commit/b8c513ffb3476d8f8d37f447fd26ee1b5bf44966))
* **facets:** wire SQL-backed aggregators for repos, message_types, action_types, has_flags ([#1694](https://github.com/Sinity/polylogue/issues/1694)) ([b124e85](https://github.com/Sinity/polylogue/commit/b124e85f8bab2905064d45138449e232e7d668fb))
* insights section headers, mutation CI, concurrent search test ([862422a](https://github.com/Sinity/polylogue/commit/862422a4637281bcef63e382a6366032d44d8d56))
* **insights:** add OTLP span-to-work-event correlation ([#1686](https://github.com/Sinity/polylogue/issues/1686)) ([a7281b1](https://github.com/Sinity/polylogue/commit/a7281b13ad47f2427c4e95bf74dc277ca456778c))
* **insights:** add session latency profiles ([#1539](https://github.com/Sinity/polylogue/issues/1539)) ([4c504ed](https://github.com/Sinity/polylogue/commit/4c504eddb8c25753e096689d1d9248a91e40dbf1))
* **insights:** add session-commit correlation and issue/PR enrichment ([#1690](https://github.com/Sinity/polylogue/issues/1690)) ([50aec22](https://github.com/Sinity/polylogue/commit/50aec222d6180dd8374abf54000b577a4fe08108))
* **insights:** add tool active duration ([#1537](https://github.com/Sinity/polylogue/issues/1537)) ([bf40a87](https://github.com/Sinity/polylogue/commit/bf40a87b1acddf8b348f862c9fb35ffea02a1387)), closes [#1526](https://github.com/Sinity/polylogue/issues/1526)
* **insights:** classify session shape and terminal state ([#1538](https://github.com/Sinity/polylogue/issues/1538)) ([cb08bac](https://github.com/Sinity/polylogue/commit/cb08bacc3e408c72211f797b9f1732757b915b10))
* **insights:** detect /goal-driven sessions ([#1687](https://github.com/Sinity/polylogue/issues/1687)) ([8b5894b](https://github.com/Sinity/polylogue/commit/8b5894b99800cae62dbd73dcb2141baeba91cd80))
* **insights:** learning feedback loop ([#1131](https://github.com/Sinity/polylogue/issues/1131)) ([#1170](https://github.com/Sinity/polylogue/issues/1170)) ([1dd65a7](https://github.com/Sinity/polylogue/commit/1dd65a75ac5176e7c8ad5fe47a198c2da8f8f37c))
* **insights:** materialize logical session identity ([#1540](https://github.com/Sinity/polylogue/issues/1540)) ([02decd7](https://github.com/Sinity/polylogue/commit/02decd741306f6490f5ce114639659d2ff3f02f8)), closes [#866](https://github.com/Sinity/polylogue/issues/866)
* **insights:** productivity rollups with explicit caveats ([#1134](https://github.com/Sinity/polylogue/issues/1134)) ([#1167](https://github.com/Sinity/polylogue/issues/1167)) ([c5c5a1f](https://github.com/Sinity/polylogue/commit/c5c5a1fb1f3758ac452a9f6e49caf70b90d6be32))
* **insights:** rank resume candidates ([#1541](https://github.com/Sinity/polylogue/issues/1541)) ([e9a247e](https://github.com/Sinity/polylogue/commit/e9a247ea981d5cf576bebe1219d97cb4c31c3222))
* **insights:** resume brief insight with CLI/MCP/facade ([#1129](https://github.com/Sinity/polylogue/issues/1129)) ([#1156](https://github.com/Sinity/polylogue/issues/1156)) ([f4f7ef0](https://github.com/Sinity/polylogue/commit/f4f7ef03a126bae3249b751faea0de216619f6a4))
* **insights:** session classification with confidence ([#1130](https://github.com/Sinity/polylogue/issues/1130)) ([#1163](https://github.com/Sinity/polylogue/issues/1163)) ([6668996](https://github.com/Sinity/polylogue/commit/66689968ed7f395078538fe1c938be8c16a2f304))
* **insights:** session timeline renderer with fidelity tags ([#1135](https://github.com/Sinity/polylogue/issues/1135)) ([#1165](https://github.com/Sinity/polylogue/issues/1165)) ([c7fc08a](https://github.com/Sinity/polylogue/commit/c7fc08ab03e1339de81fa3d8e6fd768a85aaf52e))
* **insights:** tool usage analytics per provider ([#1133](https://github.com/Sinity/polylogue/issues/1133)) ([#1164](https://github.com/Sinity/polylogue/issues/1164)) ([7d6dd65](https://github.com/Sinity/polylogue/commit/7d6dd65a7a929d8465ff1351e385259918ca7304))
* **live:** enrich paste evidence from UserPromptSubmit hook events ([#1704](https://github.com/Sinity/polylogue/issues/1704)) ([0e7be4d](https://github.com/Sinity/polylogue/commit/0e7be4d04f208a523f397eac2140cf7acf95eced))
* **maintenance:** idempotent resumable replay ([#1147](https://github.com/Sinity/polylogue/issues/1147)) ([#1161](https://github.com/Sinity/polylogue/issues/1161)) ([2b61ab7](https://github.com/Sinity/polylogue/commit/2b61ab7a50aa58248170c8be0ef6e61591aedbcc))
* **maintenance:** persistent op registry + status surface ([#1197](https://github.com/Sinity/polylogue/issues/1197)) ([#1366](https://github.com/Sinity/polylogue/issues/1366)) ([8662416](https://github.com/Sinity/polylogue/commit/8662416b10fcbf626e809bfaaec2746ee4524f0a))
* **maintenance:** route replay failures to daemon raw-failure surface ([#1198](https://github.com/Sinity/polylogue/issues/1198)) ([#1371](https://github.com/Sinity/polylogue/issues/1371)) ([7d9ffab](https://github.com/Sinity/polylogue/commit/7d9ffab2ef3b27541f4ed3a16484ecfc41bd863e))
* **maintenance:** shared operation envelope across surfaces ([#1149](https://github.com/Sinity/polylogue/issues/1149)) ([#1169](https://github.com/Sinity/polylogue/issues/1169)) ([bde1c59](https://github.com/Sinity/polylogue/commit/bde1c59d098015d13f44ab5858bafb9215e07e09))
* **maintenance:** typed planner contract ([#1144](https://github.com/Sinity/polylogue/issues/1144)) ([#1157](https://github.com/Sinity/polylogue/issues/1157)) ([7a04357](https://github.com/Sinity/polylogue/commit/7a04357bef1dc08fb062ed03d822e3313748d41b))
* **maintenance:** typed scope filters ([#1196](https://github.com/Sinity/polylogue/issues/1196)) ([#1323](https://github.com/Sinity/polylogue/issues/1323)) ([ceecf5b](https://github.com/Sinity/polylogue/commit/ceecf5b6b549ab008931143b088b292055952cbb))
* **mcp:** add aggregate_sessions tool ([#1691](https://github.com/Sinity/polylogue/issues/1691)) ([4a294e8](https://github.com/Sinity/polylogue/commit/4a294e8aad42281d555e72d4de41ed09fcc1cccc))
* **mcp:** add compare_sessions, find_similar_sessions, correlate_sessions tools ([#1691](https://github.com/Sinity/polylogue/issues/1691)) ([9120990](https://github.com/Sinity/polylogue/commit/9120990eb7bef7676e744cbf2c30cd852ab8b108))
* **mcp:** compose_context_preamble for SessionStart ([#1494](https://github.com/Sinity/polylogue/issues/1494)) ([2ccfc26](https://github.com/Sinity/polylogue/commit/2ccfc2626999a07176a0ff23385f0d4f4ce55fea))
* **mcp:** expose embedding preflight ([#1573](https://github.com/Sinity/polylogue/issues/1573)) ([68752a8](https://github.com/Sinity/polylogue/commit/68752a81a477e82486f093fedb46c97f4461c950)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **mcp:** expose embedding readiness status ([#1571](https://github.com/Sinity/polylogue/issues/1571)) ([dc302e1](https://github.com/Sinity/polylogue/commit/dc302e116c8c2152b67f90d8815622dfe6c9f034)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **mcp:** expose the agent blackboard via MCP tools ([#1801](https://github.com/Sinity/polylogue/issues/1801)) ([4faf166](https://github.com/Sinity/polylogue/commit/4faf1663f8a17c9cdf9b3655e119c6d7ff4e3124)), closes [#1697](https://github.com/Sinity/polylogue/issues/1697)
* **mcp:** forward EmbeddingRetrievalNotReadyError message verbatim ([#1503](https://github.com/Sinity/polylogue/issues/1503) AC4) ([#1663](https://github.com/Sinity/polylogue/issues/1663)) ([5e23d5c](https://github.com/Sinity/polylogue/commit/5e23d5cd09e0d53cd897e9c7501dca8a89c95dbb))
* **mcp:** support query-scoped facets ([#1569](https://github.com/Sinity/polylogue/issues/1569)) ([d575ede](https://github.com/Sinity/polylogue/commit/d575ededaaf996c4665fe9c19aa11d9fa9270fcb)), closes [#873](https://github.com/Sinity/polylogue/issues/873)
* **nix:** unified settings lib + discoverSources, xdg config, CLI flags ([#1685](https://github.com/Sinity/polylogue/issues/1685)) ([22e337a](https://github.com/Sinity/polylogue/commit/22e337a911b5adba6d73a80de4026ea00407e5ba))
* **onboarding:** autodetect chat sources and first-run setup ([#869](https://github.com/Sinity/polylogue/issues/869)) ([#1116](https://github.com/Sinity/polylogue/issues/1116)) ([3103d05](https://github.com/Sinity/polylogue/commit/3103d056f30bc670f29777cecd9f7aff9f1cae28))
* **parsers:** wire Claude Code history.jsonl paste evidence into has_paste ([#1651](https://github.com/Sinity/polylogue/issues/1651)) ([2872c65](https://github.com/Sinity/polylogue/commit/2872c6532b6b3fa418acb0e8518bdce570844c84))
* persist embedding catch-up progress ([#1523](https://github.com/Sinity/polylogue/issues/1523)) ([b6b4737](https://github.com/Sinity/polylogue/commit/b6b47374677753d9bdeef1ee31de82c5ce6c1931))
* **pipeline:** wire hook events to has_paste detection ([#1313](https://github.com/Sinity/polylogue/issues/1313)) ([#1331](https://github.com/Sinity/polylogue/issues/1331)) ([c06d8fa](https://github.com/Sinity/polylogue/commit/c06d8fafb3fb53be72f71430aec15e3d7e8da563))
* **reader:** add message anchors, folds, density toggle, keyboard nav, topology inspector ([#1518](https://github.com/Sinity/polylogue/issues/1518)) ([3b4a777](https://github.com/Sinity/polylogue/commit/3b4a777d928cad990bddb4be866603012e40404a))
* **reader:** add workspace shell modes ([#1057](https://github.com/Sinity/polylogue/issues/1057)) ([d3628b5](https://github.com/Sinity/polylogue/commit/d3628b5c402aae82cc8aa7d2da9a6864f2e00c9f))
* **reader:** attachment product surface (cards, inspector, library) ([#1199](https://github.com/Sinity/polylogue/issues/1199)) ([#1337](https://github.com/Sinity/polylogue/issues/1337)) ([70ccde0](https://github.com/Sinity/polylogue/commit/70ccde0fbd6d5cec32548ad808d2a0eb0fce6bdb))
* **reader:** bulk operations on conversation selection ([#1119](https://github.com/Sinity/polylogue/issues/1119)) ([#1162](https://github.com/Sinity/polylogue/issues/1162)) ([caa79b5](https://github.com/Sinity/polylogue/commit/caa79b5cb85804742c66b7420c5b1dcbbdcf72ee))
* **reader:** conversation compare side-by-side ([#1124](https://github.com/Sinity/polylogue/issues/1124)) ([#1175](https://github.com/Sinity/polylogue/issues/1175)) ([bd026f1](https://github.com/Sinity/polylogue/commit/bd026f154c5964629c67a4ec09c7bd836ac2fe26))
* **reader:** embedding similarity browse ([#1123](https://github.com/Sinity/polylogue/issues/1123)) ([#1178](https://github.com/Sinity/polylogue/issues/1178)) ([664502a](https://github.com/Sinity/polylogue/commit/664502a7e0a1dac36d1364767715eee8b7e3b23b))
* **reader:** granular SSE topics + live tail + progress events ([#1204](https://github.com/Sinity/polylogue/issues/1204)) ([#1363](https://github.com/Sinity/polylogue/issues/1363)) ([009ae17](https://github.com/Sinity/polylogue/commit/009ae173aea6122a1051378b3f726d779f3a27d3))
* **reader:** identity-preserving marks across reimport ([#1114](https://github.com/Sinity/polylogue/issues/1114)) ([#1188](https://github.com/Sinity/polylogue/issues/1188)) ([d42c6be](https://github.com/Sinity/polylogue/commit/d42c6bec46d575aeff38892b054aaf1729dccd80)), closes [#867](https://github.com/Sinity/polylogue/issues/867)
* **reader:** insights browser ([#1120](https://github.com/Sinity/polylogue/issues/1120)) ([#1177](https://github.com/Sinity/polylogue/issues/1177)) ([7c14cb5](https://github.com/Sinity/polylogue/commit/7c14cb5fe7e6e71f91b82587a35b0c48f1089a60))
* **reader:** message card action rail + fold policies + keyboard shortcuts ([#1202](https://github.com/Sinity/polylogue/issues/1202)) ([#1328](https://github.com/Sinity/polylogue/issues/1328)) ([d3b33d1](https://github.com/Sinity/polylogue/commit/d3b33d1b01566dbe282b3d3065e1aee482c21c96))
* **reader:** paste spans rendering + paste browser + diff detection ([#1201](https://github.com/Sinity/polylogue/issues/1201)) ([#1333](https://github.com/Sinity/polylogue/issues/1333)) ([7a85f3b](https://github.com/Sinity/polylogue/commit/7a85f3b7a15a5ccb2740651464e53fceb259a8d5))
* **reader:** persist reader workspaces ([#1056](https://github.com/Sinity/polylogue/issues/1056)) ([87c37f2](https://github.com/Sinity/polylogue/commit/87c37f20816fecd90d30deb7ca95311b29ee185f)), closes [#867](https://github.com/Sinity/polylogue/issues/867) [#993](https://github.com/Sinity/polylogue/issues/993)
* **reader:** provenance view with bounded raw access ([#1125](https://github.com/Sinity/polylogue/issues/1125)) ([#1172](https://github.com/Sinity/polylogue/issues/1172)) ([89b06f2](https://github.com/Sinity/polylogue/commit/89b06f2ba7fc7ca980b970e81c95a1d1c26449b6))
* **reader:** saved queries and named views UI ([#1118](https://github.com/Sinity/polylogue/issues/1118)) ([#1152](https://github.com/Sinity/polylogue/issues/1152)) ([1fe7487](https://github.com/Sinity/polylogue/commit/1fe74873eeeb066bff89530f80d298d8feb15fb5))
* **reader:** session lineage and topology view ([#1121](https://github.com/Sinity/polylogue/issues/1121)) ([#1176](https://github.com/Sinity/polylogue/issues/1176)) ([b3f36da](https://github.com/Sinity/polylogue/commit/b3f36da03fd5f75f25e00b4b5334b8fe2cd6293c))
* **reader:** support non-conversation target kinds for marks ([#1113](https://github.com/Sinity/polylogue/issues/1113)) ([#1179](https://github.com/Sinity/polylogue/issues/1179)) ([8a50004](https://github.com/Sinity/polylogue/commit/8a5000449f90888ba7440e6894c8fed248551ac8))
* **reader:** topology stack + branch chips + thread-continue shortcuts ([#1203](https://github.com/Sinity/polylogue/issues/1203)) ([#1343](https://github.com/Sinity/polylogue/issues/1343)) ([c6e646b](https://github.com/Sinity/polylogue/commit/c6e646b5f688f30bfcccd345cae91a7e6efba5bc))
* record daemon embedding progress ([#1532](https://github.com/Sinity/polylogue/issues/1532)) ([f727678](https://github.com/Sinity/polylogue/commit/f72767871371acc6e4bb7c43e6f37edbc7e4f441)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **schema:** add paste_boundary_state to messages ([#1706](https://github.com/Sinity/polylogue/issues/1706)) ([a06aee0](https://github.com/Sinity/polylogue/commit/a06aee090faf72531aff381d9ae89eb950f08946))
* **schema:** cwd/repo identity table ([#1253](https://github.com/Sinity/polylogue/issues/1253)) ([#1402](https://github.com/Sinity/polylogue/issues/1402)) ([931761d](https://github.com/Sinity/polylogue/commit/931761d67c4be4812f362a6d905cf7d180fa6c2e))
* **search:** explainable ranked results with stable pagination ([#873](https://github.com/Sinity/polylogue/issues/873)) ([#1108](https://github.com/Sinity/polylogue/issues/1108)) ([ad5ada2](https://github.com/Sinity/polylogue/commit/ad5ada27a7240534cdb036e6509e58a2b74d7400))
* **search:** expose lane evidence on hits ([#1562](https://github.com/Sinity/polylogue/issues/1562)) ([0dbc2e6](https://github.com/Sinity/polylogue/commit/0dbc2e60c8d2786b9abe875300de7a91dd6792b2))
* **search:** per-hit why-this-matched explanation + RRF lane preservation ([#1267](https://github.com/Sinity/polylogue/issues/1267)) ([#1386](https://github.com/Sinity/polylogue/issues/1386)) ([cd8fcb1](https://github.com/Sinity/polylogue/commit/cd8fcb19500828e97278173151981df4dda6591c))
* **search:** scoped vs global facets in archive/query layer ([#1269](https://github.com/Sinity/polylogue/issues/1269)) ([#1390](https://github.com/Sinity/polylogue/issues/1390)) ([529ea30](https://github.com/Sinity/polylogue/commit/529ea3058b42dc1590aa746296e04c9876a9b6f2))
* **search:** stable cursor/keyset pagination after ranked + filter paths ([#1268](https://github.com/Sinity/polylogue/issues/1268)) ([#1388](https://github.com/Sinity/polylogue/issues/1388)) ([02a1bb4](https://github.com/Sinity/polylogue/commit/02a1bb485a39f19edeff6826b3c49cbc30091276))
* **search:** typed ranked-result envelope + OpenAPI schema emission ([#1266](https://github.com/Sinity/polylogue/issues/1266)) ([#1370](https://github.com/Sinity/polylogue/issues/1370)) ([20d7955](https://github.com/Sinity/polylogue/commit/20d7955a1b58998aefcc1cae713897092205121f))
* session analysis primitives, commit correlation, OTLP spans, verifiability infrastructure ([#1738](https://github.com/Sinity/polylogue/issues/1738)) ([a3f5f6f](https://github.com/Sinity/polylogue/commit/a3f5f6ffbd0e0a7cbae236afd074d16c7c745718))
* show reader annotations in notes panel ([#1055](https://github.com/Sinity/polylogue/issues/1055)) ([fccf937](https://github.com/Sinity/polylogue/commit/fccf937492c28e2b9194e6fa2c11297bdeb31d45))
* **sources:** ingest Antigravity via language-server export ([#1041](https://github.com/Sinity/polylogue/issues/1041)) ([#1094](https://github.com/Sinity/polylogue/issues/1094)) ([c5e18f9](https://github.com/Sinity/polylogue/commit/c5e18f9f19e11942733e269db7280d5ea660033d))
* **sources:** ingest local agent sources ([#1024](https://github.com/Sinity/polylogue/issues/1024)) ([1618887](https://github.com/Sinity/polylogue/commit/1618887c18133d8d706f336d88bd6bfbf6da5f15))
* **surfaces:** add ContextPreamble envelope for SessionStart injection ([#1703](https://github.com/Sinity/polylogue/issues/1703)) ([85ad6d4](https://github.com/Sinity/polylogue/commit/85ad6d42a67158ab13f9587de6b5b272ab743654))
* **surfaces:** add paste_boundary_state to MessageRenderEnvelope ([#1655](https://github.com/Sinity/polylogue/issues/1655) follow-up) ([8f199e3](https://github.com/Sinity/polylogue/commit/8f199e3617205448255098553c07a02a4737823d))
* **surfaces:** closed vocabulary for reader action availability ([#1488](https://github.com/Sinity/polylogue/issues/1488)) ([#1664](https://github.com/Sinity/polylogue/issues/1664)) ([d257040](https://github.com/Sinity/polylogue/commit/d2570408fe38deb7ba9e9949a31b4e45d3c9473b))
* **surfaces:** MessageRenderEnvelope — additive fields for unified reader payload ([#1487](https://github.com/Sinity/polylogue/issues/1487)) ([#1665](https://github.com/Sinity/polylogue/issues/1665)) ([ea432b0](https://github.com/Sinity/polylogue/commit/ea432b00613cada3e09b2448b10b9deb608943ab))
* **telemetry:** add otlp_spans table ([#1686](https://github.com/Sinity/polylogue/issues/1686)) ([a6cdbaf](https://github.com/Sinity/polylogue/commit/a6cdbafbe06460a8b9d3e533901bc3097a06d9c4))
* **topology:** cycle rejection and quarantine ([#1260](https://github.com/Sinity/polylogue/issues/1260)) ([#1395](https://github.com/Sinity/polylogue/issues/1395)) ([bf715ca](https://github.com/Sinity/polylogue/commit/bf715ca9eb95795a72c8d063c77ebf936b3f7b4a))
* **topology:** late-parent arrival deterministic edge repair ([#1259](https://github.com/Sinity/polylogue/issues/1259)) ([#1392](https://github.com/Sinity/polylogue/issues/1392)) ([7295c2b](https://github.com/Sinity/polylogue/commit/7295c2bc15f019524e5bdd733f88473375b51161))
* **topology:** materialize session lineage graph ([#866](https://github.com/Sinity/polylogue/issues/866)) ([#1097](https://github.com/Sinity/polylogue/issues/1097)) ([1c116a7](https://github.com/Sinity/polylogue/commit/1c116a7d255f8aba4976b6f3d064786ecaa4eadd))
* **topology:** persist unresolved provider-native parent edges + closed enum ([#1258](https://github.com/Sinity/polylogue/issues/1258)) ([#1377](https://github.com/Sinity/polylogue/issues/1377)) ([9293cc2](https://github.com/Sinity/polylogue/commit/9293cc2da284f01aa98c0df77c8a71de7bb3847c))
* **topology:** typed read model API — ([14791ca](https://github.com/Sinity/polylogue/commit/14791ca8109d4447c6b6e8a281b507d0cc41de41))
* **topology:** typed read model API — ancestors/descendants/siblings/thread ([#1261](https://github.com/Sinity/polylogue/issues/1261)) ([#1391](https://github.com/Sinity/polylogue/issues/1391)) ([14791ca](https://github.com/Sinity/polylogue/commit/14791ca8109d4447c6b6e8a281b507d0cc41de41))
* **ui:** wire TUI through shared read-surface adapter ([#848](https://github.com/Sinity/polylogue/issues/848)) ([#1117](https://github.com/Sinity/polylogue/issues/1117)) ([ffec3f6](https://github.com/Sinity/polylogue/commit/ffec3f6c143e431c26be3ae18d16ab05798e5c46))
* **verifiability:** add MCP tool discovery, coverage lint, property tests, idempotency gate ([#1722](https://github.com/Sinity/polylogue/issues/1722)) ([da3ec3e](https://github.com/Sinity/polylogue/commit/da3ec3e3204d4bec3ff8e944a36b7524f7f31b2e))
* **verify:** CI workflow and manifest reality checks ([#590](https://github.com/Sinity/polylogue/issues/590), [#1064](https://github.com/Sinity/polylogue/issues/1064)) ([#1071](https://github.com/Sinity/polylogue/issues/1071)) ([1db6066](https://github.com/Sinity/polylogue/commit/1db60664f7dbb38cd42f0b80ebc02ecd4cab29ba))
* **verify:** parse CI workflows and prune aspirational manifest rows ([#1064](https://github.com/Sinity/polylogue/issues/1064)) ([#1088](https://github.com/Sinity/polylogue/issues/1088)) ([c31d0d2](https://github.com/Sinity/polylogue/commit/c31d0d209d3a33a5375416cdac0a2557cc53e4f6))
* **verify:** populate suppression registry and enforce type_ignore/noqa/no_cover ([#1062](https://github.com/Sinity/polylogue/issues/1062)) ([#1070](https://github.com/Sinity/polylogue/issues/1070)) ([d862a45](https://github.com/Sinity/polylogue/commit/d862a4501588480c49c2eaaf7b007d1f50c01945))
* **verify:** tier SLO catalog and enforce campaign artifact reality ([#1063](https://github.com/Sinity/polylogue/issues/1063)) ([#1086](https://github.com/Sinity/polylogue/issues/1086)) ([1f7d96a](https://github.com/Sinity/polylogue/commit/1f7d96a0f4f28f358a349e57d9362a09c031f1c3))
* **webui:** MK3 data-quality chips and informability ([#956](https://github.com/Sinity/polylogue/issues/956)) ([#1127](https://github.com/Sinity/polylogue/issues/1127)) ([a6865e6](https://github.com/Sinity/polylogue/commit/a6865e6e7e3eea7f8fa20b1f4fe91c3d24727136))
* **webui:** realtime update channel for ingest events ([#957](https://github.com/Sinity/polylogue/issues/957)) ([#1128](https://github.com/Sinity/polylogue/issues/1128)) ([632ffe7](https://github.com/Sinity/polylogue/commit/632ffe76be45a5822cb55f3286b91809b8ccb469))
* wire reader user-state controls ([#1051](https://github.com/Sinity/polylogue/issues/1051)) ([7c02d9b](https://github.com/Sinity/polylogue/commit/7c02d9badbe8bc0b181962a8a3227e25ac7c92d7))


### Fixed

* **antigravity:** flag fragmented brain-metadata sessions degraded ([#1856](https://github.com/Sinity/polylogue/issues/1856)) ([83efea0](https://github.com/Sinity/polylogue/commit/83efea0f2bbf0da245411f9f24c87daf5679f1e3)), closes [#1764](https://github.com/Sinity/polylogue/issues/1764)
* **archive:** write protocol artifact semantics at ingest, not projection ([#839](https://github.com/Sinity/polylogue/issues/839)) ([#1103](https://github.com/Sinity/polylogue/issues/1103)) ([43a94ca](https://github.com/Sinity/polylogue/commit/43a94ca7ff50fb4fdaf2f23d5c6a07a1aa799a62))
* avoid live filesystem resolution for archived/unsafe paths ([#1102](https://github.com/Sinity/polylogue/issues/1102)) ([5a539b3](https://github.com/Sinity/polylogue/commit/5a539b34fd7538c2d58142ba130eddd68bf0092c))
* **ci:** revive dead mutation-testing job and enforce kill-rate thresholds ([#1800](https://github.com/Sinity/polylogue/issues/1800)) ([d16257f](https://github.com/Sinity/polylogue/commit/d16257fb6d8813b7e5d4e133eba9052bfc777a45)), closes [#1733](https://github.com/Sinity/polylogue/issues/1733)
* **ci:** skip provenance/sbom on PR container builds ([#1373](https://github.com/Sinity/polylogue/issues/1373)) ([#1378](https://github.com/Sinity/polylogue/issues/1378)) ([cecf748](https://github.com/Sinity/polylogue/commit/cecf748659809e3159e2a4c85e6406a1f3fedb3e))
* **cli:** accept cursor option in root callback ([#1426](https://github.com/Sinity/polylogue/issues/1426)) ([0856e4f](https://github.com/Sinity/polylogue/commit/0856e4f58ab1ff2813fc20fd9731f98f04a6d160))
* **cli:** add `--json` alias to `polylogue status` to match siblings ([#1612](https://github.com/Sinity/polylogue/issues/1612)) ([#1619](https://github.com/Sinity/polylogue/issues/1619)) ([4870b56](https://github.com/Sinity/polylogue/commit/4870b564687762c7371a7fbcf9d28df3f3853c11))
* **cli:** allow status on large archives ([#1427](https://github.com/Sinity/polylogue/issues/1427)) ([fe35852](https://github.com/Sinity/polylogue/commit/fe358523f6ed2231e78cb9abc6a1d18921618e7b))
* **cli:** give honest next-step guidance after polylogue ingest ([#1693](https://github.com/Sinity/polylogue/issues/1693)) ([ae70895](https://github.com/Sinity/polylogue/commit/ae708951ca5a72365cb6ec68229008b104151081))
* **cli:** keep status responsive under ingest ([#1438](https://github.com/Sinity/polylogue/issues/1438)) ([9974bfb](https://github.com/Sinity/polylogue/commit/9974bfbaeb1c57da25f075e5feeaad7a9686dcdc))
* **cli:** resolve --latest for `messages` and `raw` verbs ([#1626](https://github.com/Sinity/polylogue/issues/1626)) ([#1637](https://github.com/Sinity/polylogue/issues/1637)) ([00e415e](https://github.com/Sinity/polylogue/commit/00e415e0aee759678f7b179116abc1051fbbd3de))
* **cli:** resolve --latest for top-level export/neighbors/diagnostics turns ([#1642](https://github.com/Sinity/polylogue/issues/1642)) ([#1649](https://github.com/Sinity/polylogue/issues/1649)) ([8f4941b](https://github.com/Sinity/polylogue/commit/8f4941ba4493fe43af5fff7ad2bc88d34e109f04))
* **cli:** resolve stale XFAIL in tags command ([#1012](https://github.com/Sinity/polylogue/issues/1012)) ([9e7dd57](https://github.com/Sinity/polylogue/commit/9e7dd57e50459c43bbeb5ecb49b88c3efc82e2dc))
* **cli:** route apply_modifiers tag mutations through custom repo when supplied ([#1012](https://github.com/Sinity/polylogue/issues/1012)) ([#1017](https://github.com/Sinity/polylogue/issues/1017)) ([8eae31e](https://github.com/Sinity/polylogue/commit/8eae31e133f967c63857c0d3c79801f4bf337498))
* **cli:** show embedding readiness in status fallback ([#1548](https://github.com/Sinity/polylogue/issues/1548)) ([8c0abd3](https://github.com/Sinity/polylogue/commit/8c0abd35ee28adfd24be1462b18b14660357c763))
* **cli:** wrap list/summary JSON output in paginated envelope ([#1618](https://github.com/Sinity/polylogue/issues/1618)) ([#1670](https://github.com/Sinity/polylogue/issues/1670)) ([4b54d77](https://github.com/Sinity/polylogue/commit/4b54d771ef6188a91babff9a8bee356d64b5c31f))
* **config:** redact secrets and harden write paths in sources ([#1748](https://github.com/Sinity/polylogue/issues/1748)) ([8e6f4c7](https://github.com/Sinity/polylogue/commit/8e6f4c74a562ef728401b34d71a540f3ad87f736))
* **core:** ensure optional_datetime always returns UTC-aware datetimes ([9735a6f](https://github.com/Sinity/polylogue/commit/9735a6fd49957c0414ce7bb4b2ec929e7f8524e3))
* **daemon/http:** suppress BrokenPipeError on client disconnect ([#1682](https://github.com/Sinity/polylogue/issues/1682)) ([7f417d0](https://github.com/Sinity/polylogue/commit/7f417d0ed7e68f8bdf6f36b2554fa6e53f46d947))
* **daemon:** align loopback definitions across daemon and browser_capture ([#1005](https://github.com/Sinity/polylogue/issues/1005)) ([#1011](https://github.com/Sinity/polylogue/issues/1011)) ([3b4f556](https://github.com/Sinity/polylogue/commit/3b4f55695778b1f4ddb3a7739eed81dfb418c62d))
* **daemon:** apply canonical SQLite profile in ArchiveStore ([#1806](https://github.com/Sinity/polylogue/issues/1806)) ([1b271e0](https://github.com/Sinity/polylogue/commit/1b271e0ca4bd5f05978bff1a5ecc1bfdd3cdf339))
* **daemon:** back off failed live ingest paths ([#1467](https://github.com/Sinity/polylogue/issues/1467)) ([154232c](https://github.com/Sinity/polylogue/commit/154232c0f163cfc6b7d32c47a80cff7795dcbfc4)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** block offline maintenance during convergence ([#1465](https://github.com/Sinity/polylogue/issues/1465)) ([7da2749](https://github.com/Sinity/polylogue/commit/7da27495c754fb55d942f5e93a3714c5c5a18df4)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** bound blob cleanup and append planning ([#1428](https://github.com/Sinity/polylogue/issues/1428)) ([1110956](https://github.com/Sinity/polylogue/commit/1110956c66cd32cae821adaaef84a03abc6a02cc))
* **daemon:** bound catch-up status and FTS freshness ([#1448](https://github.com/Sinity/polylogue/issues/1448)) ([4da83d2](https://github.com/Sinity/polylogue/commit/4da83d2993ed2d16030bc83be405ca4486abe743))
* **daemon:** bound FTS health counts ([#1436](https://github.com/Sinity/polylogue/issues/1436)) ([b1c386e](https://github.com/Sinity/polylogue/commit/b1c386e0ed7ca15c05b1fc4bbeb76e3662b3d829))
* **daemon:** bound FTS health probes ([#1445](https://github.com/Sinity/polylogue/issues/1445)) ([1eafad9](https://github.com/Sinity/polylogue/commit/1eafad90645d00922a1655c80c2bf48482941c17)), closes [#1444](https://github.com/Sinity/polylogue/issues/1444)
* **daemon:** bound FTS startup probes ([#1443](https://github.com/Sinity/polylogue/issues/1443)) ([9b9e105](https://github.com/Sinity/polylogue/commit/9b9e1059115ddc5f415a0f5befa1878a2da1e150)), closes [#1442](https://github.com/Sinity/polylogue/issues/1442)
* **daemon:** bound live catch-up ingest batches ([#1452](https://github.com/Sinity/polylogue/issues/1452)) ([9632fde](https://github.com/Sinity/polylogue/commit/9632fde68b7d73e78fa07e8c21d7d2d9ed26261d)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** bound startup FTS and isolate live ingest ([#1506](https://github.com/Sinity/polylogue/issues/1506)) ([18293ac](https://github.com/Sinity/polylogue/commit/18293ac9db705f74b3cb63f5c8849e07c275c01b))
* **daemon:** bound status and readiness probes ([#1474](https://github.com/Sinity/polylogue/issues/1474)) ([6483b25](https://github.com/Sinity/polylogue/commit/6483b25abe390fade09f04abe4f63133a4ea80e8))
* **daemon:** bulk repair streaming full-ingest FTS ([#1462](https://github.com/Sinity/polylogue/issues/1462)) ([591798e](https://github.com/Sinity/polylogue/commit/591798e7e4bdffbb55bf89f1d80e756ea77b3299))
* **daemon:** cache FTS freshness for search ([#1450](https://github.com/Sinity/polylogue/issues/1450)) ([aba8d6d](https://github.com/Sinity/polylogue/commit/aba8d6d342fc4989ae16f944586b7dfb19891dbb)), closes [#1243](https://github.com/Sinity/polylogue/issues/1243) [#1446](https://github.com/Sinity/polylogue/issues/1446) [#1447](https://github.com/Sinity/polylogue/issues/1447) [#1449](https://github.com/Sinity/polylogue/issues/1449)
* **daemon:** cap live full-ingest worker fanout ([#1425](https://github.com/Sinity/polylogue/issues/1425)) ([8f43351](https://github.com/Sinity/polylogue/commit/8f433514d684503cd7a9948a550347ba6cbab394))
* **daemon:** coalesce live convergence churn ([#1481](https://github.com/Sinity/polylogue/issues/1481)) ([279e610](https://github.com/Sinity/polylogue/commit/279e610e5829aa246842e603dd9222039ee9344f)), closes [#1249](https://github.com/Sinity/polylogue/issues/1249) [#845](https://github.com/Sinity/polylogue/issues/845)
* **daemon:** compact superseded raw snapshots ([#1483](https://github.com/Sinity/polylogue/issues/1483)) ([64bca08](https://github.com/Sinity/polylogue/commit/64bca08e6c70fcde11fc03d93d60fc7f53671ea0)), closes [#1482](https://github.com/Sinity/polylogue/issues/1482) [#818](https://github.com/Sinity/polylogue/issues/818)
* **daemon:** debt loop picks up stale profiles, not just missing ones ([#1620](https://github.com/Sinity/polylogue/issues/1620)) ([#1643](https://github.com/Sinity/polylogue/issues/1643)) ([f4e29a2](https://github.com/Sinity/polylogue/commit/f4e29a2c41f8613bf6daf5ccd09111c9f4fa2a09))
* **daemon:** defer hot insight rebuilds ([#1468](https://github.com/Sinity/polylogue/issues/1468)) ([3ff28fd](https://github.com/Sinity/polylogue/commit/3ff28fd991f36d85510c964f8bd9b0d72ccf6800)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** defer incomplete live appends ([#1460](https://github.com/Sinity/polylogue/issues/1460)) ([55f5698](https://github.com/Sinity/polylogue/commit/55f56980fe5d6d36d0a437e957c08a7592230e3a)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** drain embedding backlog under cost cap ([#1554](https://github.com/Sinity/polylogue/issues/1554)) ([397018a](https://github.com/Sinity/polylogue/commit/397018a1b6e59ffac61bdecf5849e59df75351b2))
* **daemon:** expose embedding readiness metrics ([#1547](https://github.com/Sinity/polylogue/issues/1547)) ([17cae99](https://github.com/Sinity/polylogue/commit/17cae997854fd5da76d36868d4b9c2b66d50ff04))
* **daemon:** expose live batch WAL metrics ([#1458](https://github.com/Sinity/polylogue/issues/1458)) ([3893bd2](https://github.com/Sinity/polylogue/commit/3893bd2fe91508c1c788a6673d35fec74b67a181))
* **daemon:** gate OTLP receiver on observability_enabled + body cap + remote-auth ([#1604](https://github.com/Sinity/polylogue/issues/1604)) ([#1610](https://github.com/Sinity/polylogue/issues/1610)) ([bf1886e](https://github.com/Sinity/polylogue/commit/bf1886e4d219d1b0040145009fd0f07a7c9df50e))
* **daemon:** harden live-ingest against write thrash and retry storms ([#1853](https://github.com/Sinity/polylogue/issues/1853)) ([78954aa](https://github.com/Sinity/polylogue/commit/78954aaf00cfb6516cebc47692a38a34049ea83e))
* **daemon:** ingest ZIP members in live batch instead of silently excluding ([#1797](https://github.com/Sinity/polylogue/issues/1797)) ([5098a31](https://github.com/Sinity/polylogue/commit/5098a31490406c6589e2d6386c8dbbfc1b4d6a75)), closes [#1683](https://github.com/Sinity/polylogue/issues/1683)
* **daemon:** keep Codex append identity hot ([#1461](https://github.com/Sinity/polylogue/issues/1461)) ([2179002](https://github.com/Sinity/polylogue/commit/2179002be2bbad47ae71d35779b2b540a81ee9fb))
* **daemon:** keep exact FTS scans out of startup ([#1451](https://github.com/Sinity/polylogue/issues/1451)) ([9136558](https://github.com/Sinity/polylogue/commit/9136558544896391507ac8970b2d72127b33f1e5)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** keep FTS triggers active during ingest ([#1437](https://github.com/Sinity/polylogue/issues/1437)) ([5692050](https://github.com/Sinity/polylogue/commit/56920506054b2ab77d6c550c31d0afcb177bd9d6))
* **daemon:** keep live convergence responsive ([#1473](https://github.com/Sinity/polylogue/issues/1473)) ([4e3e5c4](https://github.com/Sinity/polylogue/commit/4e3e5c49c96463e8dba24e2401496e73ddd4285f))
* **daemon:** keep live health bounded ([#1476](https://github.com/Sinity/polylogue/issues/1476)) ([34fae42](https://github.com/Sinity/polylogue/commit/34fae42dc0645caa861ea5021be60e451570da63))
* **daemon:** measure FTS coverage against search_text, not display text ([#1805](https://github.com/Sinity/polylogue/issues/1805)) ([b3b97b5](https://github.com/Sinity/polylogue/commit/b3b97b5d4020e56a13aa21299988122a3c64a869))
* **daemon:** reduce catch-up replay amplification ([#1459](https://github.com/Sinity/polylogue/issues/1459)) ([f5688cb](https://github.com/Sinity/polylogue/commit/f5688cb149e94763852411a5a8b2eea89cabbc05))
* **daemon:** reduce large ingest FTS amplification ([#1453](https://github.com/Sinity/polylogue/issues/1453)) ([6c826cf](https://github.com/Sinity/polylogue/commit/6c826cf289fcab828e28becc7f7001022c3f84bd))
* **daemon:** remove broad startup maintenance scans ([#1464](https://github.com/Sinity/polylogue/issues/1464)) ([2dc505c](https://github.com/Sinity/polylogue/commit/2dc505ca60f156147c9a923383fc87e10c69eb2e)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** repair fresh-archive ingest dropping sessions + false 0% FTS ([#1804](https://github.com/Sinity/polylogue/issues/1804)) ([09131da](https://github.com/Sinity/polylogue/commit/09131dad0499fb97254d5459528d333e1e2a791c))
* **daemon:** repair FTS overcount drift ([#1432](https://github.com/Sinity/polylogue/issues/1432)) ([c88d3be](https://github.com/Sinity/polylogue/commit/c88d3be5216956953c1520f112b4a7553b05780b))
* **daemon:** report FTS ledger counts ([#1520](https://github.com/Sinity/polylogue/issues/1520)) ([d5a5b71](https://github.com/Sinity/polylogue/commit/d5a5b71f8cb8db21715356547a7b5e8fb70a1a02)), closes [#1515](https://github.com/Sinity/polylogue/issues/1515)
* **daemon:** report WAL checkpoint blockers ([#1456](https://github.com/Sinity/polylogue/issues/1456)) ([23de93f](https://github.com/Sinity/polylogue/commit/23de93f9429e2aef461eb61d7810c35f303615a0))
* **daemon:** requeue live ingest when archive is busy ([#1454](https://github.com/Sinity/polylogue/issues/1454)) ([38a268e](https://github.com/Sinity/polylogue/commit/38a268e5c0676217584491e5f479e812bb503768))
* **daemon:** restore all active FTS triggers ([#1455](https://github.com/Sinity/polylogue/issues/1455)) ([2ba4b07](https://github.com/Sinity/polylogue/commit/2ba4b079eff474118617c77a30da3a678bd779e4)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** skip redundant write FTS repairs ([#1472](https://github.com/Sinity/polylogue/issues/1472)) ([dae61fb](https://github.com/Sinity/polylogue/commit/dae61fba9890119fde35348fb17eb3e43d3fa9a7))
* **daemon:** stop schema-blocked background work ([#1466](https://github.com/Sinity/polylogue/issues/1466)) ([fd93d0d](https://github.com/Sinity/polylogue/commit/fd93d0d3081c025251b0b1afe7368eb3907d6494))
* **daemon:** target startup FTS gap repair ([#1441](https://github.com/Sinity/polylogue/issues/1441)) ([81024be](https://github.com/Sinity/polylogue/commit/81024be24834d2c97a04de36d464fb6c348a7959)), closes [#1439](https://github.com/Sinity/polylogue/issues/1439)
* **daemon:** tolerate locked cursor bookkeeping ([#1516](https://github.com/Sinity/polylogue/issues/1516)) ([2c245d6](https://github.com/Sinity/polylogue/commit/2c245d6e43a52fb92bb293cedf5f0d96478228df)), closes [#1515](https://github.com/Sinity/polylogue/issues/1515)
* **daemon:** treat convergence locks as archive busy ([#1519](https://github.com/Sinity/polylogue/issues/1519)) ([d99baf2](https://github.com/Sinity/polylogue/commit/d99baf2ed7479346ccf5826713a3b1641c7c02d3)), closes [#1517](https://github.com/Sinity/polylogue/issues/1517)
* **daemon:** treat fresh-init bootstrap-in-flight as 'nothing to repair' ([#1603](https://github.com/Sinity/polylogue/issues/1603)) ([#1609](https://github.com/Sinity/polylogue/issues/1609)) ([d373810](https://github.com/Sinity/polylogue/commit/d373810b0cf858e3b40281d73360f99b649a2746))
* **daemon:** trust ready FTS triggers on live appends ([#1469](https://github.com/Sinity/polylogue/issues/1469)) ([1478cc1](https://github.com/Sinity/polylogue/commit/1478cc1842a2435ee72a284a77ec6062a176ffc7))
* **daemon:** write fts_freshness_state ready rows after startup readiness pass ([#1628](https://github.com/Sinity/polylogue/issues/1628)) ([#1650](https://github.com/Sinity/polylogue/issues/1650)) ([921c191](https://github.com/Sinity/polylogue/commit/921c191082269c86ac5a541668e08aaf36a6edb2))
* **devtools:** close leaked read connection in archive-space-report ([#1774](https://github.com/Sinity/polylogue/issues/1774)) ([fb6b274](https://github.com/Sinity/polylogue/commit/fb6b2745c74541088043373f7b164f4f08e1d182))
* **devtools:** fill [#1737](https://github.com/Sinity/polylogue/issues/1737) gaps — PyYAML parsers, stale spine docs ([3604177](https://github.com/Sinity/polylogue/commit/3604177556322b6591df1a4cf1b2ef4651d6e935))
* **devtools:** preserve testmon affected selection ([#1550](https://github.com/Sinity/polylogue/issues/1550)) ([a2e3bda](https://github.com/Sinity/polylogue/commit/a2e3bda44282321d2292035ee3db4aada3c29469))
* **devtools:** resolve repo root from cwd, not hard-pin ([#1209](https://github.com/Sinity/polylogue/issues/1209)) ([#1239](https://github.com/Sinity/polylogue/issues/1239)) ([c49b9b2](https://github.com/Sinity/polylogue/commit/c49b9b271b0f3afe52449209187c7576c8e6fb01)), closes [#1193](https://github.com/Sinity/polylogue/issues/1193)
* **devtools:** run testmon verify with workers ([#1478](https://github.com/Sinity/polylogue/issues/1478)) ([7db54fd](https://github.com/Sinity/polylogue/commit/7db54fdc94eb7a9983080ef63a80682c0b5a5beb))
* **embeddings:** apply run cost cap during backfill ([#1544](https://github.com/Sinity/polylogue/issues/1544)) ([a255a38](https://github.com/Sinity/polylogue/commit/a255a382fb78ff26928d7467752ce17f0e3edb44))
* **embeddings:** keep status readable across schema bumps ([#1542](https://github.com/Sinity/polylogue/issues/1542)) ([07a501e](https://github.com/Sinity/polylogue/commit/07a501e87e7acacc268e26bf41d392e07e5e465c))
* **embeddings:** unify daemon readiness status ([#1545](https://github.com/Sinity/polylogue/issues/1545)) ([2de683f](https://github.com/Sinity/polylogue/commit/2de683f328eb70320a2d59ba32b7fbb622604e80))
* **facets:** hydrate ConversationSummary.message_count from conversation_stats ([#1623](https://github.com/Sinity/polylogue/issues/1623)) ([#1636](https://github.com/Sinity/polylogue/issues/1636)) ([c07a9dd](https://github.com/Sinity/polylogue/commit/c07a9dd64f4cf42440536250f7e1c642d7910458))
* **filter:** commutativity bug across chatgpt/claude-ai provider params ([#1338](https://github.com/Sinity/polylogue/issues/1338)) ([#1374](https://github.com/Sinity/polylogue/issues/1374)) ([5679040](https://github.com/Sinity/polylogue/commit/56790404f577bc984fec3387871f87086202c513))
* **fts:** dedupe targeted repair conversations ([#1512](https://github.com/Sinity/polylogue/issues/1512)) ([4a5a53e](https://github.com/Sinity/polylogue/commit/4a5a53eb0464b108dc1869be611e4d5a0386f16b)), closes [#1509](https://github.com/Sinity/polylogue/issues/1509)
* **fts:** ignore empty text in repair predicate ([#1514](https://github.com/Sinity/polylogue/issues/1514)) ([6f89fa3](https://github.com/Sinity/polylogue/commit/6f89fa3b78de2a413b8f21b40ca306766bc339f6)), closes [#1513](https://github.com/Sinity/polylogue/issues/1513)
* harden retrieval freshness and embedding catch-up ([#1522](https://github.com/Sinity/polylogue/issues/1522)) ([28abc7e](https://github.com/Sinity/polylogue/commit/28abc7e0b6aecbd9cac45441a35036681fdd72c2)), closes [#1521](https://github.com/Sinity/polylogue/issues/1521) [#1503](https://github.com/Sinity/polylogue/issues/1503)
* harden search freshness and semantic readiness ([#1504](https://github.com/Sinity/polylogue/issues/1504)) ([cd46fcf](https://github.com/Sinity/polylogue/commit/cd46fcfb9860850c841ef8a5d2de80181e76d1a3)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* hermes message_count off-by-one, dead tool_result comparison, duplicate DDL ([b9307ff](https://github.com/Sinity/polylogue/commit/b9307ff5551bce75cc88bdfbfa517a300407f7ce)), closes [#1751](https://github.com/Sinity/polylogue/issues/1751) [#1752](https://github.com/Sinity/polylogue/issues/1752) [#1753](https://github.com/Sinity/polylogue/issues/1753)
* **ingest:** bound targeted FTS repair memory ([#1510](https://github.com/Sinity/polylogue/issues/1510)) ([ccfb29b](https://github.com/Sinity/polylogue/commit/ccfb29b9eb7b3ed569d47b0bd8b968592e2bfb10)), closes [#1509](https://github.com/Sinity/polylogue/issues/1509)
* **ingest:** preserve detected provider; allow null raw source_name; assurance + pins ([#1770](https://github.com/Sinity/polylogue/issues/1770)) ([10cacb7](https://github.com/Sinity/polylogue/commit/10cacb743b21c2391bc95a79ce09c05c7667a78e))
* **ingest:** release drained worker result payloads ([#1508](https://github.com/Sinity/polylogue/issues/1508)) ([a577cd6](https://github.com/Sinity/polylogue/commit/a577cd6552c6bcf636c7ce17af1e46bf9f5fab09))
* **insights:** expose tool-active rollup evidence ([#1551](https://github.com/Sinity/polylogue/issues/1551)) ([5d2842e](https://github.com/Sinity/polylogue/commit/5d2842e2149a6fd8ee81c2f1e0acfe4466e0dea2))
* **insights:** infer session topics ([#1536](https://github.com/Sinity/polylogue/issues/1536)) ([d9863f7](https://github.com/Sinity/polylogue/commit/d9863f7bbc3a7339a5b5e1e778b3e3fae3d96ac8)), closes [#1525](https://github.com/Sinity/polylogue/issues/1525)
* **insights:** materialize session costs ([#1535](https://github.com/Sinity/polylogue/issues/1535)) ([d4828b4](https://github.com/Sinity/polylogue/commit/d4828b4fdddd6d2eeb4515ae72b401cc79a426fe)), closes [#1525](https://github.com/Sinity/polylogue/issues/1525)
* **insights:** normalize session timestamps ([#1534](https://github.com/Sinity/polylogue/issues/1534)) ([5bb72fb](https://github.com/Sinity/polylogue/commit/5bb72fb68da83b05be184b6299a77e347acb2cb7)), closes [#1525](https://github.com/Sinity/polylogue/issues/1525)
* **maintenance:** filter replay sources lexically ([#1457](https://github.com/Sinity/polylogue/issues/1457)) ([6da1b69](https://github.com/Sinity/polylogue/commit/6da1b695f04183b42a1db1be40e7acf44475291e))
* **maintenance:** support tuple repair candidate rows ([#1475](https://github.com/Sinity/polylogue/issues/1475)) ([9e7f0a3](https://github.com/Sinity/polylogue/commit/9e7f0a381f9dee946545eb51eace7a53898434a1))
* **mcp:** add correlate_session and session_tool_timing to envelope contracts, regen witness ([144abbe](https://github.com/Sinity/polylogue/commit/144abbe303308e9ce7cea3d1e35a98354e5d2bd5))
* **mcp:** gate semantic search on embedding readiness ([#1574](https://github.com/Sinity/polylogue/issues/1574)) ([3695456](https://github.com/Sinity/polylogue/commit/3695456f7635dd01230b980462e3d6496307e056)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **mcp:** isolate tool exceptions; never let one tool kill the stdio server ([#1611](https://github.com/Sinity/polylogue/issues/1611), [#1621](https://github.com/Sinity/polylogue/issues/1621)) ([#1631](https://github.com/Sinity/polylogue/issues/1631)) ([789fc77](https://github.com/Sinity/polylogue/commit/789fc77bc1461765ea849f461da88f05f625b082))
* **parsers/claude-code:** skip type=progress hook lifecycle events ([#1617](https://github.com/Sinity/polylogue/issues/1617)) ([#1641](https://github.com/Sinity/polylogue/issues/1641)) ([34b957a](https://github.com/Sinity/polylogue/commit/34b957a83e6f0e770ac53e0e1532ddbe7fdb15a4))
* **parser:** traverse ChatGPT message graph and extract non-parts content ([#1744](https://github.com/Sinity/polylogue/issues/1744)) ([2d63fea](https://github.com/Sinity/polylogue/commit/2d63feacb02dd098e738b8594b650681c9ef3f03))
* **phases:** fall back to provider_event timestamps when messages have none ([#1624](https://github.com/Sinity/polylogue/issues/1624)) ([#1634](https://github.com/Sinity/polylogue/issues/1634)) ([483a9bb](https://github.com/Sinity/polylogue/commit/483a9bbe7ec8a5fbdf2673cfb6b45b296cf1c88e))
* **pipeline:** surface silent parse-time record loss across ingest ([#1745](https://github.com/Sinity/polylogue/issues/1745)) ([3a47c8f](https://github.com/Sinity/polylogue/commit/3a47c8f8b8e2c3af7fd1f080374efbfd72d4e58a))
* repair silently-broken read-surface filters and stats queries post-[#1743](https://github.com/Sinity/polylogue/issues/1743) ([#1796](https://github.com/Sinity/polylogue/issues/1796)) ([829e47a](https://github.com/Sinity/polylogue/commit/829e47a7028df57d2766f050c6a74be053ddc76e))
* report pytest counts in verify notifications ([#1049](https://github.com/Sinity/polylogue/issues/1049)) ([b7536ae](https://github.com/Sinity/polylogue/commit/b7536ae81cda5607b9cba176d9d81f0dd0712497))
* **repository:** hydrate message_count in get_summary + search_summaries ([#1630](https://github.com/Sinity/polylogue/issues/1630)) ([#1669](https://github.com/Sinity/polylogue/issues/1669)) ([92cb112](https://github.com/Sinity/polylogue/commit/92cb11253035f5bb32a421982a44b02a70857e57))
* resolve 18 pre-existing test failures and optimize dev infrastructure ([fc28ddc](https://github.com/Sinity/polylogue/commit/fc28ddc9a1f3f592444a9b9148de1c3e9873f33d))
* resolve mypy strict-mode issues in verify cache read path ([4c16c78](https://github.com/Sinity/polylogue/commit/4c16c7822d95032da7895d6fd9325dbd4f253691))
* **schema:** backfill claude-ai distribution annotations ([#1027](https://github.com/Sinity/polylogue/issues/1027)) ([#1111](https://github.com/Sinity/polylogue/issues/1111)) ([1acf3a0](https://github.com/Sinity/polylogue/commit/1acf3a0c8863ba20ac012850755830e02c3705bc))
* **schema:** remove duplicate source_name column from artifact_observations DDL ([#1022](https://github.com/Sinity/polylogue/issues/1022)) ([081a659](https://github.com/Sinity/polylogue/commit/081a6591cb8027ca3c3595e4673a899718ec1b8d))
* **search:** allow auto cursor follow-up ([#1563](https://github.com/Sinity/polylogue/issues/1563)) ([265b5e7](https://github.com/Sinity/polylogue/commit/265b5e7d0ad81ac7ef53b5b8909c79445d94f6cf))
* **search:** enforce FTS freshness invariant ([#1440](https://github.com/Sinity/polylogue/issues/1440)) ([39c636d](https://github.com/Sinity/polylogue/commit/39c636dc2df95e612e66e982e466a60f7ca78e63)), closes [#1439](https://github.com/Sinity/polylogue/issues/1439)
* **search:** gate semantic queries on embedding readiness ([#1546](https://github.com/Sinity/polylogue/issues/1546)) ([1be50a4](https://github.com/Sinity/polylogue/commit/1be50a4dae7e43e590f78d09e8607d56d84c3095))
* **search:** ground FTS freshness in invariants ([#1500](https://github.com/Sinity/polylogue/issues/1500)) ([ee5a507](https://github.com/Sinity/polylogue/commit/ee5a50741a2f944df47e6f3fa2a4f4cebfcdb652)), closes [#1499](https://github.com/Sinity/polylogue/issues/1499)
* **search:** repair FTS during live ingest ([#1502](https://github.com/Sinity/polylogue/issues/1502)) ([ff0fcc9](https://github.com/Sinity/polylogue/commit/ff0fcc9489da1b0b7ad92dbb46065661bf273550)), closes [#1501](https://github.com/Sinity/polylogue/issues/1501)
* serialize web-reader HTTP server tests under shared xdist group ([d30e967](https://github.com/Sinity/polylogue/commit/d30e967a1e9032ced24e4fcd28e8a1455ae9ae62))
* **sources:** parse JSON from non-seekable zip-entry streams ([#1768](https://github.com/Sinity/polylogue/issues/1768)) ([1ee0907](https://github.com/Sinity/polylogue/commit/1ee0907d8ff12b5e9b52902951224661c67a82c7))
* stop live ingest on schema layout mismatch ([382b702](https://github.com/Sinity/polylogue/commit/382b702dc7874c915c3f4d6298609644a342023b))
* **storage:** canonicalize archive conversation timestamps ([#1555](https://github.com/Sinity/polylogue/issues/1555)) ([576c4b4](https://github.com/Sinity/polylogue/commit/576c4b4f5d5eb7f1fb8ba7a20dc1b3d866e55fb7))
* **storage:** close leaked sqlite connections, unsuppress ResourceWarning ([#1772](https://github.com/Sinity/polylogue/issues/1772)) ([7f5dacd](https://github.com/Sinity/polylogue/commit/7f5dacd2bf1a140499fabda11549e1327ad32c24))
* **storage:** FTS trigger commit ordering + SIGKILL safety ([#1242](https://github.com/Sinity/polylogue/issues/1242)) ([#1362](https://github.com/Sinity/polylogue/issues/1362)) ([c931bca](https://github.com/Sinity/polylogue/commit/c931bca874cb73dc04504534e169f5a66ebc371f))
* **storage:** index raw cleanup on existing archives ([#1485](https://github.com/Sinity/polylogue/issues/1485)) ([6e05414](https://github.com/Sinity/polylogue/commit/6e05414db6b52710006e21fa9e0a752bd8176d3c)), closes [#1484](https://github.com/Sinity/polylogue/issues/1484)
* **storage:** lower WAL autocheckpoint from 40 MB to 4 MB ([3c5cc1a](https://github.com/Sinity/polylogue/commit/3c5cc1a075ef6e5c7b55df89dced68a2fc767644))
* **storage:** prevent FTS orphans during replacement ([#1435](https://github.com/Sinity/polylogue/issues/1435)) ([f0219c6](https://github.com/Sinity/polylogue/commit/f0219c6f7b3d17949b00f305ee684d33168bbdf7))
* **storage:** release blob leases on failure and enforce GC generation gate ([#1746](https://github.com/Sinity/polylogue/issues/1746)) ([2744076](https://github.com/Sinity/polylogue/commit/2744076b4b6d57f0a30a5fb81afda56db352fb35))
* **storage:** remove divergent cross-validator, trust canonical content hash ([#1747](https://github.com/Sinity/polylogue/issues/1747)) ([c471e5b](https://github.com/Sinity/polylogue/commit/c471e5b1efc306c3c483891f0fc53aea40b8eaac))
* **storage:** run_blob_gc unlinks sharded path + dry-run + history ([#1190](https://github.com/Sinity/polylogue/issues/1190)) ([#1318](https://github.com/Sinity/polylogue/issues/1318)) ([ef6a09a](https://github.com/Sinity/polylogue/commit/ef6a09a183495fbb267efc82d93f096a20ff56d2))
* **suppressions:** tighten native enforcement and discipline ledger ([#1062](https://github.com/Sinity/polylogue/issues/1062)) ([#1093](https://github.com/Sinity/polylogue/issues/1093)) ([7167e30](https://github.com/Sinity/polylogue/commit/7167e305f39d2fad1bf7b37e7545f93899fdeecf))
* **synthetic:** weighted anyOf selection + diverse wire format fallbacks ([133f812](https://github.com/Sinity/polylogue/commit/133f8126e62bff9dc4965c679f9dcc28a78d2be9)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **tests:** clear inherited proof-law staleness blocking PR wave ([#1090](https://github.com/Sinity/polylogue/issues/1090)) ([f95cd9c](https://github.com/Sinity/polylogue/commit/f95cd9c353dfbdff320ee02ebc2b4e27f81c2029))
* **tests:** clear inherited test drift ([#1180](https://github.com/Sinity/polylogue/issues/1180)) ([#1187](https://github.com/Sinity/polylogue/issues/1187)) ([c3f3d82](https://github.com/Sinity/polylogue/commit/c3f3d82673d00df68a5296026d5ea5a17feb1da9))
* **tests:** clear inherited test failures from [#1007](https://github.com/Sinity/polylogue/issues/1007)/master ([#1095](https://github.com/Sinity/polylogue/issues/1095)) ([d9111f0](https://github.com/Sinity/polylogue/commit/d9111f0fa3c217c868370f5e0bd8ffb16e81ca37)), closes [#1012](https://github.com/Sinity/polylogue/issues/1012)
* **tests:** isolate concurrent pytest runs with per-run tmpfs basetemp ([#1785](https://github.com/Sinity/polylogue/issues/1785)) ([e789ca5](https://github.com/Sinity/polylogue/commit/e789ca5e24e5b6538911b090e5b19bc3068bdd34))
* **tests:** repair master drift from schema v15→v17 changes ([#1208](https://github.com/Sinity/polylogue/issues/1208)) ([#1288](https://github.com/Sinity/polylogue/issues/1288)) ([3012bda](https://github.com/Sinity/polylogue/commit/3012bda2de82f67c170e2cd1cb06a5a054c8afdd))
* **tests:** restore inherited-failure baseline; aggregate_message_stats works on read profile ([#1675](https://github.com/Sinity/polylogue/issues/1675)) ([b9c8081](https://github.com/Sinity/polylogue/commit/b9c8081e2df04bb6e624d80e7c4c292d798b6531))
* **tests:** teach CLI parity adapter the typed search-result envelope ([#1384](https://github.com/Sinity/polylogue/issues/1384)) ([9743e00](https://github.com/Sinity/polylogue/commit/9743e00370ed6428ec70ed702a1ff4f56e5340ca))
* **tests:** workspace_env unsets POLYLOGUE_* host env vars ([#1325](https://github.com/Sinity/polylogue/issues/1325)) ([#1326](https://github.com/Sinity/polylogue/issues/1326)) ([ee22230](https://github.com/Sinity/polylogue/commit/ee22230ebf0a71113fd4ea12fa32a39ec2656613))
* **verification:** narrow proof routing false positives and activate evidence report blocking ([#1079](https://github.com/Sinity/polylogue/issues/1079)) ([fc3c43f](https://github.com/Sinity/polylogue/commit/fc3c43f3e1f73c9ea9f249b13471991c54274672))
* **viewport:** tighten affected_paths heuristic — reject sed, version strings, attribute access ([#1622](https://github.com/Sinity/polylogue/issues/1622)) ([#1646](https://github.com/Sinity/polylogue/issues/1646)) ([3596e08](https://github.com/Sinity/polylogue/commit/3596e08ddfcd790d3d3029da287bdd43bb9ff8a0))
* **watcher:** accept .zip, .json, .ndjson in inbox watch source ([#1692](https://github.com/Sinity/polylogue/issues/1692)) ([9038e09](https://github.com/Sinity/polylogue/commit/9038e09f4091f236dd889c1e2a95668b916947a5))
* **watcher:** interleave catch-up scan by source family ([#1616](https://github.com/Sinity/polylogue/issues/1616)) ([#1644](https://github.com/Sinity/polylogue/issues/1644)) ([3bfa0da](https://github.com/Sinity/polylogue/commit/3bfa0da3641df38b673d7d95f921029d7a6142ed))


### Changed

* **bench:** add FTS trigger amplification benchmark ([#1698](https://github.com/Sinity/polylogue/issues/1698)) ([8798b22](https://github.com/Sinity/polylogue/commit/8798b2280be81050ccce16060ae506cc189d7641))
* content-hash caching + parallel rendering for render-all ([e003e51](https://github.com/Sinity/polylogue/commit/e003e5111d855d19d2460c0afa712f52110e9b3c))
* **daemon/metrics:** aggregate per-source counts via conversation_stats ([#1629](https://github.com/Sinity/polylogue/issues/1629)) ([#1645](https://github.com/Sinity/polylogue/issues/1645)) ([acd1f1f](https://github.com/Sinity/polylogue/commit/acd1f1f9216206951879fd1e910d3065cfd56bf4))
* **daemon:** avoid virtual FTS status counts ([#1431](https://github.com/Sinity/polylogue/issues/1431)) ([a148aee](https://github.com/Sinity/polylogue/commit/a148aee1a14434ec5231e8a98fc1f0bb0a8ab205))
* **daemon:** bound live ingest memory ([#1429](https://github.com/Sinity/polylogue/issues/1429)) ([bdc1d72](https://github.com/Sinity/polylogue/commit/bdc1d724f7dd3ce6ff50309ff7077d6b54f050fc))
* **daemon:** bound session insight convergence reads ([#1463](https://github.com/Sinity/polylogue/issues/1463)) ([cc0bf5b](https://github.com/Sinity/polylogue/commit/cc0bf5ba02c57499fedd8a93879e3a546f617fd6)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** bound startup FTS checks ([#1434](https://github.com/Sinity/polylogue/issues/1434)) ([4cab346](https://github.com/Sinity/polylogue/commit/4cab346a1cd23358fdb296187b05de5ac98be483))
* **daemon:** bound startup FTS readiness ([#1038](https://github.com/Sinity/polylogue/issues/1038)) ([566001f](https://github.com/Sinity/polylogue/commit/566001f5f417d32a23e2f7d6d8d662324f0221e7)), closes [#1036](https://github.com/Sinity/polylogue/issues/1036)
* **daemon:** chunk large live appends ([#1433](https://github.com/Sinity/polylogue/issues/1433)) ([3dea102](https://github.com/Sinity/polylogue/commit/3dea102964747c98b81629b7cf9b205b82d79e59))
* **daemon:** converge hot session ingest serially ([#1039](https://github.com/Sinity/polylogue/issues/1039)) ([b4563b5](https://github.com/Sinity/polylogue/commit/b4563b5b6b930ddc2fe6728470fc27f9c8e7728f))
* **daemon:** downgrade FTS trigger drift to INFO during fresh bulk catch-up ([#1613](https://github.com/Sinity/polylogue/issues/1613)) ([#1652](https://github.com/Sinity/polylogue/issues/1652)) ([afc5118](https://github.com/Sinity/polylogue/commit/afc5118a77d5d89b1ac2690de1337774c1c85e10))
* **daemon:** huge-session streaming + write-amplification reduction ([#1244](https://github.com/Sinity/polylogue/issues/1244)) ([#1365](https://github.com/Sinity/polylogue/issues/1365)) ([58540b1](https://github.com/Sinity/polylogue/commit/58540b1b26c814070134c8e4b453baabbeec77cb))
* **daemon:** observe and retry live catch-up convergence ([#1033](https://github.com/Sinity/polylogue/issues/1033)) ([a73d69b](https://github.com/Sinity/polylogue/commit/a73d69b758f0945d24fd6541fc32dfb5bb2c9f49))
* **daemon:** reduce convergence residual workload ([#1037](https://github.com/Sinity/polylogue/issues/1037)) ([3311caf](https://github.com/Sinity/polylogue/commit/3311caf3ec477d2748f35a940376d9e1053bfaa8))
* **daemon:** reduce live catch-up amplification ([#1430](https://github.com/Sinity/polylogue/issues/1430)) ([dd9f200](https://github.com/Sinity/polylogue/commit/dd9f200947a229cf452b15278b97e29fdfef5453))
* **daemon:** skip inode-churn catch-up rehash ([#1034](https://github.com/Sinity/polylogue/issues/1034)) ([d1fc26b](https://github.com/Sinity/polylogue/commit/d1fc26bf3d5ca863d6a774f7d27f23a64b236a2e))
* **daemon:** update append stats incrementally ([#1470](https://github.com/Sinity/polylogue/issues/1470)) ([010fe6f](https://github.com/Sinity/polylogue/commit/010fe6f7b1a9dc7d6ed84a0a7eec4a1a648f5d91))
* **daemon:** use indexed action FTS probe ([#1035](https://github.com/Sinity/polylogue/issues/1035)) ([62193c7](https://github.com/Sinity/polylogue/commit/62193c7743e03010dde1c0e18e5ea06ba78d9bde))
* defer heavy imports in CLI startup path (2.8s → 1.7s) ([ba2745e](https://github.com/Sinity/polylogue/commit/ba2745e473955cb1918d3c4474ab00dadb0116d4))
* **fts:** make content_blocks INSERT trigger O(1) per block ([#1705](https://github.com/Sinity/polylogue/issues/1705)) ([8ac581f](https://github.com/Sinity/polylogue/commit/8ac581f162ac253360bef608b785efe77b44ce85))
* **hooks:** skip pre-push verify when HEAD already stamped by devtools verify ([b17c522](https://github.com/Sinity/polylogue/commit/b17c522b7d8f9378835f8f421166b75ac3b4a303)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **insights:** emit progress per table during full session-insight rebuild ([#1653](https://github.com/Sinity/polylogue/issues/1653)) ([8d3555d](https://github.com/Sinity/polylogue/commit/8d3555decd1dd3737e45477d18374ed5f029f3fd))
* **maintenance:** target stale session insight repairs ([#1471](https://github.com/Sinity/polylogue/issues/1471)) ([27c40d7](https://github.com/Sinity/polylogue/commit/27c40d7124c816ace085065fa87b9dca5c1866eb))
* **metrics:** consolidate raw_conversations COUNT queries ([#1702](https://github.com/Sinity/polylogue/issues/1702)) ([4629b2e](https://github.com/Sinity/polylogue/commit/4629b2ebb886144438f4c17a9869c5e3f62de52a))
* optimize dev loop — verify tiers, flake checks, shellHook caching, systemd hardening ([f469afc](https://github.com/Sinity/polylogue/commit/f469afc5c9def3dfff58d4f2ba25060c7513bf78))
* skip redundant verify runs via worktree-fingerprint cache ([7b080c3](https://github.com/Sinity/polylogue/commit/7b080c37de574492feab09093c56a66728776ec8))
* **stats:** use conversation_stats for aggregate message counts ([#1695](https://github.com/Sinity/polylogue/issues/1695)) ([2253e25](https://github.com/Sinity/polylogue/commit/2253e2549772601cda89a2c94674f5b8978b81fb))
* **storage:** cap WAL file via journal_size_limit + fail read-profile writes fast ([#1614](https://github.com/Sinity/polylogue/issues/1614) AC1) ([#1659](https://github.com/Sinity/polylogue/issues/1659)) ([a776e9d](https://github.com/Sinity/polylogue/commit/a776e9d7ae03c4bee294237b45a0b561c3adb652))
* **storage:** FTS content-sync for messages_fts + action_events_fts ([#1241](https://github.com/Sinity/polylogue/issues/1241)) ([#1344](https://github.com/Sinity/polylogue/issues/1344)) ([4e74662](https://github.com/Sinity/polylogue/commit/4e746622d1e59c1a0363ffa26b51d262039d659e))
* **storage:** keyset pagination in iter_messages, O(1) blob scan ([#1750](https://github.com/Sinity/polylogue/issues/1750)) ([e304de5](https://github.com/Sinity/polylogue/commit/e304de55a7474855114a865cadb56dc492aad4df))
* **storage:** rescue perf items from [#809](https://github.com/Sinity/polylogue/issues/809) fold ([#1314](https://github.com/Sinity/polylogue/issues/1314)) ([#1334](https://github.com/Sinity/polylogue/issues/1334)) ([39bfddd](https://github.com/Sinity/polylogue/commit/39bfddd3c67153d45fecd04d833838a54b258f70))
* **test:** route pytest temp dirs to tmpfs, add daily DB optimize to daemon ([dcfeec3](https://github.com/Sinity/polylogue/commit/dcfeec358023230242f4d33c3a7b38cfd0384519)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **tests:** beef up seeded_db and add COW writable variant ([58404d2](https://github.com/Sinity/polylogue/commit/58404d239f8808be429c1baff2c6463eb9df56b8)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **tests:** cap schema obligations Hypothesis examples instead of excluding ([ba8e4b0](https://github.com/Sinity/polylogue/commit/ba8e4b0ecdcbf7d44905044585a890ee504b692c)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **tests:** parallelize crashlessness tests and add --skip-slow to verify ([0584576](https://github.com/Sinity/polylogue/commit/0584576263f45dccbbbc1ce527ea68ef461afa7a)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **tests:** share seeded_db across xdist workers via file lock ([fe69100](https://github.com/Sinity/polylogue/commit/fe69100e3a978c8d2514cf4ca1395e115c2f7e57)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **verify:** check baseline existence before running tier-0 exercises ([9e58f25](https://github.com/Sinity/polylogue/commit/9e58f256237fb1cbf3ba6a8374550c44f380fc23)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **verify:** prefer dmypy for warm type-check (~0.5s vs ~13s) ([76a5d0f](https://github.com/Sinity/polylogue/commit/76a5d0f4d39463a1d8b5461d3ee10f174f4a8b7e)), closes [#1026](https://github.com/Sinity/polylogue/issues/1026)
* **verify:** structured pytest reports and worker tuning evidence ([#1026](https://github.com/Sinity/polylogue/issues/1026)) ([#1087](https://github.com/Sinity/polylogue/issues/1087)) ([b2f7490](https://github.com/Sinity/polylogue/commit/b2f74901a02ff0b056eed5d7d1dc1a0d004018fa)), closes [#998](https://github.com/Sinity/polylogue/issues/998)

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
  `verify-distribution-surface` gate now runs as a `distribution` job in
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
