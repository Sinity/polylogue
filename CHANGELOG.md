# Changelog

All notable user-visible changes to Polylogue are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

User-visible: anything an operator running `polylogue` would notice — new
flags, removed or renamed commands, output changes, breaking schema
migrations, security fixes. Internal refactors, test additions, and
documentation polish do not require an entry.

## [0.2.0](https://github.com/Sinity/polylogue/compare/v0.1.0...v0.2.0) (2026-06-18)


### Added

* add benchmark infra, fix FTS5 prefix, fix hooks health, add schema v21, lazy structlog ([#1742](https://github.com/Sinity/polylogue/issues/1742)) ([bcba9d1](https://github.com/Sinity/polylogue/commit/bcba9d118b03dd283f240f30556d987cd6b83066))
* add blob integrity probe ([#1480](https://github.com/Sinity/polylogue/issues/1480)) ([07c6def](https://github.com/Sinity/polylogue/commit/07c6def9ce0a1c3efbe88485fee790e9144d4680)), closes [#1231](https://github.com/Sinity/polylogue/issues/1231)
* add bounded embedding preflight ([#1531](https://github.com/Sinity/polylogue/issues/1531)) ([6c2b86d](https://github.com/Sinity/polylogue/commit/6c2b86d2c8c43664b9bdb1e42ce5447651f67ace))
* add shared query-expression compiler (CLI slice) ([#1861](https://github.com/Sinity/polylogue/issues/1861)) ([c854dfd](https://github.com/Sinity/polylogue/commit/c854dfd975ad65e9b0bce1d583427bf550984034))
* **api:** expose assertion claim reads ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#2086](https://github.com/Sinity/polylogue/issues/2086)) ([277dbbf](https://github.com/Sinity/polylogue/commit/277dbbff9553e1f7c9b1b4b2fb3cb4c990e8d623))
* **api:** expose embedding readiness helpers ([#1575](https://github.com/Sinity/polylogue/issues/1575)) ([723b808](https://github.com/Sinity/polylogue/commit/723b808fa7e337fba5c64be95164d9eb6917bc60)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **api:** expose read view profiles ([#1997](https://github.com/Sinity/polylogue/issues/1997)) ([#2023](https://github.com/Sinity/polylogue/issues/2023)) ([9b1f300](https://github.com/Sinity/polylogue/commit/9b1f300a747a7890e9e7569e412e652906a909b4)), closes [#1825](https://github.com/Sinity/polylogue/issues/1825)
* **api:** expose recovery digest transform ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1930](https://github.com/Sinity/polylogue/issues/1930)) ([7bfbfe5](https://github.com/Sinity/polylogue/commit/7bfbfe5167566b2db3ec3fc75b43d1f691564034))
* **assembly:** log unmatched/ambiguous history.jsonl rows ([#1700](https://github.com/Sinity/polylogue/issues/1700)) ([8d9195a](https://github.com/Sinity/polylogue/commit/8d9195a3c426f31eb184843c7939c2490a5acd52))
* **assertions:** export user-tier assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#2098](https://github.com/Sinity/polylogue/issues/2098)) ([7135ed3](https://github.com/Sinity/polylogue/commit/7135ed34fca4676e5aa4ce57b7e0cdf7e1c391a8))
* **backup:** support named archive profiles ([#1830](https://github.com/Sinity/polylogue/issues/1830)) ([#1963](https://github.com/Sinity/polylogue/issues/1963)) ([0f70c91](https://github.com/Sinity/polylogue/commit/0f70c91af62a079004f654ea858f29fae50d7cc6))
* **blackboard:** persistent agent-addressable notes ([#1697](https://github.com/Sinity/polylogue/issues/1697)) ([cd52d93](https://github.com/Sinity/polylogue/commit/cd52d93cd3524057b76987ce4dfaf572c2af2ff5))
* **cli:** add --json flag ([#1689](https://github.com/Sinity/polylogue/issues/1689)) ([7cd0b79](https://github.com/Sinity/polylogue/commit/7cd0b7929433a20c2c1bfd85256c099f6df4ddb0))
* **cli:** add context compose command ([#1494](https://github.com/Sinity/polylogue/issues/1494)) ([7755656](https://github.com/Sinity/polylogue/commit/77556562c13e3da68007b2f03fc61dec02ed4c5b))
* **cli:** add mark/analyze actions and delete cardinality guards ([#1872](https://github.com/Sinity/polylogue/issues/1872)) ([757ded9](https://github.com/Sinity/polylogue/commit/757ded9d15b3446031311494dd394d367bf5e806)), closes [#1814](https://github.com/Sinity/polylogue/issues/1814)
* **cli:** add polylogue commands listing ([#1681](https://github.com/Sinity/polylogue/issues/1681)) ([f5146ee](https://github.com/Sinity/polylogue/commit/f5146ee162f3b4ee1131bf4475b868c3e78ccf1e))
* **cli:** add polylogue paths command and defensive archive.db symlink ([#1627](https://github.com/Sinity/polylogue/issues/1627)) ([bb419eb](https://github.com/Sinity/polylogue/commit/bb419eb360ee38333ccf7f18aa146e58165eed94))
* **cli:** add polylogue recent command ([#1701](https://github.com/Sinity/polylogue/issues/1701)) ([d12fb95](https://github.com/Sinity/polylogue/commit/d12fb95529be2f9daf0024fbf831e239814e2556))
* **cli:** add public action contracts ([#1816](https://github.com/Sinity/polylogue/issues/1816)) ([#1941](https://github.com/Sinity/polylogue/issues/1941)) ([6e8eb9e](https://github.com/Sinity/polylogue/commit/6e8eb9ed0564351455b12901b5adc1d753d437d7))
* **cli:** add systematic --json output test and snapshot gate ([#1689](https://github.com/Sinity/polylogue/issues/1689)) ([06de763](https://github.com/Sinity/polylogue/commit/06de76340e2676e0415f88e05072d8785d2c6f75))
* **cli:** complete actions from command contracts ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2019](https://github.com/Sinity/polylogue/issues/2019)) ([a91557f](https://github.com/Sinity/polylogue/commit/a91557f7849fdacbc587c2ad92940ba1e93feda7))
* **cli:** complete query actions after then ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2026](https://github.com/Sinity/polylogue/issues/2026)) ([d84e37a](https://github.com/Sinity/polylogue/commit/d84e37a8db7d66a6a5c8652d884e10125a506779))
* **cli:** complete query count operators ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2029](https://github.com/Sinity/polylogue/issues/2029)) ([7b05904](https://github.com/Sinity/polylogue/commit/7b059049feac808a4c94717c63bdb0620a5f86bf)), closes [#2006](https://github.com/Sinity/polylogue/issues/2006)
* **cli:** complete query date operators ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2032](https://github.com/Sinity/polylogue/issues/2032)) ([3154698](https://github.com/Sinity/polylogue/commit/3154698191236ba4bc5cd2aea19b920bcabeb8c1))
* **cli:** complete query field values from descriptors ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2017](https://github.com/Sinity/polylogue/issues/2017)) ([7c963d4](https://github.com/Sinity/polylogue/commit/7c963d4c3e50abdd512e752754f1326d7de68b83))
* **cli:** complete query fields from grammar registry ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2016](https://github.com/Sinity/polylogue/issues/2016)) ([1b7c697](https://github.com/Sinity/polylogue/commit/1b7c697b6c07d28bdae089c83fdb502c12f7db7c))
* **cli:** complete read formats from view profiles ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2034](https://github.com/Sinity/polylogue/issues/2034)) ([ddf9694](https://github.com/Sinity/polylogue/commit/ddf969491cf9eddc0f1488431cf512e150d88773))
* **cli:** correlate sessions with git commits ([#1690](https://github.com/Sinity/polylogue/issues/1690)) ([292628e](https://github.com/Sinity/polylogue/commit/292628ebbad010266391e1ff3489fb8a8d86106b))
* **cli:** expose query completion metadata ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2037](https://github.com/Sinity/polylogue/issues/2037)) ([a615c85](https://github.com/Sinity/polylogue/commit/a615c852038fa65e6b2b7cfe5a657235b3efa09d))
* **cli:** expose query-backed select verb ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2036](https://github.com/Sinity/polylogue/issues/2036)) ([836504f](https://github.com/Sinity/polylogue/commit/836504f8b7f772d443092e7f4290dd27fe7d3fb8))
* **cli:** expose read view profiles ([#1997](https://github.com/Sinity/polylogue/issues/1997)) ([#2022](https://github.com/Sinity/polylogue/issues/2022)) ([837af0a](https://github.com/Sinity/polylogue/commit/837af0ae11dab370f9a24d463b37a5cea398723b))
* **cli:** expose recovery digest as a read view ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1929](https://github.com/Sinity/polylogue/issues/1929)) ([bad02a1](https://github.com/Sinity/polylogue/commit/bad02a168d0eefde469f53fbf8ad1fa41801e5d0))
* **cli:** expose recovery report presets ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1947](https://github.com/Sinity/polylogue/issues/1947)) ([8703692](https://github.com/Sinity/polylogue/commit/8703692926c55fc3ff18c8bf0a571e731ed2d0c3))
* **cli:** harmonize list output fields with MCP shape ([#1699](https://github.com/Sinity/polylogue/issues/1699)) ([d0173a4](https://github.com/Sinity/polylogue/commit/d0173a4971f8010b6358551db5051d63cdefec77))
* **cli:** schedule approved demo fixture import ([#1843](https://github.com/Sinity/polylogue/issues/1843)) ([#1935](https://github.com/Sinity/polylogue/issues/1935)) ([f38b639](https://github.com/Sinity/polylogue/commit/f38b639f68ce88db8556aa8100e593b1298e20ba))
* **cli:** unify read surface under single `read` verb ([#1813](https://github.com/Sinity/polylogue/issues/1813)) ([#1863](https://github.com/Sinity/polylogue/issues/1863)) ([4856248](https://github.com/Sinity/polylogue/commit/485624898a29e3c68a79e7890ee57c77487fbb64))
* **daemon:** /metrics Prometheus endpoint ([#1321](https://github.com/Sinity/polylogue/issues/1321)) ([#1407](https://github.com/Sinity/polylogue/issues/1407)) ([cae4602](https://github.com/Sinity/polylogue/commit/cae4602f2d6df55fee9280b06ab094bb48dc8d84))
* **daemon:** add OTLP HTTP receiver and rich Prometheus metrics ([#1321](https://github.com/Sinity/polylogue/issues/1321)) ([10e2a78](https://github.com/Sinity/polylogue/commit/10e2a78657d9fed7f8111bdb9ded42ef073bde12))
* **daemon:** classify HTTP route contracts ([#1847](https://github.com/Sinity/polylogue/issues/1847)) ([#2053](https://github.com/Sinity/polylogue/issues/2053)) ([9471bf2](https://github.com/Sinity/polylogue/commit/9471bf25f4e53f051d719b0dfd995aa1ddbafe3a))
* **daemon:** distinguish slow-but-progressing from stuck attempts ([#1246](https://github.com/Sinity/polylogue/issues/1246)) ([#1399](https://github.com/Sinity/polylogue/issues/1399)) ([2c0901e](https://github.com/Sinity/polylogue/commit/2c0901e2a85ca4acb649d5d5e4d6dd8e91bc630c))
* **daemon:** expose query completion metadata endpoint ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2039](https://github.com/Sinity/polylogue/issues/2039)) ([2f4d57d](https://github.com/Sinity/polylogue/commit/2f4d57d92405d83eb5595d505d761f49974f433b)), closes [#1846](https://github.com/Sinity/polylogue/issues/1846)
* **daemon:** expose read view profile metadata ([#1997](https://github.com/Sinity/polylogue/issues/1997)) ([#2041](https://github.com/Sinity/polylogue/issues/2041)) ([f2eb724](https://github.com/Sinity/polylogue/commit/f2eb7248613c90ba0d630825b0536c7571b3e2dd)), closes [#1846](https://github.com/Sinity/polylogue/issues/1846)
* **demo:** seed deterministic user overlays ([#1843](https://github.com/Sinity/polylogue/issues/1843)) ([#1975](https://github.com/Sinity/polylogue/issues/1975)) ([b61bb63](https://github.com/Sinity/polylogue/commit/b61bb63bd092ac70fac8e6a18543e0192a820aee))
* **devtools:** add `devtools test` focused runner so agents avoid raw pytest ([#1786](https://github.com/Sinity/polylogue/issues/1786)) ([15af28c](https://github.com/Sinity/polylogue/commit/15af28cde2d753bbd8566c9ea3d0551d1a5ed48d))
* **devtools:** add archive space report ([#1572](https://github.com/Sinity/polylogue/issues/1572)) ([46aa9f5](https://github.com/Sinity/polylogue/commit/46aa9f5c5d29764eb751043ca6ae005ef9fe7e7e)), closes [#1486](https://github.com/Sinity/polylogue/issues/1486) [#1552](https://github.com/Sinity/polylogue/issues/1552)
* **devtools:** add safe worktree gc ([#1222](https://github.com/Sinity/polylogue/issues/1222)) ([68ae333](https://github.com/Sinity/polylogue/commit/68ae3330b7313455c455bece8b78ce124d17fb6a))
* **devtools:** wire nightly scale workflow into benchmark-campaign index ([#1220](https://github.com/Sinity/polylogue/issues/1220)) ([ee60a7b](https://github.com/Sinity/polylogue/commit/ee60a7b143a3dad4aee2d68e9129726daae83eae))
* **dist:** publish polylogue-mcp and polylogue-hooks as separate PyPI packages ([#1309](https://github.com/Sinity/polylogue/issues/1309)) ([#1423](https://github.com/Sinity/polylogue/issues/1423)) ([873146f](https://github.com/Sinity/polylogue/commit/873146f47dd0f8114d52da1bf7d18fd7ff43c174))
* **embeddings:** bound catch-up windows by cost ([#1543](https://github.com/Sinity/polylogue/issues/1543)) ([3775bee](https://github.com/Sinity/polylogue/commit/3775bee4afb9cd2014e974a5056abbc73adafc1f))
* **embeddings:** emit preflight json plan ([#1561](https://github.com/Sinity/polylogue/issues/1561)) ([1c39aaf](https://github.com/Sinity/polylogue/commit/1c39aaf21eaa36818470feeb90929ab7118f0f92))
* **embeddings:** expose status next actions ([#1560](https://github.com/Sinity/polylogue/issues/1560)) ([a6af6bb](https://github.com/Sinity/polylogue/commit/a6af6bbf090890fb64bb0492d686ea73391851de)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* enable cloud-agent lane (Claude Web, Codex Cloud) ([#1676](https://github.com/Sinity/polylogue/issues/1676)) ([4760c0a](https://github.com/Sinity/polylogue/commit/4760c0aa27ec3f69d9c76826b02717ae1659f367))
* expose embedding catch-up metrics ([#1524](https://github.com/Sinity/polylogue/issues/1524)) ([f42f053](https://github.com/Sinity/polylogue/commit/f42f053d1aaf9cc479fbcb59631cc9903f615f82))
* **facets:** wire SQL-backed aggregators for repos, message_types, action_types, has_flags ([#1694](https://github.com/Sinity/polylogue/issues/1694)) ([b124e85](https://github.com/Sinity/polylogue/commit/b124e85f8bab2905064d45138449e232e7d668fb))
* insights section headers, mutation CI, concurrent search test ([862422a](https://github.com/Sinity/polylogue/commit/862422a4637281bcef63e382a6366032d44d8d56))
* **insights:** add deterministic recovery digest transform ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1912](https://github.com/Sinity/polylogue/issues/1912)) ([c34d900](https://github.com/Sinity/polylogue/commit/c34d9008dc22d074e3e3d3bed972566a17a6f013))
* **insights:** add OTLP span-to-work-event correlation ([#1686](https://github.com/Sinity/polylogue/issues/1686)) ([a7281b1](https://github.com/Sinity/polylogue/commit/a7281b13ad47f2427c4e95bf74dc277ca456778c))
* **insights:** add session latency profiles ([#1539](https://github.com/Sinity/polylogue/issues/1539)) ([4c504ed](https://github.com/Sinity/polylogue/commit/4c504eddb8c25753e096689d1d9248a91e40dbf1))
* **insights:** add session-commit correlation and issue/PR enrichment ([#1690](https://github.com/Sinity/polylogue/issues/1690)) ([50aec22](https://github.com/Sinity/polylogue/commit/50aec222d6180dd8374abf54000b577a4fe08108))
* **insights:** add tool active duration ([#1537](https://github.com/Sinity/polylogue/issues/1537)) ([bf40a87](https://github.com/Sinity/polylogue/commit/bf40a87b1acddf8b348f862c9fb35ffea02a1387)), closes [#1526](https://github.com/Sinity/polylogue/issues/1526)
* **insights:** add work-packet target refs ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2092](https://github.com/Sinity/polylogue/issues/2092)) ([3589ad6](https://github.com/Sinity/polylogue/commit/3589ad61dcbfc301571750274ad0ac46497eb675)), closes [#1845](https://github.com/Sinity/polylogue/issues/1845)
* **insights:** carry object refs in work packets ([#1845](https://github.com/Sinity/polylogue/issues/1845)) ([#2089](https://github.com/Sinity/polylogue/issues/2089)) ([9e6eb30](https://github.com/Sinity/polylogue/commit/9e6eb3012f5c4460ef67f5e23da3f59b107a68e0))
* **insights:** cite check runs in work packets ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2095](https://github.com/Sinity/polylogue/issues/2095)) ([0d1e316](https://github.com/Sinity/polylogue/commit/0d1e316d92b0061cffd198aceb11e9cac0982377)), closes [#1845](https://github.com/Sinity/polylogue/issues/1845)
* **insights:** classify recovery tool envelopes ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1916](https://github.com/Sinity/polylogue/issues/1916)) ([513daa9](https://github.com/Sinity/polylogue/commit/513daa99065675b0d5b629806ab86a2f89e16579))
* **insights:** classify session shape and terminal state ([#1538](https://github.com/Sinity/polylogue/issues/1538)) ([cb08bac](https://github.com/Sinity/polylogue/commit/cb08bacc3e408c72211f797b9f1732757b915b10))
* **insights:** detect /goal-driven sessions ([#1687](https://github.com/Sinity/polylogue/issues/1687)) ([8b5894b](https://github.com/Sinity/polylogue/commit/8b5894b99800cae62dbd73dcb2141baeba91cd80))
* **insights:** extract subagent reports into recovery digests ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1913](https://github.com/Sinity/polylogue/issues/1913)) ([2c3b130](https://github.com/Sinity/polylogue/commit/2c3b13044e15035ef39eeed0d7edae0e4328546d))
* **insights:** materialize logical session identity ([#1540](https://github.com/Sinity/polylogue/issues/1540)) ([02decd7](https://github.com/Sinity/polylogue/commit/02decd741306f6490f5ce114639659d2ff3f02f8)), closes [#866](https://github.com/Sinity/polylogue/issues/866)
* **insights:** parse run state into recovery digests ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1914](https://github.com/Sinity/polylogue/issues/1914)) ([19415a0](https://github.com/Sinity/polylogue/commit/19415a031667040bc51ebf210af3ea9ea842c533))
* **insights:** project run evidence into work packets ([#1882](https://github.com/Sinity/polylogue/issues/1882)) ([#2097](https://github.com/Sinity/polylogue/issues/2097)) ([72bbbcd](https://github.com/Sinity/polylogue/commit/72bbbcde9b1e6d3d8f99bef504cdaa275eb2f521))
* **insights:** rank resume candidates ([#1541](https://github.com/Sinity/polylogue/issues/1541)) ([e9a247e](https://github.com/Sinity/polylogue/commit/e9a247ea981d5cf576bebe1219d97cb4c31c3222))
* **live:** enrich paste evidence from UserPromptSubmit hook events ([#1704](https://github.com/Sinity/polylogue/issues/1704)) ([0e7be4d](https://github.com/Sinity/polylogue/commit/0e7be4d04f208a523f397eac2140cf7acf95eced))
* **maintenance:** add archive backup plan surface ([#1830](https://github.com/Sinity/polylogue/issues/1830)) ([#1952](https://github.com/Sinity/polylogue/issues/1952)) ([087456c](https://github.com/Sinity/polylogue/commit/087456c377f187ad4e3bc490e26dbb845421d067))
* **maintenance:** report blob GC dry-runs ([#1830](https://github.com/Sinity/polylogue/issues/1830)) ([#1985](https://github.com/Sinity/polylogue/issues/1985)) ([bb43e40](https://github.com/Sinity/polylogue/commit/bb43e40730c1474bca6220dd7e6a80020ef12e37))
* **mcp:** add aggregate_sessions tool ([#1691](https://github.com/Sinity/polylogue/issues/1691)) ([4a294e8](https://github.com/Sinity/polylogue/commit/4a294e8aad42281d555e72d4de41ed09fcc1cccc))
* **mcp:** add compare_sessions, find_similar_sessions, correlate_sessions tools ([#1691](https://github.com/Sinity/polylogue/issues/1691)) ([9120990](https://github.com/Sinity/polylogue/commit/9120990eb7bef7676e744cbf2c30cd852ab8b108))
* **mcp:** compose_context_preamble for SessionStart ([#1494](https://github.com/Sinity/polylogue/issues/1494)) ([2ccfc26](https://github.com/Sinity/polylogue/commit/2ccfc2626999a07176a0ff23385f0d4f4ce55fea))
* **mcp:** expose assertion claim reads ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#2088](https://github.com/Sinity/polylogue/issues/2088)) ([9a21b17](https://github.com/Sinity/polylogue/commit/9a21b176e73848805470e4322cd0457d4f9389e9)), closes [#1825](https://github.com/Sinity/polylogue/issues/1825)
* **mcp:** expose embedding component readiness ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1961](https://github.com/Sinity/polylogue/issues/1961)) ([3347ba4](https://github.com/Sinity/polylogue/commit/3347ba463db9a3715a02e820039d4a92cf5996d8))
* **mcp:** expose embedding preflight ([#1573](https://github.com/Sinity/polylogue/issues/1573)) ([68752a8](https://github.com/Sinity/polylogue/commit/68752a81a477e82486f093fedb46c97f4461c950)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **mcp:** expose embedding readiness status ([#1571](https://github.com/Sinity/polylogue/issues/1571)) ([dc302e1](https://github.com/Sinity/polylogue/commit/dc302e116c8c2152b67f90d8815622dfe6c9f034)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **mcp:** expose readiness component map ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1971](https://github.com/Sinity/polylogue/issues/1971)) ([b0d7c4f](https://github.com/Sinity/polylogue/commit/b0d7c4fbb82ae4daadeeecf52045b3679629c6e9))
* **mcp:** expose recovery reports ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2096](https://github.com/Sinity/polylogue/issues/2096)) ([0474e7d](https://github.com/Sinity/polylogue/commit/0474e7d2766b27ed822f96cfe70c8959557b82f0)), closes [#2006](https://github.com/Sinity/polylogue/issues/2006)
* **mcp:** expose recovery work packets ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2083](https://github.com/Sinity/polylogue/issues/2083)) ([38376c4](https://github.com/Sinity/polylogue/commit/38376c41f8a357752a5310a3cdfef956dd165d77))
* **mcp:** expose the agent blackboard via MCP tools ([#1801](https://github.com/Sinity/polylogue/issues/1801)) ([4faf166](https://github.com/Sinity/polylogue/commit/4faf1663f8a17c9cdf9b3655e119c6d7ff4e3124)), closes [#1697](https://github.com/Sinity/polylogue/issues/1697)
* **mcp:** forward EmbeddingRetrievalNotReadyError message verbatim ([#1503](https://github.com/Sinity/polylogue/issues/1503) AC4) ([#1663](https://github.com/Sinity/polylogue/issues/1663)) ([5e23d5c](https://github.com/Sinity/polylogue/commit/5e23d5cd09e0d53cd897e9c7501dca8a89c95dbb))
* **mcp:** support query-scoped facets ([#1569](https://github.com/Sinity/polylogue/issues/1569)) ([d575ede](https://github.com/Sinity/polylogue/commit/d575ededaaf996c4665fe9c19aa11d9fa9270fcb)), closes [#873](https://github.com/Sinity/polylogue/issues/873)
* **nix:** unified settings lib + discoverSources, xdg config, CLI flags ([#1685](https://github.com/Sinity/polylogue/issues/1685)) ([22e337a](https://github.com/Sinity/polylogue/commit/22e337a911b5adba6d73a80de4026ea00407e5ba))
* **parsers:** wire Claude Code history.jsonl paste evidence into has_paste ([#1651](https://github.com/Sinity/polylogue/issues/1651)) ([2872c65](https://github.com/Sinity/polylogue/commit/2872c6532b6b3fa418acb0e8518bdce570844c84))
* persist embedding catch-up progress ([#1523](https://github.com/Sinity/polylogue/issues/1523)) ([b6b4737](https://github.com/Sinity/polylogue/commit/b6b47374677753d9bdeef1ee31de82c5ce6c1931))
* **query:** add continue action ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2094](https://github.com/Sinity/polylogue/issues/2094)) ([e15ef4b](https://github.com/Sinity/polylogue/commit/e15ef4ba232bb9c55da4069c8c69346b957af209)), closes [#1807](https://github.com/Sinity/polylogue/issues/1807)
* **query:** describe structural DSL fields ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2093](https://github.com/Sinity/polylogue/issues/2093)) ([7fae014](https://github.com/Sinity/polylogue/commit/7fae0146dad0a0a6847a0acf9ef01b2827da832d))
* **query:** execute block structural predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2024](https://github.com/Sinity/polylogue/issues/2024)) ([75cf1c4](https://github.com/Sinity/polylogue/commit/75cf1c48e2f6064d58d69761337b3294b6015915))
* **query:** execute Boolean FTS predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2012](https://github.com/Sinity/polylogue/issues/2012)) ([a30dc78](https://github.com/Sinity/polylogue/commit/a30dc789fe910f634c7222c6ee9ca00a388b4b48))
* **query:** execute Boolean session predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2010](https://github.com/Sinity/polylogue/issues/2010)) ([5fd1652](https://github.com/Sinity/polylogue/commit/5fd16520a879227c7ce2caa2e5f0ae86ff2485e9))
* **query:** execute lineage DSL predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2014](https://github.com/Sinity/polylogue/issues/2014)) ([cb641e6](https://github.com/Sinity/polylogue/commit/cb641e631614f839b643d4010fdd0b83ce50f791))
* **query:** execute near:id:&lt;ref&gt; session-seeded similarity ([#1842](https://github.com/Sinity/polylogue/issues/1842)) ([#1904](https://github.com/Sinity/polylogue/issues/1904)) ([12fac7d](https://github.com/Sinity/polylogue/commit/12fac7d805f0254066d873b34b9514011731db1a))
* **query:** execute semantic DSL predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2013](https://github.com/Sinity/polylogue/issues/2013)) ([c96a046](https://github.com/Sinity/polylogue/commit/c96a046cb0e846b49420f3277dc65f645350d1b4))
* **query:** execute structural and sequence predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2011](https://github.com/Sinity/polylogue/issues/2011)) ([def884e](https://github.com/Sinity/polylogue/commit/def884ef305bdaf340b0f79717249e80194c7585))
* **query:** execute unit-scoped where predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2046](https://github.com/Sinity/polylogue/issues/2046)) ([9323f3d](https://github.com/Sinity/polylogue/commit/9323f3d677b0863ff117977cca9df263d84f4760))
* **query:** explain terminal unit sources ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2054](https://github.com/Sinity/polylogue/issues/2054)) ([94ca249](https://github.com/Sinity/polylogue/commit/94ca249ce838c31a1c7aed036774f849b8ca56e2))
* **query:** expose assertions as terminal query units ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2079](https://github.com/Sinity/polylogue/issues/2079)) ([b855726](https://github.com/Sinity/polylogue/commit/b855726092353b39f7f5f92c1f1894dc782decbf)), closes [#1883](https://github.com/Sinity/polylogue/issues/1883)
* **query:** expose DSL explain command ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2015](https://github.com/Sinity/polylogue/issues/2015)) ([2ae9e4d](https://github.com/Sinity/polylogue/commit/2ae9e4d308b43db152f57127dd92441ac6738158))
* **query:** expose query explain execution legs ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2027](https://github.com/Sinity/polylogue/issues/2027)) ([75759ef](https://github.com/Sinity/polylogue/commit/75759efbcbf4978645c205a8c2ce73b048ae0c68))
* **query:** expose query explain through API and MCP ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2030](https://github.com/Sinity/polylogue/issues/2030)) ([7038099](https://github.com/Sinity/polylogue/commit/70380995104e91664882c70bf95924cbd7961f89))
* **query:** expose structural grammar completion metadata ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2025](https://github.com/Sinity/polylogue/issues/2025)) ([2d41b05](https://github.com/Sinity/polylogue/commit/2d41b0571c55e2a433c8088ff828de1a61ffc80a))
* **query:** expose structured query lowering plans ([#2059](https://github.com/Sinity/polylogue/issues/2059)) ([c777771](https://github.com/Sinity/polylogue/commit/c777771716ba034018f75733393d36f8f36832eb))
* **query:** near:id:&lt;ref&gt; session-seeded similarity plumbing ([#1842](https://github.com/Sinity/polylogue/issues/1842)) ([#1899](https://github.com/Sinity/polylogue/issues/1899)) ([cf904dc](https://github.com/Sinity/polylogue/commit/cf904dc0223659b65b6027d581324ad9872655f0))
* **query:** parse query expressions with Lark ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2009](https://github.com/Sinity/polylogue/issues/2009)) ([0a4fbfd](https://github.com/Sinity/polylogue/commit/0a4fbfdbda470b396068c0c7a7ff4d3b71cdc512))
* **query:** parse readable count comparisons ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2028](https://github.com/Sinity/polylogue/issues/2028)) ([29a11d2](https://github.com/Sinity/polylogue/commit/29a11d2636f5a31e36b204717b8811c1929ef132))
* **query:** parse readable date comparisons ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2031](https://github.com/Sinity/polylogue/issues/2031)) ([b919e28](https://github.com/Sinity/polylogue/commit/b919e28a82d9db4d5e58b4ad3eddbf96e4704b2e))
* **query:** return terminal unit rows for source queries ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2049](https://github.com/Sinity/polylogue/issues/2049)) ([975336c](https://github.com/Sinity/polylogue/commit/975336c6a5e089a2f85fc995d2ba012b0da48437)), closes [#1825](https://github.com/Sinity/polylogue/issues/1825)
* **query:** scope unit predicates by owning session ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2076](https://github.com/Sinity/polylogue/issues/2076)) ([4b5601e](https://github.com/Sinity/polylogue/commit/4b5601e9c40e54083f76eb0dc77c19cd39e5c97d))
* **query:** share completion metadata across API and MCP ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2038](https://github.com/Sinity/polylogue/issues/2038)) ([ce34edd](https://github.com/Sinity/polylogue/commit/ce34edddc33814f41c08b37eb3b76716c8718a70)), closes [#1825](https://github.com/Sinity/polylogue/issues/1825)
* **query:** share unit-row session filters ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2056](https://github.com/Sinity/polylogue/issues/2056)) ([9233a45](https://github.com/Sinity/polylogue/commit/9233a451733736641e420793925e5a5d0ec02a04)), closes [#2006](https://github.com/Sinity/polylogue/issues/2006)
* **query:** suggest close field names in DSL errors ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2018](https://github.com/Sinity/polylogue/issues/2018)) ([6ccd785](https://github.com/Sinity/polylogue/commit/6ccd785a3d6c34e7c3df3a293f25316fd3610a6f))
* **reader:** add message anchors, folds, density toggle, keyboard nav, topology inspector ([#1518](https://github.com/Sinity/polylogue/issues/1518)) ([3b4a777](https://github.com/Sinity/polylogue/commit/3b4a777d928cad990bddb4be866603012e40404a))
* **readiness:** add capability readiness contract ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1938](https://github.com/Sinity/polylogue/issues/1938)) ([11635b0](https://github.com/Sinity/polylogue/commit/11635b0a70bac9517305463de23d9ca1d6125cf4))
* **read:** profile executable session views ([#1997](https://github.com/Sinity/polylogue/issues/1997)) ([#2021](https://github.com/Sinity/polylogue/issues/2021)) ([8febe37](https://github.com/Sinity/polylogue/commit/8febe37d003f29868c066f88e8079a126d342c36)), closes [#1844](https://github.com/Sinity/polylogue/issues/1844)
* record daemon embedding progress ([#1532](https://github.com/Sinity/polylogue/issues/1532)) ([f727678](https://github.com/Sinity/polylogue/commit/f72767871371acc6e4bb7c43e6f37edbc7e4f441)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **recovery:** enrich work-packet evidence details ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2084](https://github.com/Sinity/polylogue/issues/2084)) ([8a21afb](https://github.com/Sinity/polylogue/commit/8a21afb752d4bdb9b00977987de2267551fdb6e8)), closes [#1883](https://github.com/Sinity/polylogue/issues/1883)
* **recovery:** mark work-packet evidence support ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2082](https://github.com/Sinity/polylogue/issues/2082)) ([2551fde](https://github.com/Sinity/polylogue/commit/2551fde9250757dbf6131b1aab03187c14498b33))
* **recovery:** resolve subagent child links in digests ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#2050](https://github.com/Sinity/polylogue/issues/2050)) ([f5c8827](https://github.com/Sinity/polylogue/commit/f5c882787e9843c33fcdd182914d32fe5f0366e8))
* **scenarios:** define demo corpus specs ([#1843](https://github.com/Sinity/polylogue/issues/1843)) ([#1934](https://github.com/Sinity/polylogue/issues/1934)) ([87a4592](https://github.com/Sinity/polylogue/commit/87a4592ed9a1f278b709ea25f7fe6378302cd3af))
* **schema:** add paste_boundary_state to messages ([#1706](https://github.com/Sinity/polylogue/issues/1706)) ([a06aee0](https://github.com/Sinity/polylogue/commit/a06aee090faf72531aff381d9ae89eb950f08946))
* **schema:** cwd/repo identity table ([#1253](https://github.com/Sinity/polylogue/issues/1253)) ([#1402](https://github.com/Sinity/polylogue/issues/1402)) ([931761d](https://github.com/Sinity/polylogue/commit/931761d67c4be4812f362a6d905cf7d180fa6c2e))
* **search:** expose lane evidence on hits ([#1562](https://github.com/Sinity/polylogue/issues/1562)) ([0dbc2e6](https://github.com/Sinity/polylogue/commit/0dbc2e60c8d2786b9abe875300de7a91dd6792b2))
* session analysis primitives, commit correlation, OTLP spans, verifiability infrastructure ([#1738](https://github.com/Sinity/polylogue/issues/1738)) ([a3f5f6f](https://github.com/Sinity/polylogue/commit/a3f5f6ffbd0e0a7cbae236afd074d16c7c745718))
* **status:** expose archive surface readiness components ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1958](https://github.com/Sinity/polylogue/issues/1958)) ([99bee67](https://github.com/Sinity/polylogue/commit/99bee678f5ddd05324aa9a55541760a112f6af8a))
* **status:** expose assertion and transform readiness ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1976](https://github.com/Sinity/polylogue/issues/1976)) ([cbcc2e1](https://github.com/Sinity/polylogue/commit/cbcc2e1ee9c9cbcd3fbe26bc7836e064b4034c63))
* **status:** expose assertion overlay storage audit ([#2062](https://github.com/Sinity/polylogue/issues/2062)) ([708d6fc](https://github.com/Sinity/polylogue/commit/708d6fc79b7e83fb6586076ca5824501c3034dd8))
* **status:** expose daemon component readiness map ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1960](https://github.com/Sinity/polylogue/issues/1960)) ([57c1024](https://github.com/Sinity/polylogue/commit/57c1024b92168dd6a19389886414d8099e43f0b4))
* **status:** expose embedding component readiness ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1949](https://github.com/Sinity/polylogue/issues/1949)) ([9911af0](https://github.com/Sinity/polylogue/commit/9911af092851f6f36d0682225624f10dbc4b6976))
* **status:** expose search component readiness ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1956](https://github.com/Sinity/polylogue/issues/1956)) ([7f48d9f](https://github.com/Sinity/polylogue/commit/7f48d9f084ad4065292fe88e95472c0e2f0fa1a2))
* **storage:** mark deleted overlays in assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1928](https://github.com/Sinity/polylogue/issues/1928)) ([842eea9](https://github.com/Sinity/polylogue/commit/842eea9b9e29f8659129e7005dfd66b4f0107c5e))
* **storage:** mirror recall packs and workspaces into assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1915](https://github.com/Sinity/polylogue/issues/1915)) ([f69c5fa](https://github.com/Sinity/polylogue/commit/f69c5fa009c31e59903290352f54d2cb39f44941))
* **storage:** mirror transform candidates as assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1973](https://github.com/Sinity/polylogue/issues/1973)) ([6800efc](https://github.com/Sinity/polylogue/commit/6800efc095c3cd48b2e854845edb22d1b8e5f758))
* **storage:** preserve blackboard assertion metadata ([#1839](https://github.com/Sinity/polylogue/issues/1839)) ([#1932](https://github.com/Sinity/polylogue/issues/1932)) ([a32e9ef](https://github.com/Sinity/polylogue/commit/a32e9ef7d2c8b5288cb6d44ec8695453bb0ef38a))
* **storage:** read user overlays through assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1967](https://github.com/Sinity/polylogue/issues/1967)) ([beddd91](https://github.com/Sinity/polylogue/commit/beddd91cef683b57cf0ca9b1f1c7549b98b106ed))
* **storage:** route saved views through assertion adapters ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1921](https://github.com/Sinity/polylogue/issues/1921)) ([266cd89](https://github.com/Sinity/polylogue/commit/266cd899b00f34236cea762f1d80fa5ef71debe2))
* **storage:** unified assertions substrate table ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1902](https://github.com/Sinity/polylogue/issues/1902)) ([a11d988](https://github.com/Sinity/polylogue/commit/a11d9885f439cd9b48d3ffb7302cea6312294c97))
* **storage:** write-through adapters mirror overlays into assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1905](https://github.com/Sinity/polylogue/issues/1905)) ([00aad0b](https://github.com/Sinity/polylogue/commit/00aad0b838c59db809fbfd0edacc35be1ffda2f7))
* **surfaces:** add ContextPreamble envelope for SessionStart injection ([#1703](https://github.com/Sinity/polylogue/issues/1703)) ([85ad6d4](https://github.com/Sinity/polylogue/commit/85ad6d42a67158ab13f9587de6b5b272ab743654))
* **surfaces:** add paste_boundary_state to MessageRenderEnvelope ([#1655](https://github.com/Sinity/polylogue/issues/1655) follow-up) ([8f199e3](https://github.com/Sinity/polylogue/commit/8f199e3617205448255098553c07a02a4737823d))
* **surfaces:** closed vocabulary for reader action availability ([#1488](https://github.com/Sinity/polylogue/issues/1488)) ([#1664](https://github.com/Sinity/polylogue/issues/1664)) ([d257040](https://github.com/Sinity/polylogue/commit/d2570408fe38deb7ba9e9949a31b4e45d3c9473b))
* **surfaces:** MessageRenderEnvelope — additive fields for unified reader payload ([#1487](https://github.com/Sinity/polylogue/issues/1487)) ([#1665](https://github.com/Sinity/polylogue/issues/1665)) ([ea432b0](https://github.com/Sinity/polylogue/commit/ea432b00613cada3e09b2448b10b9deb608943ab))
* **telemetry:** add otlp_spans table ([#1686](https://github.com/Sinity/polylogue/issues/1686)) ([a6cdbaf](https://github.com/Sinity/polylogue/commit/a6cdbafbe06460a8b9d3e533901bc3097a06d9c4))
* **topology:** cycle rejection and quarantine ([#1260](https://github.com/Sinity/polylogue/issues/1260)) ([#1395](https://github.com/Sinity/polylogue/issues/1395)) ([bf715ca](https://github.com/Sinity/polylogue/commit/bf715ca9eb95795a72c8d063c77ebf936b3f7b4a))
* **topology:** typed read model API — ([14791ca](https://github.com/Sinity/polylogue/commit/14791ca8109d4447c6b6e8a281b507d0cc41de41))
* **topology:** typed read model API — ancestors/descendants/siblings/thread ([#1261](https://github.com/Sinity/polylogue/issues/1261)) ([#1391](https://github.com/Sinity/polylogue/issues/1391)) ([14791ca](https://github.com/Sinity/polylogue/commit/14791ca8109d4447c6b6e8a281b507d0cc41de41))
* **transforms:** add recovery forensic index ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1939](https://github.com/Sinity/polylogue/issues/1939)) ([ebd3165](https://github.com/Sinity/polylogue/commit/ebd31658f640132bd27afc3a30dd67b85d475bd0))
* **transforms:** add recovery report presets ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1945](https://github.com/Sinity/polylogue/issues/1945)) ([264e86b](https://github.com/Sinity/polylogue/commit/264e86b5a3e04ee339a62af96f007e6a4334baf4))
* **transforms:** expose subagent spawn refs ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1974](https://github.com/Sinity/polylogue/issues/1974)) ([43e5f2a](https://github.com/Sinity/polylogue/commit/43e5f2addd57caafcc52e574df9d3de0f64f94b7))
* **transforms:** extract GitHub event failures ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1931](https://github.com/Sinity/polylogue/issues/1931)) ([afb1787](https://github.com/Sinity/polylogue/commit/afb178740e05953a2752ef54dfca1c500bf5e084))
* **verifiability:** add MCP tool discovery, coverage lint, property tests, idempotency gate ([#1722](https://github.com/Sinity/polylogue/issues/1722)) ([da3ec3e](https://github.com/Sinity/polylogue/commit/da3ec3e3204d4bec3ff8e944a36b7524f7f31b2e))
* **web:** consume component readiness in status strip ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1972](https://github.com/Sinity/polylogue/issues/1972)) ([3f942ad](https://github.com/Sinity/polylogue/commit/3f942adb68b2678c693851c6ab509dba05b13301))
* wire query-expression compiler into MCP search and /api/sessions ([#1860](https://github.com/Sinity/polylogue/issues/1860)) ([#1869](https://github.com/Sinity/polylogue/issues/1869)) ([108f98c](https://github.com/Sinity/polylogue/commit/108f98c1db9a7af8b4582cb611ef57a093c461a2))


### Fixed

* **antigravity:** flag fragmented brain-metadata sessions degraded ([#1856](https://github.com/Sinity/polylogue/issues/1856)) ([83efea0](https://github.com/Sinity/polylogue/commit/83efea0f2bbf0da245411f9f24c87daf5679f1e3)), closes [#1764](https://github.com/Sinity/polylogue/issues/1764)
* audit nits — test-helper connection leak + cloud-agent --no-api docs ([#1874](https://github.com/Sinity/polylogue/issues/1874)) ([984c7f6](https://github.com/Sinity/polylogue/commit/984c7f6656c9c25269467af267900d7887f320fe))
* **ci:** restore green test gate (daemon deadlock + masked failures) ([#1877](https://github.com/Sinity/polylogue/issues/1877)) ([2ae26e8](https://github.com/Sinity/polylogue/commit/2ae26e85fce0798c8a1a9cc6c1274429ff68a1ae))
* **ci:** revive dead mutation-testing job and enforce kill-rate thresholds ([#1800](https://github.com/Sinity/polylogue/issues/1800)) ([d16257f](https://github.com/Sinity/polylogue/commit/d16257fb6d8813b7e5d4e133eba9052bfc777a45)), closes [#1733](https://github.com/Sinity/polylogue/issues/1733)
* **cli:** accept cursor option in root callback ([#1426](https://github.com/Sinity/polylogue/issues/1426)) ([0856e4f](https://github.com/Sinity/polylogue/commit/0856e4f58ab1ff2813fc20fd9731f98f04a6d160))
* **cli:** add `--json` alias to `polylogue status` to match siblings ([#1612](https://github.com/Sinity/polylogue/issues/1612)) ([#1619](https://github.com/Sinity/polylogue/issues/1619)) ([4870b56](https://github.com/Sinity/polylogue/commit/4870b564687762c7371a7fbcf9d28df3f3853c11))
* **cli:** allow status on large archives ([#1427](https://github.com/Sinity/polylogue/issues/1427)) ([fe35852](https://github.com/Sinity/polylogue/commit/fe358523f6ed2231e78cb9abc6a1d18921618e7b))
* **cli:** delete --all/--yes uses pre-resolved IDs; resolve uses compiled spec ([6a10759](https://github.com/Sinity/polylogue/commit/6a1075995e4307a653c6b132f83ab94aecb32f8f))
* **cli:** enforce read-view format profiles ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2099](https://github.com/Sinity/polylogue/issues/2099)) ([371657a](https://github.com/Sinity/polylogue/commit/371657aee64c2269491e7f3c3cb6e6e3107e1483))
* **cli:** give honest next-step guidance after polylogue ingest ([#1693](https://github.com/Sinity/polylogue/issues/1693)) ([ae70895](https://github.com/Sinity/polylogue/commit/ae708951ca5a72365cb6ec68229008b104151081))
* **cli:** keep status responsive under ingest ([#1438](https://github.com/Sinity/polylogue/issues/1438)) ([9974bfb](https://github.com/Sinity/polylogue/commit/9974bfbaeb1c57da25f075e5feeaad7a9686dcdc))
* **cli:** messages --format ndjson emits JSON-per-line, not markdown ([#1818](https://github.com/Sinity/polylogue/issues/1818)) ([#1897](https://github.com/Sinity/polylogue/issues/1897)) ([65e22ff](https://github.com/Sinity/polylogue/commit/65e22ff57ec954d7c919f0b8d893fe81251f5880))
* **cli:** reject --sample with mutating verbs instead of silently ignoring it ([#1892](https://github.com/Sinity/polylogue/issues/1892)) ([9725856](https://github.com/Sinity/polylogue/commit/97258561b1cc5fc7cc6d45493ae63afa43dd0c92))
* **cli:** rename read --view conversation to transcript (vocab policy) ([#1867](https://github.com/Sinity/polylogue/issues/1867)) ([a7d0cc8](https://github.com/Sinity/polylogue/commit/a7d0cc8d9d31686d238f514dd0809b669a646e9d))
* **cli:** reset --database preserves irreplaceable user.db by default ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1896](https://github.com/Sinity/polylogue/issues/1896)) ([a7f7ce5](https://github.com/Sinity/polylogue/commit/a7f7ce5727bea720031459ee4fafb415fe5d2e2a))
* **cli:** resolve --latest for `messages` and `raw` verbs ([#1626](https://github.com/Sinity/polylogue/issues/1626)) ([#1637](https://github.com/Sinity/polylogue/issues/1637)) ([00e415e](https://github.com/Sinity/polylogue/commit/00e415e0aee759678f7b179116abc1051fbbd3de))
* **cli:** resolve --latest for top-level export/neighbors/diagnostics turns ([#1642](https://github.com/Sinity/polylogue/issues/1642)) ([#1649](https://github.com/Sinity/polylogue/issues/1649)) ([8f4941b](https://github.com/Sinity/polylogue/commit/8f4941ba4493fe43af5fff7ad2bc88d34e109f04))
* **cli:** resolve stale XFAIL in tags command ([#1012](https://github.com/Sinity/polylogue/issues/1012)) ([9e7dd57](https://github.com/Sinity/polylogue/commit/9e7dd57e50459c43bbeb5ecb49b88c3efc82e2dc))
* **cli:** route recovery reads through archive facade ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#2057](https://github.com/Sinity/polylogue/issues/2057)) ([0cea5c3](https://github.com/Sinity/polylogue/commit/0cea5c37c79e4be7e00b09f85be6110aa44bd1b3))
* **cli:** show embedding readiness in status fallback ([#1548](https://github.com/Sinity/polylogue/issues/1548)) ([8c0abd3](https://github.com/Sinity/polylogue/commit/8c0abd35ee28adfd24be1462b18b14660357c763))
* **cli:** wrap list/summary JSON output in paginated envelope ([#1618](https://github.com/Sinity/polylogue/issues/1618)) ([#1670](https://github.com/Sinity/polylogue/issues/1670)) ([4b54d77](https://github.com/Sinity/polylogue/commit/4b54d771ef6188a91babff9a8bee356d64b5c31f))
* **config:** redact secrets and harden write paths in sources ([#1748](https://github.com/Sinity/polylogue/issues/1748)) ([8e6f4c7](https://github.com/Sinity/polylogue/commit/8e6f4c74a562ef728401b34d71a540f3ad87f736))
* **core:** ensure optional_datetime always returns UTC-aware datetimes ([9735a6f](https://github.com/Sinity/polylogue/commit/9735a6fd49957c0414ce7bb4b2ec929e7f8524e3))
* **daemon:** /api/sessions contains not compiled; session_id filter wired ([c0680b4](https://github.com/Sinity/polylogue/commit/c0680b4e1d4afdba4e0fd90065dc26f0f6d0081d))
* **daemon/http:** suppress BrokenPipeError on client disconnect ([#1682](https://github.com/Sinity/polylogue/issues/1682)) ([7f417d0](https://github.com/Sinity/polylogue/commit/7f417d0ed7e68f8bdf6f36b2554fa6e53f46d947))
* **daemon:** apply canonical SQLite profile in ArchiveStore ([#1806](https://github.com/Sinity/polylogue/issues/1806)) ([1b271e0](https://github.com/Sinity/polylogue/commit/1b271e0ca4bd5f05978bff1a5ecc1bfdd3cdf339))
* **daemon:** apply message projection flags in reader API ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2043](https://github.com/Sinity/polylogue/issues/2043)) ([9f23294](https://github.com/Sinity/polylogue/commit/9f23294828e7319430dc7c0b6d499a621e844380))
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
* **daemon:** preserve inbox import suffixes ([#1843](https://github.com/Sinity/polylogue/issues/1843)) ([#1953](https://github.com/Sinity/polylogue/issues/1953)) ([23d9ae7](https://github.com/Sinity/polylogue/commit/23d9ae77c4bdd1cb697939f9ad3836877d08e677))
* **daemon:** reduce catch-up replay amplification ([#1459](https://github.com/Sinity/polylogue/issues/1459)) ([f5688cb](https://github.com/Sinity/polylogue/commit/f5688cb149e94763852411a5a8b2eea89cabbc05))
* **daemon:** reduce large ingest FTS amplification ([#1453](https://github.com/Sinity/polylogue/issues/1453)) ([6c826cf](https://github.com/Sinity/polylogue/commit/6c826cf289fcab828e28becc7f7001022c3f84bd))
* **daemon:** remove broad startup maintenance scans ([#1464](https://github.com/Sinity/polylogue/issues/1464)) ([2dc505c](https://github.com/Sinity/polylogue/commit/2dc505ca60f156147c9a923383fc87e10c69eb2e)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** repair fresh-archive ingest dropping sessions + false 0% FTS ([#1804](https://github.com/Sinity/polylogue/issues/1804)) ([09131da](https://github.com/Sinity/polylogue/commit/09131dad0499fb97254d5459528d333e1e2a791c))
* **daemon:** repair FTS overcount drift ([#1432](https://github.com/Sinity/polylogue/issues/1432)) ([c88d3be](https://github.com/Sinity/polylogue/commit/c88d3be5216956953c1520f112b4a7553b05780b))
* **daemon:** report FTS ledger counts ([#1520](https://github.com/Sinity/polylogue/issues/1520)) ([d5a5b71](https://github.com/Sinity/polylogue/commit/d5a5b71f8cb8db21715356547a7b5e8fb70a1a02)), closes [#1515](https://github.com/Sinity/polylogue/issues/1515)
* **daemon:** report WAL checkpoint blockers ([#1456](https://github.com/Sinity/polylogue/issues/1456)) ([23de93f](https://github.com/Sinity/polylogue/commit/23de93f9429e2aef461eb61d7810c35f303615a0))
* **daemon:** requeue live ingest when archive is busy ([#1454](https://github.com/Sinity/polylogue/issues/1454)) ([38a268e](https://github.com/Sinity/polylogue/commit/38a268e5c0676217584491e5f479e812bb503768))
* **daemon:** restore all active FTS triggers ([#1455](https://github.com/Sinity/polylogue/issues/1455)) ([2ba4b07](https://github.com/Sinity/polylogue/commit/2ba4b079eff474118617c77a30da3a678bd779e4)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** restore Drive source catch-up ([#1994](https://github.com/Sinity/polylogue/issues/1994)) ([f3bbd26](https://github.com/Sinity/polylogue/commit/f3bbd26fc409bbe11691b34c61fec59c0a961fb1))
* **daemon:** skip redundant write FTS repairs ([#1472](https://github.com/Sinity/polylogue/issues/1472)) ([dae61fb](https://github.com/Sinity/polylogue/commit/dae61fba9890119fde35348fb17eb3e43d3fa9a7))
* **daemon:** stop schema-blocked background work ([#1466](https://github.com/Sinity/polylogue/issues/1466)) ([fd93d0d](https://github.com/Sinity/polylogue/commit/fd93d0d3081c025251b0b1afe7368eb3907d6494))
* **daemon:** target startup FTS gap repair ([#1441](https://github.com/Sinity/polylogue/issues/1441)) ([81024be](https://github.com/Sinity/polylogue/commit/81024be24834d2c97a04de36d464fb6c348a7959)), closes [#1439](https://github.com/Sinity/polylogue/issues/1439)
* **daemon:** tolerate locked cursor bookkeeping ([#1516](https://github.com/Sinity/polylogue/issues/1516)) ([2c245d6](https://github.com/Sinity/polylogue/commit/2c245d6e43a52fb92bb293cedf5f0d96478228df)), closes [#1515](https://github.com/Sinity/polylogue/issues/1515)
* **daemon:** treat convergence locks as archive busy ([#1519](https://github.com/Sinity/polylogue/issues/1519)) ([d99baf2](https://github.com/Sinity/polylogue/commit/d99baf2ed7479346ccf5826713a3b1641c7c02d3)), closes [#1517](https://github.com/Sinity/polylogue/issues/1517)
* **daemon:** treat fresh-init bootstrap-in-flight as 'nothing to repair' ([#1603](https://github.com/Sinity/polylogue/issues/1603)) ([#1609](https://github.com/Sinity/polylogue/issues/1609)) ([d373810](https://github.com/Sinity/polylogue/commit/d373810b0cf858e3b40281d73360f99b649a2746))
* **daemon:** trust ready FTS triggers on live appends ([#1469](https://github.com/Sinity/polylogue/issues/1469)) ([1478cc1](https://github.com/Sinity/polylogue/commit/1478cc1842a2435ee72a284a77ec6062a176ffc7))
* **daemon:** write fts_freshness_state ready rows after startup readiness pass ([#1628](https://github.com/Sinity/polylogue/issues/1628)) ([#1650](https://github.com/Sinity/polylogue/issues/1650)) ([921c191](https://github.com/Sinity/polylogue/commit/921c191082269c86ac5a541668e08aaf36a6edb2))
* delete full-set cardinality + Codex P2 read-surface findings ([#1873](https://github.com/Sinity/polylogue/issues/1873)) ([#1876](https://github.com/Sinity/polylogue/issues/1876)) ([0eb2300](https://github.com/Sinity/polylogue/commit/0eb2300bbbd665be1015b98469f8fe1c891d5452))
* **devtools:** clear stale pytest reports on verify timeout ([#1807](https://github.com/Sinity/polylogue/issues/1807)) ([#2090](https://github.com/Sinity/polylogue/issues/2090)) ([4f37816](https://github.com/Sinity/polylogue/commit/4f37816892b2f2b084c7f8a930876e8ee1dd2a8c))
* **devtools:** close leaked read connection in archive-space-report ([#1774](https://github.com/Sinity/polylogue/issues/1774)) ([fb6b274](https://github.com/Sinity/polylogue/commit/fb6b2745c74541088043373f7b164f4f08e1d182))
* **devtools:** expose live pytest verify progress ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2074](https://github.com/Sinity/polylogue/issues/2074)) ([28f7903](https://github.com/Sinity/polylogue/commit/28f79030e033bb1ff00d9672eb7a4f5ef0f4d596))
* **devtools:** fill [#1737](https://github.com/Sinity/polylogue/issues/1737) gaps — PyYAML parsers, stale spine docs ([3604177](https://github.com/Sinity/polylogue/commit/3604177556322b6591df1a4cf1b2ef4651d6e935))
* **devtools:** preserve testmon affected selection ([#1550](https://github.com/Sinity/polylogue/issues/1550)) ([a2e3bda](https://github.com/Sinity/polylogue/commit/a2e3bda44282321d2292035ee3db4aada3c29469))
* **devtools:** run testmon verify with workers ([#1478](https://github.com/Sinity/polylogue/issues/1478)) ([7db54fd](https://github.com/Sinity/polylogue/commit/7db54fdc94eb7a9983080ef63a80682c0b5a5beb))
* **devtools:** stream pytest progress during verify ([#2073](https://github.com/Sinity/polylogue/issues/2073)) ([43e088e](https://github.com/Sinity/polylogue/commit/43e088e014ab67a3893c931f94462dcd7a4ba821))
* **embeddings:** apply run cost cap during backfill ([#1544](https://github.com/Sinity/polylogue/issues/1544)) ([a255a38](https://github.com/Sinity/polylogue/commit/a255a382fb78ff26928d7467752ce17f0e3edb44))
* **embeddings:** keep status readable across schema bumps ([#1542](https://github.com/Sinity/polylogue/issues/1542)) ([07a501e](https://github.com/Sinity/polylogue/commit/07a501e87e7acacc268e26bf41d392e07e5e465c))
* **embeddings:** unify daemon readiness status ([#1545](https://github.com/Sinity/polylogue/issues/1545)) ([2de683f](https://github.com/Sinity/polylogue/commit/2de683f328eb70320a2d59ba32b7fbb622604e80))
* **facets:** hydrate ConversationSummary.message_count from conversation_stats ([#1623](https://github.com/Sinity/polylogue/issues/1623)) ([#1636](https://github.com/Sinity/polylogue/issues/1636)) ([c07a9dd](https://github.com/Sinity/polylogue/commit/c07a9dd64f4cf42440536250f7e1c642d7910458))
* **fts:** dedupe targeted repair conversations ([#1512](https://github.com/Sinity/polylogue/issues/1512)) ([4a5a53e](https://github.com/Sinity/polylogue/commit/4a5a53eb0464b108dc1869be611e4d5a0386f16b)), closes [#1509](https://github.com/Sinity/polylogue/issues/1509)
* **fts:** ignore empty text in repair predicate ([#1514](https://github.com/Sinity/polylogue/issues/1514)) ([6f89fa3](https://github.com/Sinity/polylogue/commit/6f89fa3b78de2a413b8f21b40ca306766bc339f6)), closes [#1513](https://github.com/Sinity/polylogue/issues/1513)
* harden retrieval freshness and embedding catch-up ([#1522](https://github.com/Sinity/polylogue/issues/1522)) ([28abc7e](https://github.com/Sinity/polylogue/commit/28abc7e0b6aecbd9cac45441a35036681fdd72c2)), closes [#1521](https://github.com/Sinity/polylogue/issues/1521) [#1503](https://github.com/Sinity/polylogue/issues/1503)
* harden search freshness and semantic readiness ([#1504](https://github.com/Sinity/polylogue/issues/1504)) ([cd46fcf](https://github.com/Sinity/polylogue/commit/cd46fcfb9860850c841ef8a5d2de80181e76d1a3)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* hermes message_count off-by-one, dead tool_result comparison, duplicate DDL ([b9307ff](https://github.com/Sinity/polylogue/commit/b9307ff5551bce75cc88bdfbfa517a300407f7ce)), closes [#1751](https://github.com/Sinity/polylogue/issues/1751) [#1752](https://github.com/Sinity/polylogue/issues/1752) [#1753](https://github.com/Sinity/polylogue/issues/1753)
* **import:** ingest decimal-bearing ChatGPT exports ([#1871](https://github.com/Sinity/polylogue/issues/1871)) ([a14c05d](https://github.com/Sinity/polylogue/commit/a14c05d8aa1518f548abe4b4c8a1f051ca1dd057))
* **import:** preflight unsupported sources truthfully ([#1815](https://github.com/Sinity/polylogue/issues/1815)) ([#1969](https://github.com/Sinity/polylogue/issues/1969)) ([0c4a348](https://github.com/Sinity/polylogue/commit/0c4a34842214fbaea49eef0ef81c8d0d40757188))
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
* **mcp:** apply projection flags to message reads ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#1954](https://github.com/Sinity/polylogue/issues/1954)) ([8d46985](https://github.com/Sinity/polylogue/commit/8d46985b2ea88595311047ee4c6be8a8d0415ac4))
* **mcp:** expose shared error envelope core ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2044](https://github.com/Sinity/polylogue/issues/2044)) ([f666000](https://github.com/Sinity/polylogue/commit/f666000f1d4ca7ff34d2dccf34aaf2ebeacb4b8c))
* **mcp:** gate semantic search on embedding readiness ([#1574](https://github.com/Sinity/polylogue/issues/1574)) ([3695456](https://github.com/Sinity/polylogue/commit/3695456f7635dd01230b980462e3d6496307e056)), closes [#1503](https://github.com/Sinity/polylogue/issues/1503)
* **mcp:** isolate tool exceptions; never let one tool kill the stdio server ([#1611](https://github.com/Sinity/polylogue/issues/1611), [#1621](https://github.com/Sinity/polylogue/issues/1621)) ([#1631](https://github.com/Sinity/polylogue/issues/1631)) ([789fc77](https://github.com/Sinity/polylogue/commit/789fc77bc1461765ea849f461da88f05f625b082))
* **parsers/claude-code:** skip type=progress hook lifecycle events ([#1617](https://github.com/Sinity/polylogue/issues/1617)) ([#1641](https://github.com/Sinity/polylogue/issues/1641)) ([34b957a](https://github.com/Sinity/polylogue/commit/34b957a83e6f0e770ac53e0e1532ddbe7fdb15a4))
* **parser:** traverse ChatGPT message graph and extract non-parts content ([#1744](https://github.com/Sinity/polylogue/issues/1744)) ([2d63fea](https://github.com/Sinity/polylogue/commit/2d63feacb02dd098e738b8594b650681c9ef3f03))
* **phases:** fall back to provider_event timestamps when messages have none ([#1624](https://github.com/Sinity/polylogue/issues/1624)) ([#1634](https://github.com/Sinity/polylogue/issues/1634)) ([483a9bb](https://github.com/Sinity/polylogue/commit/483a9bbe7ec8a5fbdf2673cfb6b45b296cf1c88e))
* **pipeline:** surface silent parse-time record loss across ingest ([#1745](https://github.com/Sinity/polylogue/issues/1745)) ([3a47c8f](https://github.com/Sinity/polylogue/commit/3a47c8f8b8e2c3af7fd1f080374efbfd72d4e58a))
* **query:** correct words filter inversion, negation guard, phrase escape, JSON strict ([0dfe9e0](https://github.com/Sinity/polylogue/commit/0dfe9e0f6ac8c2464930063d2817f5c0082cd288))
* **query:** enforce adapter query parity across MCP and daemon ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2042](https://github.com/Sinity/polylogue/issues/2042)) ([1524fd5](https://github.com/Sinity/polylogue/commit/1524fd5a26d04b1192df31cb00e301b3b1235027))
* repair silently-broken read-surface filters and stats queries post-[#1743](https://github.com/Sinity/polylogue/issues/1743) ([#1796](https://github.com/Sinity/polylogue/issues/1796)) ([829e47a](https://github.com/Sinity/polylogue/commit/829e47a7028df57d2766f050c6a74be053ddc76e))
* **repository:** hydrate message_count in get_summary + search_summaries ([#1630](https://github.com/Sinity/polylogue/issues/1630)) ([#1669](https://github.com/Sinity/polylogue/issues/1669)) ([92cb112](https://github.com/Sinity/polylogue/commit/92cb11253035f5bb32a421982a44b02a70857e57))
* retarget query DSL planning and restore quick gate ([#2007](https://github.com/Sinity/polylogue/issues/2007)) ([123c84e](https://github.com/Sinity/polylogue/commit/123c84e28ae9ae3c43db991d0e611b335996732c))
* **scenarios:** keep demo fixture tool paths relative ([#1843](https://github.com/Sinity/polylogue/issues/1843)) ([#1937](https://github.com/Sinity/polylogue/issues/1937)) ([716c19c](https://github.com/Sinity/polylogue/commit/716c19cf15e6690e292b828334bde1d6b93378c1))
* **schema:** remove duplicate source_name column from artifact_observations DDL ([#1022](https://github.com/Sinity/polylogue/issues/1022)) ([081a659](https://github.com/Sinity/polylogue/commit/081a6591cb8027ca3c3595e4673a899718ec1b8d))
* **search:** allow auto cursor follow-up ([#1563](https://github.com/Sinity/polylogue/issues/1563)) ([265b5e7](https://github.com/Sinity/polylogue/commit/265b5e7d0ad81ac7ef53b5b8909c79445d94f6cf))
* **search:** enforce FTS freshness invariant ([#1440](https://github.com/Sinity/polylogue/issues/1440)) ([39c636d](https://github.com/Sinity/polylogue/commit/39c636dc2df95e612e66e982e466a60f7ca78e63)), closes [#1439](https://github.com/Sinity/polylogue/issues/1439)
* **search:** gate semantic queries on embedding readiness ([#1546](https://github.com/Sinity/polylogue/issues/1546)) ([1be50a4](https://github.com/Sinity/polylogue/commit/1be50a4dae7e43e590f78d09e8607d56d84c3095))
* **search:** ground FTS freshness in invariants ([#1500](https://github.com/Sinity/polylogue/issues/1500)) ([ee5a507](https://github.com/Sinity/polylogue/commit/ee5a50741a2f944df47e6f3fa2a4f4cebfcdb652)), closes [#1499](https://github.com/Sinity/polylogue/issues/1499)
* **search:** repair FTS during live ingest ([#1502](https://github.com/Sinity/polylogue/issues/1502)) ([ff0fcc9](https://github.com/Sinity/polylogue/commit/ff0fcc9489da1b0b7ad92dbb46065661bf273550)), closes [#1501](https://github.com/Sinity/polylogue/issues/1501)
* **sources:** parse JSON from non-seekable zip-entry streams ([#1768](https://github.com/Sinity/polylogue/issues/1768)) ([1ee0907](https://github.com/Sinity/polylogue/commit/1ee0907d8ff12b5e9b52902951224661c67a82c7))
* **storage:** canonicalize archive conversation timestamps ([#1555](https://github.com/Sinity/polylogue/issues/1555)) ([576c4b4](https://github.com/Sinity/polylogue/commit/576c4b4f5d5eb7f1fb8ba7a20dc1b3d866e55fb7))
* **storage:** close leaked sqlite connections, unsuppress ResourceWarning ([#1772](https://github.com/Sinity/polylogue/issues/1772)) ([7f5dacd](https://github.com/Sinity/polylogue/commit/7f5dacd2bf1a140499fabda11549e1327ad32c24))
* **storage:** index raw cleanup on existing archives ([#1485](https://github.com/Sinity/polylogue/issues/1485)) ([6e05414](https://github.com/Sinity/polylogue/commit/6e05414db6b52710006e21fa9e0a752bd8176d3c)), closes [#1484](https://github.com/Sinity/polylogue/issues/1484)
* **storage:** lower WAL autocheckpoint from 40 MB to 4 MB ([3c5cc1a](https://github.com/Sinity/polylogue/commit/3c5cc1a075ef6e5c7b55df89dced68a2fc767644))
* **storage:** prevent FTS orphans during replacement ([#1435](https://github.com/Sinity/polylogue/issues/1435)) ([f0219c6](https://github.com/Sinity/polylogue/commit/f0219c6f7b3d17949b00f305ee684d33168bbdf7))
* **storage:** release blob leases on failure and enforce GC generation gate ([#1746](https://github.com/Sinity/polylogue/issues/1746)) ([2744076](https://github.com/Sinity/polylogue/commit/2744076b4b6d57f0a30a5fb81afda56db352fb35))
* **storage:** remove divergent cross-validator, trust canonical content hash ([#1747](https://github.com/Sinity/polylogue/issues/1747)) ([c471e5b](https://github.com/Sinity/polylogue/commit/c471e5b1efc306c3c483891f0fc53aea40b8eaac))
* **test:** isolate convergence benchmark environment ([#1878](https://github.com/Sinity/polylogue/issues/1878)) ([#1911](https://github.com/Sinity/polylogue/issues/1911)) ([0faa1ea](https://github.com/Sinity/polylogue/commit/0faa1ea6d2d023636321f5c4f9ebd25d2b48e062))
* **tests:** isolate concurrent pytest runs with per-run tmpfs basetemp ([#1785](https://github.com/Sinity/polylogue/issues/1785)) ([e789ca5](https://github.com/Sinity/polylogue/commit/e789ca5e24e5b6538911b090e5b19bc3068bdd34))
* **tests:** restore inherited-failure baseline; aggregate_message_stats works on read profile ([#1675](https://github.com/Sinity/polylogue/issues/1675)) ([b9c8081](https://github.com/Sinity/polylogue/commit/b9c8081e2df04bb6e624d80e7c4c292d798b6531))
* **tests:** stop default pytest basetemps from using shm ([#1995](https://github.com/Sinity/polylogue/issues/1995)) ([8647ceb](https://github.com/Sinity/polylogue/commit/8647cebb9a4386333877be004f0724b55aaa4fcb))
* **viewport:** tighten affected_paths heuristic — reject sed, version strings, attribute access ([#1622](https://github.com/Sinity/polylogue/issues/1622)) ([#1646](https://github.com/Sinity/polylogue/issues/1646)) ([3596e08](https://github.com/Sinity/polylogue/commit/3596e08ddfcd790d3d3029da287bdd43bb9ff8a0))
* **watcher:** accept .zip, .json, .ndjson in inbox watch source ([#1692](https://github.com/Sinity/polylogue/issues/1692)) ([9038e09](https://github.com/Sinity/polylogue/commit/9038e09f4091f236dd889c1e2a95668b916947a5))
* **watcher:** interleave catch-up scan by source family ([#1616](https://github.com/Sinity/polylogue/issues/1616)) ([#1644](https://github.com/Sinity/polylogue/issues/1644)) ([3bfa0da](https://github.com/Sinity/polylogue/commit/3bfa0da3641df38b673d7d95f921029d7a6142ed))


### Changed

* **bench:** add FTS trigger amplification benchmark ([#1698](https://github.com/Sinity/polylogue/issues/1698)) ([8798b22](https://github.com/Sinity/polylogue/commit/8798b2280be81050ccce16060ae506cc189d7641))
* **daemon/metrics:** aggregate per-source counts via conversation_stats ([#1629](https://github.com/Sinity/polylogue/issues/1629)) ([#1645](https://github.com/Sinity/polylogue/issues/1645)) ([acd1f1f](https://github.com/Sinity/polylogue/commit/acd1f1f9216206951879fd1e910d3065cfd56bf4))
* **daemon:** avoid virtual FTS status counts ([#1431](https://github.com/Sinity/polylogue/issues/1431)) ([a148aee](https://github.com/Sinity/polylogue/commit/a148aee1a14434ec5231e8a98fc1f0bb0a8ab205))
* **daemon:** bound live ingest memory ([#1429](https://github.com/Sinity/polylogue/issues/1429)) ([bdc1d72](https://github.com/Sinity/polylogue/commit/bdc1d724f7dd3ce6ff50309ff7077d6b54f050fc))
* **daemon:** bound session insight convergence reads ([#1463](https://github.com/Sinity/polylogue/issues/1463)) ([cc0bf5b](https://github.com/Sinity/polylogue/commit/cc0bf5ba02c57499fedd8a93879e3a546f617fd6)), closes [#1447](https://github.com/Sinity/polylogue/issues/1447)
* **daemon:** bound startup FTS checks ([#1434](https://github.com/Sinity/polylogue/issues/1434)) ([4cab346](https://github.com/Sinity/polylogue/commit/4cab346a1cd23358fdb296187b05de5ac98be483))
* **daemon:** chunk large live appends ([#1433](https://github.com/Sinity/polylogue/issues/1433)) ([3dea102](https://github.com/Sinity/polylogue/commit/3dea102964747c98b81629b7cf9b205b82d79e59))
* **daemon:** cut live-ingest write amplification via FTS5 automerge tuning ([#1864](https://github.com/Sinity/polylogue/issues/1864)) ([559920b](https://github.com/Sinity/polylogue/commit/559920b77df72916c486ac7ac233ae48df5e902f))
* **daemon:** downgrade FTS trigger drift to INFO during fresh bulk catch-up ([#1613](https://github.com/Sinity/polylogue/issues/1613)) ([#1652](https://github.com/Sinity/polylogue/issues/1652)) ([afc5118](https://github.com/Sinity/polylogue/commit/afc5118a77d5d89b1ac2690de1337774c1c85e10))
* **daemon:** reduce live catch-up amplification ([#1430](https://github.com/Sinity/polylogue/issues/1430)) ([dd9f200](https://github.com/Sinity/polylogue/commit/dd9f200947a229cf452b15278b97e29fdfef5453))
* **daemon:** targeted FTS repair in archive convergence, not full rebuild ([#1851](https://github.com/Sinity/polylogue/issues/1851)) ([#1895](https://github.com/Sinity/polylogue/issues/1895)) ([bc4bd65](https://github.com/Sinity/polylogue/commit/bc4bd65ec8be94b2ed09a34a3d1be17621cd1203))
* **daemon:** update append stats incrementally ([#1470](https://github.com/Sinity/polylogue/issues/1470)) ([010fe6f](https://github.com/Sinity/polylogue/commit/010fe6f7b1a9dc7d6ed84a0a7eec4a1a648f5d91))
* **devtools:** ingest write-amplification measurement probe ([#1851](https://github.com/Sinity/polylogue/issues/1851)) ([#1900](https://github.com/Sinity/polylogue/issues/1900)) ([bbbc20d](https://github.com/Sinity/polylogue/commit/bbbc20d853c324776dfec895c1b61826abd329d4))
* **fts:** make content_blocks INSERT trigger O(1) per block ([#1705](https://github.com/Sinity/polylogue/issues/1705)) ([8ac581f](https://github.com/Sinity/polylogue/commit/8ac581f162ac253360bef608b785efe77b44ce85))
* **insights:** emit progress per table during full session-insight rebuild ([#1653](https://github.com/Sinity/polylogue/issues/1653)) ([8d3555d](https://github.com/Sinity/polylogue/commit/8d3555decd1dd3737e45477d18374ed5f029f3fd))
* **maintenance:** target stale session insight repairs ([#1471](https://github.com/Sinity/polylogue/issues/1471)) ([27c40d7](https://github.com/Sinity/polylogue/commit/27c40d7124c816ace085065fa87b9dca5c1866eb))
* **metrics:** consolidate raw_conversations COUNT queries ([#1702](https://github.com/Sinity/polylogue/issues/1702)) ([4629b2e](https://github.com/Sinity/polylogue/commit/4629b2ebb886144438f4c17a9869c5e3f62de52a))
* **stats:** use conversation_stats for aggregate message counts ([#1695](https://github.com/Sinity/polylogue/issues/1695)) ([2253e25](https://github.com/Sinity/polylogue/commit/2253e2549772601cda89a2c94674f5b8978b81fb))
* **storage:** cap WAL file via journal_size_limit + fail read-profile writes fast ([#1614](https://github.com/Sinity/polylogue/issues/1614) AC1) ([#1659](https://github.com/Sinity/polylogue/issues/1659)) ([a776e9d](https://github.com/Sinity/polylogue/commit/a776e9d7ae03c4bee294237b45a0b561c3adb652))
* **storage:** keyset pagination in iter_messages, O(1) blob scan ([#1750](https://github.com/Sinity/polylogue/issues/1750)) ([e304de5](https://github.com/Sinity/polylogue/commit/e304de55a7474855114a865cadb56dc492aad4df))

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
