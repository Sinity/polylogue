# Changelog

All notable user-visible changes to Polylogue are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

User-visible: anything an operator running `polylogue` would notice — new
flags, removed or renamed commands, output changes, breaking schema
migrations, security fixes. Internal refactors, test additions, and
documentation polish do not require an entry.

## [0.2.0](https://github.com/Sinity/polylogue/compare/v0.1.0...v0.2.0) (2026-07-02)


### Added

* **analyze:** add --postmortem distilled session bundle ([#2380](https://github.com/Sinity/polylogue/issues/2380)) ([#2431](https://github.com/Sinity/polylogue/issues/2431)) ([b2b12e1](https://github.com/Sinity/polylogue/commit/b2b12e1c48bd91bcb855194420c0a77131326d3b))
* **analyze:** add corpus-wide sanitized portfolio report ([#2437](https://github.com/Sinity/polylogue/issues/2437)) ([#2451](https://github.com/Sinity/polylogue/issues/2451)) ([d101a37](https://github.com/Sinity/polylogue/commit/d101a37c989177dc9d8a2b49a241a7a4818697ac))
* **api:** expose assertion claim reads ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#2086](https://github.com/Sinity/polylogue/issues/2086)) ([277dbbf](https://github.com/Sinity/polylogue/commit/277dbbff9553e1f7c9b1b4b2fb3cb4c990e8d623))
* **api:** expose import explain payloads ([#2178](https://github.com/Sinity/polylogue/issues/2178)) ([#2200](https://github.com/Sinity/polylogue/issues/2200)) ([0de4331](https://github.com/Sinity/polylogue/commit/0de43319d8690b218d046758937f05d6a3039d8f))
* **api:** expose read view profiles ([#1997](https://github.com/Sinity/polylogue/issues/1997)) ([#2023](https://github.com/Sinity/polylogue/issues/2023)) ([9b1f300](https://github.com/Sinity/polylogue/commit/9b1f300a747a7890e9e7569e412e652906a909b4)), closes [#1825](https://github.com/Sinity/polylogue/issues/1825)
* **api:** unify user overlay mutation envelopes ([#1847](https://github.com/Sinity/polylogue/issues/1847)) ([#2155](https://github.com/Sinity/polylogue/issues/2155)) ([71e2671](https://github.com/Sinity/polylogue/commit/71e2671c6f1a471c3b749fb357550fea58528824))
* **archive:** classify message material origin ([#2324](https://github.com/Sinity/polylogue/issues/2324)) ([41b1a82](https://github.com/Sinity/polylogue/commit/41b1a826325e84cd4c2ce517ccf89d3c4c28a217)), closes [#2318](https://github.com/Sinity/polylogue/issues/2318)
* **archive:** preserve structured session evidence ([#2496](https://github.com/Sinity/polylogue/issues/2496)) ([e55c675](https://github.com/Sinity/polylogue/commit/e55c675aaeee8c565c7cd1bb7276af7fccc2866d))
* **assertions:** add candidate judgment workflow ([#2182](https://github.com/Sinity/polylogue/issues/2182)) ([#2204](https://github.com/Sinity/polylogue/issues/2204)) ([28455f9](https://github.com/Sinity/polylogue/commit/28455f9cf1cdd0f39b43a69cecfa2891f386e5ad))
* **assertions:** default lifecycle fields explicitly ([#2113](https://github.com/Sinity/polylogue/issues/2113)) ([8f6ca96](https://github.com/Sinity/polylogue/commit/8f6ca96005a51eb170c937fc732839742713578c)), closes [#1883](https://github.com/Sinity/polylogue/issues/1883)
* **assertions:** export user-tier assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#2098](https://github.com/Sinity/polylogue/issues/2098)) ([7135ed3](https://github.com/Sinity/polylogue/commit/7135ed34fca4676e5aa4ce57b7e0cdf7e1c391a8))
* **backup:** support named archive profiles ([#1830](https://github.com/Sinity/polylogue/issues/1830)) ([#1963](https://github.com/Sinity/polylogue/issues/1963)) ([0f70c91](https://github.com/Sinity/polylogue/commit/0f70c91af62a079004f654ea858f29fae50d7cc6))
* **browser-capture:** add gated post commands ([#2494](https://github.com/Sinity/polylogue/issues/2494)) ([79a8ac0](https://github.com/Sinity/polylogue/commit/79a8ac097aa68304bdb5b3e53ef1c1a7d6556afd))
* **browser-capture:** require auth for web origins ([#2116](https://github.com/Sinity/polylogue/issues/2116)) ([a2260dc](https://github.com/Sinity/polylogue/commit/a2260dc4a7b3d25ad5f5d4a55359ab28c2f06c08)), closes [#1824](https://github.com/Sinity/polylogue/issues/1824) [#1847](https://github.com/Sinity/polylogue/issues/1847)
* **capture:** materialize rich web constructs ([#2378](https://github.com/Sinity/polylogue/issues/2378)) ([bad677e](https://github.com/Sinity/polylogue/commit/bad677ebb6c92bda2a474c9ba647ca6352d99712))
* **cli:** add direct facets command ([#2361](https://github.com/Sinity/polylogue/issues/2361)) ([7be9b59](https://github.com/Sinity/polylogue/commit/7be9b59e148380780a325e67ac0482db50fc42c8)), closes [#2317](https://github.com/Sinity/polylogue/issues/2317)
* **cli:** add public action contracts ([#1816](https://github.com/Sinity/polylogue/issues/1816)) ([#1941](https://github.com/Sinity/polylogue/issues/1941)) ([6e8eb9e](https://github.com/Sinity/polylogue/commit/6e8eb9ed0564351455b12901b5adc1d753d437d7))
* **cli:** complete actions from command contracts ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2019](https://github.com/Sinity/polylogue/issues/2019)) ([a91557f](https://github.com/Sinity/polylogue/commit/a91557f7849fdacbc587c2ad92940ba1e93feda7))
* **cli:** complete query actions after then ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2026](https://github.com/Sinity/polylogue/issues/2026)) ([d84e37a](https://github.com/Sinity/polylogue/commit/d84e37a8db7d66a6a5c8652d884e10125a506779))
* **cli:** complete query count operators ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2029](https://github.com/Sinity/polylogue/issues/2029)) ([7b05904](https://github.com/Sinity/polylogue/commit/7b059049feac808a4c94717c63bdb0620a5f86bf)), closes [#2006](https://github.com/Sinity/polylogue/issues/2006)
* **cli:** complete query date operators ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2032](https://github.com/Sinity/polylogue/issues/2032)) ([3154698](https://github.com/Sinity/polylogue/commit/3154698191236ba4bc5cd2aea19b920bcabeb8c1))
* **cli:** complete query field values from descriptors ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2017](https://github.com/Sinity/polylogue/issues/2017)) ([7c963d4](https://github.com/Sinity/polylogue/commit/7c963d4c3e50abdd512e752754f1326d7de68b83))
* **cli:** complete query fields from grammar registry ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2016](https://github.com/Sinity/polylogue/issues/2016)) ([1b7c697](https://github.com/Sinity/polylogue/commit/1b7c697b6c07d28bdae089c83fdb502c12f7db7c))
* **cli:** complete read formats from view profiles ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2034](https://github.com/Sinity/polylogue/issues/2034)) ([ddf9694](https://github.com/Sinity/polylogue/commit/ddf969491cf9eddc0f1488431cf512e150d88773))
* **cli:** expose json aliases on action commands ([#2368](https://github.com/Sinity/polylogue/issues/2368)) ([3a56081](https://github.com/Sinity/polylogue/commit/3a56081b66da984c4e6b821e2428e8e4e7ad6ec5)), closes [#1844](https://github.com/Sinity/polylogue/issues/1844) [#2006](https://github.com/Sinity/polylogue/issues/2006) [#2317](https://github.com/Sinity/polylogue/issues/2317)
* **cli:** expose query completion metadata ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2037](https://github.com/Sinity/polylogue/issues/2037)) ([a615c85](https://github.com/Sinity/polylogue/commit/a615c852038fa65e6b2b7cfe5a657235b3efa09d))
* **cli:** expose query completion metadata ([#2292](https://github.com/Sinity/polylogue/issues/2292)) ([7ab6dbe](https://github.com/Sinity/polylogue/commit/7ab6dbe1d1832520d21f47985818e607b2609c78))
* **cli:** expose query-backed select verb ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2036](https://github.com/Sinity/polylogue/issues/2036)) ([836504f](https://github.com/Sinity/polylogue/commit/836504f8b7f772d443092e7f4290dd27fe7d3fb8))
* **cli:** expose read view profiles ([#1997](https://github.com/Sinity/polylogue/issues/1997)) ([#2022](https://github.com/Sinity/polylogue/issues/2022)) ([837af0a](https://github.com/Sinity/polylogue/commit/837af0ae11dab370f9a24d463b37a5cea398723b))
* **cli:** expose recovery report presets ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1947](https://github.com/Sinity/polylogue/issues/1947)) ([8703692](https://github.com/Sinity/polylogue/commit/8703692926c55fc3ff18c8bf0a571e731ed2d0c3))
* **context:** compile bounded context images ([#2195](https://github.com/Sinity/polylogue/issues/2195)) ([d1cb180](https://github.com/Sinity/polylogue/commit/d1cb18066a15b15ae44f3010988c47f7f0e45ff6))
* **context:** compile message read-view segments ([#2193](https://github.com/Sinity/polylogue/issues/2193)) ([#2237](https://github.com/Sinity/polylogue/issues/2237)) ([0416d7b](https://github.com/Sinity/polylogue/commit/0416d7b60f0ac05cd671fedcade96c280f9be0f0))
* **context:** compile recovery views through one payload ([#2118](https://github.com/Sinity/polylogue/issues/2118)) ([82d8b20](https://github.com/Sinity/polylogue/commit/82d8b20b39aed3e444523c5d3b064a0bba998f9a)), closes [#1838](https://github.com/Sinity/polylogue/issues/1838) [#1882](https://github.com/Sinity/polylogue/issues/1882)
* **context:** feed assertion claims into recovery context ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#2128](https://github.com/Sinity/polylogue/issues/2128)) ([8b34aaf](https://github.com/Sinity/polylogue/commit/8b34aaf875e1457daeb51129fd9e6a970b8d2ab9))
* **context:** record delivered context snapshots ([#2193](https://github.com/Sinity/polylogue/issues/2193)) ([#2203](https://github.com/Sinity/polylogue/issues/2203)) ([24703f6](https://github.com/Sinity/polylogue/commit/24703f6baf43ff291963b6b748f91cc8c3fab21f))
* converge dogfood branch state ([#2504](https://github.com/Sinity/polylogue/issues/2504)) ([d40d00a](https://github.com/Sinity/polylogue/commit/d40d00a312e5530a76e3e1ba29f3ddc37ce89d82))
* **cost:** price current Opus 4.7/4.8 flagships ([#2445](https://github.com/Sinity/polylogue/issues/2445)) ([fbb8f09](https://github.com/Sinity/polylogue/commit/fbb8f09187ae013b6a27db01e2961440f507cacb))
* **daemon:** classify HTTP route contracts ([#1847](https://github.com/Sinity/polylogue/issues/1847)) ([#2053](https://github.com/Sinity/polylogue/issues/2053)) ([9471bf2](https://github.com/Sinity/polylogue/commit/9471bf25f4e53f051d719b0dfd995aa1ddbafe3a))
* **daemon:** expose query completion metadata endpoint ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2039](https://github.com/Sinity/polylogue/issues/2039)) ([2f4d57d](https://github.com/Sinity/polylogue/commit/2f4d57d92405d83eb5595d505d761f49974f433b)), closes [#1846](https://github.com/Sinity/polylogue/issues/1846)
* **daemon:** expose read view profile metadata ([#1997](https://github.com/Sinity/polylogue/issues/1997)) ([#2041](https://github.com/Sinity/polylogue/issues/2041)) ([f2eb724](https://github.com/Sinity/polylogue/commit/f2eb7248613c90ba0d630825b0536c7571b3e2dd)), closes [#1846](https://github.com/Sinity/polylogue/issues/1846)
* **daemon:** publish stable read-view route envelope ([#1847](https://github.com/Sinity/polylogue/issues/1847)) ([#2166](https://github.com/Sinity/polylogue/issues/2166)) ([addfc1b](https://github.com/Sinity/polylogue/commit/addfc1b091527f3d8a6ef0668ccf94e2ae12bc3b))
* **daemon:** stabilize evidence read API routes ([#1847](https://github.com/Sinity/polylogue/issues/1847)) ([#2169](https://github.com/Sinity/polylogue/issues/2169)) ([69b6368](https://github.com/Sinity/polylogue/commit/69b6368d63191bd41308ed583235d2986dea4218))
* **demo:** add deterministic archive seed commands ([#2196](https://github.com/Sinity/polylogue/issues/2196)) ([#2208](https://github.com/Sinity/polylogue/issues/2208)) ([c7cedbb](https://github.com/Sinity/polylogue/commit/c7cedbbe2d80639fadd6ed86a0c0ffe47d07a70e))
* **demo:** enrich demo cost and fix profile token-lane round-trip ([#2446](https://github.com/Sinity/polylogue/issues/2446)) ([05f8c98](https://github.com/Sinity/polylogue/commit/05f8c980c5c516bec358abcdd841edbf742c1e07))
* **demo:** give demo session a canonical repo for repos_touched ([#2447](https://github.com/Sinity/polylogue/issues/2447)) ([b73f01d](https://github.com/Sinity/polylogue/commit/b73f01db8e2f008bee5801f17a297d80867a105d))
* **demo:** let daemon demo import wait for readiness ([#2196](https://github.com/Sinity/polylogue/issues/2196)) ([#2210](https://github.com/Sinity/polylogue/issues/2210)) ([3b99bbd](https://github.com/Sinity/polylogue/commit/3b99bbd16e3c042821bf7c9d450b0d92dda9c9ee))
* **demo:** materialize session insights in no-daemon seed ([#2439](https://github.com/Sinity/polylogue/issues/2439)) ([21951de](https://github.com/Sinity/polylogue/commit/21951de986200babb056d1f9c77ad17758ec4fad))
* **demo:** seed deterministic user overlays ([#1843](https://github.com/Sinity/polylogue/issues/1843)) ([#1975](https://github.com/Sinity/polylogue/issues/1975)) ([b61bb63](https://github.com/Sinity/polylogue/commit/b61bb63bd092ac70fac8e6a18543e0192a820aee))
* **devtools:** add bench ingest-throughput wall-clock benchmark ([#2490](https://github.com/Sinity/polylogue/issues/2490)) ([d415412](https://github.com/Sinity/polylogue/commit/d415412d7d68008a92296f6f6dd29f53b4cb7a10))
* **devtools:** catch subcommand/flag drift in verify doc-commands ([#2442](https://github.com/Sinity/polylogue/issues/2442)) ([0d5a3bb](https://github.com/Sinity/polylogue/commit/0d5a3bb6ee3c18e1abc4690a249aa465d0ce5e9d))
* **devtools:** instrument ingest with cpu/memory/io/stage metrics ([#2491](https://github.com/Sinity/polylogue/issues/2491)) ([52b5bc0](https://github.com/Sinity/polylogue/commit/52b5bc038a0bd959563cf61db4a1e0fd15001ea3))
* **export:** add fail-closed sanitized shareable export ([#2381](https://github.com/Sinity/polylogue/issues/2381)) ([#2433](https://github.com/Sinity/polylogue/issues/2433)) ([f5ab219](https://github.com/Sinity/polylogue/commit/f5ab2195558823fd01d6db432a32e1768d6ba930))
* **export:** add sanitized spans-jsonl bundle leg ([#2434](https://github.com/Sinity/polylogue/issues/2434)) ([#2455](https://github.com/Sinity/polylogue/issues/2455)) ([c4b6771](https://github.com/Sinity/polylogue/commit/c4b677181847375d9636cb9448e3accbf4688595))
* **export:** redact Windows paths and emails in sanitized export ([#2444](https://github.com/Sinity/polylogue/issues/2444)) ([c05e83e](https://github.com/Sinity/polylogue/commit/c05e83ee9d1c2a8605f48f2d640749c0ddae62db))
* harden reader, context, and config surfaces ([#2322](https://github.com/Sinity/polylogue/issues/2322)) ([d1ea8de](https://github.com/Sinity/polylogue/commit/d1ea8debe99973e6562bc7aeb2a221d54b55a830)), closes [#2304](https://github.com/Sinity/polylogue/issues/2304) [#2306](https://github.com/Sinity/polylogue/issues/2306) [#2307](https://github.com/Sinity/polylogue/issues/2307) [#2309](https://github.com/Sinity/polylogue/issues/2309)
* **import:** explain parser decisions for local imports ([#2178](https://github.com/Sinity/polylogue/issues/2178)) ([#2186](https://github.com/Sinity/polylogue/issues/2186)) ([6b20350](https://github.com/Sinity/polylogue/commit/6b20350a1ca648a5f91641d41061aad088edc4de))
* **import:** expose archive-backed import explain ([#2249](https://github.com/Sinity/polylogue/issues/2249)) ([a818508](https://github.com/Sinity/polylogue/commit/a818508d8961307439a49a8fb75ff5d91d7b0719))
* **insights:** add work-packet target refs ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2092](https://github.com/Sinity/polylogue/issues/2092)) ([3589ad6](https://github.com/Sinity/polylogue/commit/3589ad61dcbfc301571750274ad0ac46497eb675)), closes [#1845](https://github.com/Sinity/polylogue/issues/1845)
* **insights:** carry object refs in work packets ([#1845](https://github.com/Sinity/polylogue/issues/1845)) ([#2089](https://github.com/Sinity/polylogue/issues/2089)) ([9e6eb30](https://github.com/Sinity/polylogue/commit/9e6eb3012f5c4460ef67f5e23da3f59b107a68e0))
* **insights:** cite check runs in work packets ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2095](https://github.com/Sinity/polylogue/issues/2095)) ([0d1e316](https://github.com/Sinity/polylogue/commit/0d1e316d92b0061cffd198aceb11e9cac0982377)), closes [#1845](https://github.com/Sinity/polylogue/issues/1845)
* **insights:** deterministic agent-workflow pathology detectors ([#2448](https://github.com/Sinity/polylogue/issues/2448)) ([72d0623](https://github.com/Sinity/polylogue/commit/72d0623ea331f5348ceb7c6fde61e7c6ce41d200))
* **insights:** expose pathology distribution via API + MCP tool ([#2383](https://github.com/Sinity/polylogue/issues/2383)) ([#2449](https://github.com/Sinity/polylogue/issues/2449)) ([e7a39b3](https://github.com/Sinity/polylogue/commit/e7a39b37fb0de39ba103bb4633bce8d4ba90da23))
* **insights:** project run evidence into work packets ([#1882](https://github.com/Sinity/polylogue/issues/1882)) ([#2097](https://github.com/Sinity/polylogue/issues/2097)) ([72bbbcd](https://github.com/Sinity/polylogue/commit/72bbbcde9b1e6d3d8f99bef504cdaa275eb2f521))
* **insights:** render review refs in work packets ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2106](https://github.com/Sinity/polylogue/issues/2106)) ([a8cd1c1](https://github.com/Sinity/polylogue/commit/a8cd1c1516b29068ec9ce1493f262d663407ffa5)), closes [#1845](https://github.com/Sinity/polylogue/issues/1845)
* **insights:** track review delivery events ([#1882](https://github.com/Sinity/polylogue/issues/1882)) ([#2104](https://github.com/Sinity/polylogue/issues/2104)) ([6d9f178](https://github.com/Sinity/polylogue/commit/6d9f178d6d01b606332846bba1edbd7a49718d48))
* integrate archive contract patch wave ([#2358](https://github.com/Sinity/polylogue/issues/2358)) ([dc47a63](https://github.com/Sinity/polylogue/commit/dc47a63147757fc01ef233d039e4cf85cea2e6c7))
* integrate query-action archive diagnostics ([#2350](https://github.com/Sinity/polylogue/issues/2350)) ([8ae4eaf](https://github.com/Sinity/polylogue/commit/8ae4eaf8660a1f26fa39f8b3b94ae45bf269708c))
* **maintenance:** add archive backup plan surface ([#1830](https://github.com/Sinity/polylogue/issues/1830)) ([#1952](https://github.com/Sinity/polylogue/issues/1952)) ([087456c](https://github.com/Sinity/polylogue/commit/087456c377f187ad4e3bc490e26dbb845421d067))
* **maintenance:** report blob GC dry-runs ([#1830](https://github.com/Sinity/polylogue/issues/1830)) ([#1985](https://github.com/Sinity/polylogue/issues/1985)) ([bb43e40](https://github.com/Sinity/polylogue/commit/bb43e40730c1474bca6220dd7e6a80020ef12e37))
* **mcp:** expose assertion claim reads ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#2088](https://github.com/Sinity/polylogue/issues/2088)) ([9a21b17](https://github.com/Sinity/polylogue/commit/9a21b176e73848805470e4322cd0457d4f9389e9)), closes [#1825](https://github.com/Sinity/polylogue/issues/1825)
* **mcp:** expose embedding component readiness ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1961](https://github.com/Sinity/polylogue/issues/1961)) ([3347ba4](https://github.com/Sinity/polylogue/commit/3347ba463db9a3715a02e820039d4a92cf5996d8))
* **mcp:** expose postmortem bundle and sanitized export as agent tools ([#2443](https://github.com/Sinity/polylogue/issues/2443)) ([66fb40a](https://github.com/Sinity/polylogue/commit/66fb40af5c4a7aeb8cde44b6b605414602e0d60c))
* **mcp:** expose readiness component map ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1971](https://github.com/Sinity/polylogue/issues/1971)) ([b0d7c4f](https://github.com/Sinity/polylogue/commit/b0d7c4fbb82ae4daadeeecf52045b3679629c6e9))
* **mcp:** expose recovery reports ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2096](https://github.com/Sinity/polylogue/issues/2096)) ([0474e7d](https://github.com/Sinity/polylogue/commit/0474e7d2766b27ed822f96cfe70c8959557b82f0)), closes [#2006](https://github.com/Sinity/polylogue/issues/2006)
* **mcp:** expose recovery work packets ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2083](https://github.com/Sinity/polylogue/issues/2083)) ([38376c4](https://github.com/Sinity/polylogue/commit/38376c41f8a357752a5310a3cdfef956dd165d77))
* normalize session lineage and correct cross-provider cost accounting ([#2469](https://github.com/Sinity/polylogue/issues/2469)) ([7412b69](https://github.com/Sinity/polylogue/commit/7412b69a9b64863a1bf4df11d7feeb99f61262ff))
* **ops:** expose archive debt across surfaces ([#2179](https://github.com/Sinity/polylogue/issues/2179)) ([#2213](https://github.com/Sinity/polylogue/issues/2213)) ([7214a41](https://github.com/Sinity/polylogue/commit/7214a41a30660c355301828ddbf92f74e3444dca))
* **ops:** list unified archive debt rows ([#2179](https://github.com/Sinity/polylogue/issues/2179)) ([#2202](https://github.com/Sinity/polylogue/issues/2202)) ([0b62abd](https://github.com/Sinity/polylogue/commit/0b62abd94577bed806721508f0d30e26f31e4715))
* **parsers:** capture Claude server_tool_use web request counts ([#2486](https://github.com/Sinity/polylogue/issues/2486)) ([badd731](https://github.com/Sinity/polylogue/commit/badd73180f1e33db3bcf26db02800f4648e4a965))
* **providers:** report importer package completeness ([#2180](https://github.com/Sinity/polylogue/issues/2180)) ([#2189](https://github.com/Sinity/polylogue/issues/2189)) ([25b9ccd](https://github.com/Sinity/polylogue/commit/25b9ccdbfd7967b68378d06273d8e9c32f4f91c4))
* **query:** add affected-file query unit ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2287](https://github.com/Sinity/polylogue/issues/2287)) ([04cc229](https://github.com/Sinity/polylogue/commit/04cc229ce17ed33f5f44070642f06df243a61c86))
* **query:** add aggregate count pipelines ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2233](https://github.com/Sinity/polylogue/issues/2233)) ([6503340](https://github.com/Sinity/polylogue/commit/6503340e71a0e91574114c33a08988910c6591c3))
* **query:** add aggregate session count predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2241](https://github.com/Sinity/polylogue/issues/2241)) ([3e0dacc](https://github.com/Sinity/polylogue/commit/3e0dacc4f3b7189055eb26aa57570710da9999f5))
* **query:** add continue action ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2094](https://github.com/Sinity/polylogue/issues/2094)) ([e15ef4b](https://github.com/Sinity/polylogue/commit/e15ef4ba232bb9c55da4069c8c69346b957af209)), closes [#1807](https://github.com/Sinity/polylogue/issues/1807)
* **query:** add numeric duration and token predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2242](https://github.com/Sinity/polylogue/issues/2242)) ([7fe698a](https://github.com/Sinity/polylogue/commit/7fe698ac48eb974865321b768fe4b0adc2c40d3a))
* **query:** add row-time predicates for query units ([#2235](https://github.com/Sinity/polylogue/issues/2235)) ([5acff0f](https://github.com/Sinity/polylogue/commit/5acff0fa27a13696749ac0818115feda440decff))
* **query:** add Terminal pipeline stage; route verbs through one executor ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2463](https://github.com/Sinity/polylogue/issues/2463)) ([fb9073c](https://github.com/Sinity/polylogue/commit/fb9073cfc801a6b4d8150e6998c96b966e39047c))
* **query:** add terminal pipeline window stages ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2176](https://github.com/Sinity/polylogue/issues/2176)) ([988e75d](https://github.com/Sinity/polylogue/commit/988e75d788b35e4d6bec95ea62255479c7511467))
* **query:** allow direct terminal pipeline stages ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2199](https://github.com/Sinity/polylogue/issues/2199)) ([8ae5b4a](https://github.com/Sinity/polylogue/commit/8ae5b4aa96da71507f0a04a49c3522d8ef0e784f))
* **query:** attach related units to session queries ([#2495](https://github.com/Sinity/polylogue/issues/2495)) ([4a9c7a0](https://github.com/Sinity/polylogue/commit/4a9c7a07a12787a2b71a13f588b1b4dc835747a5))
* **query:** bind predicates to field refs ([#2254](https://github.com/Sinity/polylogue/issues/2254)) ([e6032ac](https://github.com/Sinity/polylogue/commit/e6032ac8d0ad255db3faeabbdca25103da5a7638))
* **query:** complete terminal query-unit fields ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2152](https://github.com/Sinity/polylogue/issues/2152)) ([5b17ad1](https://github.com/Sinity/polylogue/commit/5b17ad15f06260c491528a62d5f2713a0e011d64))
* **query:** derive terminal unit surfaces from descriptors ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2243](https://github.com/Sinity/polylogue/issues/2243)) ([8fc492e](https://github.com/Sinity/polylogue/commit/8fc492ef9f7d8b5bf63718c52374153d539aac6c))
* **query:** describe structural DSL fields ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2093](https://github.com/Sinity/polylogue/issues/2093)) ([7fae014](https://github.com/Sinity/polylogue/commit/7fae0146dad0a0a6847a0acf9ef01b2827da832d))
* **query:** execute block structural predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2024](https://github.com/Sinity/polylogue/issues/2024)) ([75cf1c4](https://github.com/Sinity/polylogue/commit/75cf1c48e2f6064d58d69761337b3294b6015915))
* **query:** execute Boolean FTS predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2012](https://github.com/Sinity/polylogue/issues/2012)) ([a30dc78](https://github.com/Sinity/polylogue/commit/a30dc789fe910f634c7222c6ee9ca00a388b4b48))
* **query:** execute Boolean session predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2010](https://github.com/Sinity/polylogue/issues/2010)) ([5fd1652](https://github.com/Sinity/polylogue/commit/5fd16520a879227c7ce2caa2e5f0ae86ff2485e9))
* **query:** execute lineage DSL predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2014](https://github.com/Sinity/polylogue/issues/2014)) ([cb641e6](https://github.com/Sinity/polylogue/commit/cb641e631614f839b643d4010fdd0b83ce50f791))
* **query:** execute role-count predicates through DSL ([#2240](https://github.com/Sinity/polylogue/issues/2240)) ([f6151e2](https://github.com/Sinity/polylogue/commit/f6151e2957da55d8677b532c9b6fbd4632034662))
* **query:** execute semantic DSL predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2013](https://github.com/Sinity/polylogue/issues/2013)) ([c96a046](https://github.com/Sinity/polylogue/commit/c96a046cb0e846b49420f3277dc65f645350d1b4))
* **query:** execute structural and sequence predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2011](https://github.com/Sinity/polylogue/issues/2011)) ([def884e](https://github.com/Sinity/polylogue/commit/def884ef305bdaf340b0f79717249e80194c7585))
* **query:** execute unit-scoped where predicates ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2046](https://github.com/Sinity/polylogue/issues/2046)) ([9323f3d](https://github.com/Sinity/polylogue/commit/9323f3d677b0863ff117977cca9df263d84f4760))
* **query:** explain terminal unit sources ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2054](https://github.com/Sinity/polylogue/issues/2054)) ([94ca249](https://github.com/Sinity/polylogue/commit/94ca249ce838c31a1c7aed036774f849b8ca56e2))
* **query:** expose assertions as terminal query units ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2079](https://github.com/Sinity/polylogue/issues/2079)) ([b855726](https://github.com/Sinity/polylogue/commit/b855726092353b39f7f5f92c1f1894dc782decbf)), closes [#1883](https://github.com/Sinity/polylogue/issues/1883)
* **query:** expose authoredness filters ([#2325](https://github.com/Sinity/polylogue/issues/2325)) ([b66d71b](https://github.com/Sinity/polylogue/commit/b66d71b8eaf0882e3477122bd90204bfb3c4ce8b)), closes [#2318](https://github.com/Sinity/polylogue/issues/2318)
* **query:** expose DSL explain command ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2015](https://github.com/Sinity/polylogue/issues/2015)) ([2ae9e4d](https://github.com/Sinity/polylogue/commit/2ae9e4d308b43db152f57127dd92441ac6738158))
* **query:** expose observed-events terminal unit ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2108](https://github.com/Sinity/polylogue/issues/2108)) ([3599153](https://github.com/Sinity/polylogue/commit/3599153c27f0a85f5a62f28e63d447d49642f0ee))
* **query:** expose pipeline stage completions ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2239](https://github.com/Sinity/polylogue/issues/2239)) ([e659d0f](https://github.com/Sinity/polylogue/commit/e659d0f5b7d4453eb217543f01a5ea81af9872d1))
* **query:** expose pipeline stages on query-unit results ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2232](https://github.com/Sinity/polylogue/issues/2232)) ([7eadd82](https://github.com/Sinity/polylogue/commit/7eadd827f5edb91ff852080287bdfcba7ad62f47))
* **query:** expose query explain execution legs ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2027](https://github.com/Sinity/polylogue/issues/2027)) ([75759ef](https://github.com/Sinity/polylogue/commit/75759efbcbf4978645c205a8c2ce73b048ae0c68))
* **query:** expose query explain through API and MCP ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2030](https://github.com/Sinity/polylogue/issues/2030)) ([7038099](https://github.com/Sinity/polylogue/commit/70380995104e91664882c70bf95924cbd7961f89))
* **query:** expose run terminal query rows ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2157](https://github.com/Sinity/polylogue/issues/2157)) ([e62a2d9](https://github.com/Sinity/polylogue/commit/e62a2d90038cd4c99c8c0b7989ff98523896b9d0))
* **query:** expose structural grammar completion metadata ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2025](https://github.com/Sinity/polylogue/issues/2025)) ([2d41b05](https://github.com/Sinity/polylogue/commit/2d41b0571c55e2a433c8088ff828de1a61ffc80a))
* **query:** expose structured query lowering plans ([#2059](https://github.com/Sinity/polylogue/issues/2059)) ([c777771](https://github.com/Sinity/polylogue/commit/c777771716ba034018f75733393d36f8f36832eb))
* **query:** expose terminal pipeline stages in explain ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2231](https://github.com/Sinity/polylogue/issues/2231)) ([feddca7](https://github.com/Sinity/polylogue/commit/feddca7d00db88caa0f7d13b0b976edb122fdb2d))
* **query:** expose unit payload models from descriptors ([#2218](https://github.com/Sinity/polylogue/issues/2218)) ([46fea9b](https://github.com/Sinity/polylogue/commit/46fea9bf6ef2a326248b230cc046c393b712c27b))
* **query:** honor scoped session summary comparisons ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2133](https://github.com/Sinity/polylogue/issues/2133)) ([a4ae0fe](https://github.com/Sinity/polylogue/commit/a4ae0fea3c0bfaa986b7e5d607e9346b9a96c494))
* **query:** let unit descriptors own terminal metadata ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2207](https://github.com/Sinity/polylogue/issues/2207)) ([088d620](https://github.com/Sinity/polylogue/commit/088d6209b0992107d0a061f0a912a108e4e7a346))
* **query:** lower session stages in terminal pipelines ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2275](https://github.com/Sinity/polylogue/issues/2275)) ([dc182ed](https://github.com/Sinity/polylogue/commit/dc182edbc1e6fa9ecf65cf47cf80066c3785343f))
* **query:** model terminal pipelines as execution objects ([#2289](https://github.com/Sinity/polylogue/issues/2289)) ([825e57c](https://github.com/Sinity/polylogue/commit/825e57cf269673d50c550825b39a00539e8e9979))
* **query:** parse query expressions with Lark ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2009](https://github.com/Sinity/polylogue/issues/2009)) ([0a4fbfd](https://github.com/Sinity/polylogue/commit/0a4fbfdbda470b396068c0c7a7ff4d3b71cdc512))
* **query:** parse readable count comparisons ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2028](https://github.com/Sinity/polylogue/issues/2028)) ([29a11d2](https://github.com/Sinity/polylogue/commit/29a11d2636f5a31e36b204717b8811c1929ef132))
* **query:** parse readable date comparisons ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2031](https://github.com/Sinity/polylogue/issues/2031)) ([b919e28](https://github.com/Sinity/polylogue/commit/b919e28a82d9db4d5e58b4ad3eddbf96e4704b2e))
* **query:** pipe session scopes into terminal rows ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2175](https://github.com/Sinity/polylogue/issues/2175)) ([c61ff42](https://github.com/Sinity/polylogue/commit/c61ff42a8609c168c3b17c2e7cc56449a6a09d90))
* **query:** return terminal unit rows for source queries ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2049](https://github.com/Sinity/polylogue/issues/2049)) ([975336c](https://github.com/Sinity/polylogue/commit/975336c6a5e089a2f85fc995d2ba012b0da48437)), closes [#1825](https://github.com/Sinity/polylogue/issues/1825)
* **query:** scope unit predicates by owning session ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2076](https://github.com/Sinity/polylogue/issues/2076)) ([4b5601e](https://github.com/Sinity/polylogue/commit/4b5601e9c40e54083f76eb0dc77c19cd39e5c97d))
* **query:** share completion metadata across API and MCP ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2038](https://github.com/Sinity/polylogue/issues/2038)) ([ce34edd](https://github.com/Sinity/polylogue/commit/ce34edddc33814f41c08b37eb3b76716c8718a70)), closes [#1825](https://github.com/Sinity/polylogue/issues/1825)
* **query:** share unit-row session filters ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2056](https://github.com/Sinity/polylogue/issues/2056)) ([9233a45](https://github.com/Sinity/polylogue/commit/9233a451733736641e420793925e5a5d0ec02a04)), closes [#2006](https://github.com/Sinity/polylogue/issues/2006)
* **query:** sort terminal aggregate pipelines ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2238](https://github.com/Sinity/polylogue/issues/2238)) ([c3d48e1](https://github.com/Sinity/polylogue/commit/c3d48e130abf3f7b757f3213611ae4158e779ca3))
* **query:** sort terminal pipeline rows by time ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2187](https://github.com/Sinity/polylogue/issues/2187)) ([4aee6b6](https://github.com/Sinity/polylogue/commit/4aee6b6a15225b27d125328eccd236c358d932a8))
* **query:** suggest close field names in DSL errors ([#1844](https://github.com/Sinity/polylogue/issues/1844)) ([#2018](https://github.com/Sinity/polylogue/issues/2018)) ([6ccd785](https://github.com/Sinity/polylogue/commit/6ccd785a3d6c34e7c3df3a293f25316fd3610a6f))
* **query:** support predicate-constrained action sequences ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2244](https://github.com/Sinity/polylogue/issues/2244)) ([ac1bbf9](https://github.com/Sinity/polylogue/commit/ac1bbf9693a839ad54b8daa7ed704fcba2c09c85))
* **read:** add projection render specs ([#2503](https://github.com/Sinity/polylogue/issues/2503)) ([ce2f62a](https://github.com/Sinity/polylogue/commit/ce2f62ac65e6e7c29558a74f1dd10ba78d0999e2))
* **readiness:** add capability readiness contract ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1938](https://github.com/Sinity/polylogue/issues/1938)) ([11635b0](https://github.com/Sinity/polylogue/commit/11635b0a70bac9517305463de23d9ca1d6125cf4))
* **read:** profile executable session views ([#1997](https://github.com/Sinity/polylogue/issues/1997)) ([#2021](https://github.com/Sinity/polylogue/issues/2021)) ([8febe37](https://github.com/Sinity/polylogue/commit/8febe37d003f29868c066f88e8079a126d342c36)), closes [#1844](https://github.com/Sinity/polylogue/issues/1844)
* **recovery:** enrich work-packet evidence details ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2084](https://github.com/Sinity/polylogue/issues/2084)) ([8a21afb](https://github.com/Sinity/polylogue/commit/8a21afb752d4bdb9b00977987de2267551fdb6e8)), closes [#1883](https://github.com/Sinity/polylogue/issues/1883)
* **recovery:** mark work-packet evidence support ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2082](https://github.com/Sinity/polylogue/issues/2082)) ([2551fde](https://github.com/Sinity/polylogue/commit/2551fde9250757dbf6131b1aab03187c14498b33))
* **recovery:** resolve subagent child links in digests ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#2050](https://github.com/Sinity/polylogue/issues/2050)) ([f5c8827](https://github.com/Sinity/polylogue/commit/f5c882787e9843c33fcdd182914d32fe5f0366e8))
* **refs:** carry object refs through evidence surfaces ([#2150](https://github.com/Sinity/polylogue/issues/2150)) ([e60ed3b](https://github.com/Sinity/polylogue/commit/e60ed3b2beb206022686a787b6155394232fc2c7))
* **refs:** normalize assertion public refs ([#2114](https://github.com/Sinity/polylogue/issues/2114)) ([2dad8ce](https://github.com/Sinity/polylogue/commit/2dad8ce472fb3cfc729e63aafae902eaeaa92d8b)), closes [#1845](https://github.com/Sinity/polylogue/issues/1845)
* **refs:** resolve public archive refs ([#2181](https://github.com/Sinity/polylogue/issues/2181)) ([#2194](https://github.com/Sinity/polylogue/issues/2194)) ([d758d33](https://github.com/Sinity/polylogue/commit/d758d3330922966d55e115d6bc59aea1764475cd))
* **runs:** cite tool and subagent evidence refs ([#2120](https://github.com/Sinity/polylogue/issues/2120)) ([917ad96](https://github.com/Sinity/polylogue/commit/917ad96755ae854d96baf7de2bbb06b6d4008a32)), closes [#1838](https://github.com/Sinity/polylogue/issues/1838) [#1845](https://github.com/Sinity/polylogue/issues/1845) [#1882](https://github.com/Sinity/polylogue/issues/1882)
* **status:** expose archive surface readiness components ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1958](https://github.com/Sinity/polylogue/issues/1958)) ([99bee67](https://github.com/Sinity/polylogue/commit/99bee678f5ddd05324aa9a55541760a112f6af8a))
* **status:** expose assertion and transform readiness ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1976](https://github.com/Sinity/polylogue/issues/1976)) ([cbcc2e1](https://github.com/Sinity/polylogue/commit/cbcc2e1ee9c9cbcd3fbe26bc7836e064b4034c63))
* **status:** expose assertion overlay storage audit ([#2062](https://github.com/Sinity/polylogue/issues/2062)) ([708d6fc](https://github.com/Sinity/polylogue/commit/708d6fc79b7e83fb6586076ca5824501c3034dd8))
* **status:** expose daemon component readiness map ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1960](https://github.com/Sinity/polylogue/issues/1960)) ([57c1024](https://github.com/Sinity/polylogue/commit/57c1024b92168dd6a19389886414d8099e43f0b4))
* **status:** expose embedding component readiness ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1949](https://github.com/Sinity/polylogue/issues/1949)) ([9911af0](https://github.com/Sinity/polylogue/commit/9911af092851f6f36d0682225624f10dbc4b6976))
* **status:** expose search component readiness ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1956](https://github.com/Sinity/polylogue/issues/1956)) ([7f48d9f](https://github.com/Sinity/polylogue/commit/7f48d9f084ad4065292fe88e95472c0e2f0fa1a2))
* **storage:** mirror transform candidates as assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1973](https://github.com/Sinity/polylogue/issues/1973)) ([6800efc](https://github.com/Sinity/polylogue/commit/6800efc095c3cd48b2e854845edb22d1b8e5f758))
* **storage:** persist typed session kind ([#2419](https://github.com/Sinity/polylogue/issues/2419)) ([f2f95f6](https://github.com/Sinity/polylogue/commit/f2f95f6cdc6acfb34b3520f9c7b61dc41b543ec5))
* **storage:** read user overlays through assertions ([#1883](https://github.com/Sinity/polylogue/issues/1883)) ([#1967](https://github.com/Sinity/polylogue/issues/1967)) ([beddd91](https://github.com/Sinity/polylogue/commit/beddd91cef683b57cf0ca9b1f1c7549b98b106ed))
* **telemetry:** project query-unit evidence to OTel ([#2183](https://github.com/Sinity/polylogue/issues/2183)) ([#2215](https://github.com/Sinity/polylogue/issues/2215)) ([40cfd4e](https://github.com/Sinity/polylogue/commit/40cfd4effe6e35ddd32c0369682461cc5626a072))
* **transforms:** add recovery forensic index ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1939](https://github.com/Sinity/polylogue/issues/1939)) ([ebd3165](https://github.com/Sinity/polylogue/commit/ebd31658f640132bd27afc3a30dd67b85d475bd0))
* **transforms:** add recovery report presets ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1945](https://github.com/Sinity/polylogue/issues/1945)) ([264e86b](https://github.com/Sinity/polylogue/commit/264e86b5a3e04ee339a62af96f007e6a4334baf4))
* **transforms:** expose subagent spawn refs ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#1974](https://github.com/Sinity/polylogue/issues/1974)) ([43e5f2a](https://github.com/Sinity/polylogue/commit/43e5f2addd57caafcc52e574df9d3de0f64f94b7))
* **usage:** materialize provider token events ([#2338](https://github.com/Sinity/polylogue/issues/2338)) ([2e24668](https://github.com/Sinity/polylogue/commit/2e24668a5d950284c62f1446cddd241415043862))
* **web:** add evidence workbench read vertical ([#2121](https://github.com/Sinity/polylogue/issues/2121)) ([dcdca02](https://github.com/Sinity/polylogue/commit/dcdca02e324b63e20649b666f16a83174dcb9faa)), closes [#1846](https://github.com/Sinity/polylogue/issues/1846) [#1847](https://github.com/Sinity/polylogue/issues/1847) [#1838](https://github.com/Sinity/polylogue/issues/1838) [#1845](https://github.com/Sinity/polylogue/issues/1845) [#1883](https://github.com/Sinity/polylogue/issues/1883)
* **web:** complete reader operator flow ([#1846](https://github.com/Sinity/polylogue/issues/1846)) ([#2174](https://github.com/Sinity/polylogue/issues/2174)) ([1686a01](https://github.com/Sinity/polylogue/commit/1686a01cf0672a96abb1075b7d412909ad5b5b22))
* **web:** consume component readiness in status strip ([#1832](https://github.com/Sinity/polylogue/issues/1832)) ([#1972](https://github.com/Sinity/polylogue/issues/1972)) ([3f942ad](https://github.com/Sinity/polylogue/commit/3f942adb68b2678c693851c6ab509dba05b13301))
* **web:** execute context preamble read view ([#1846](https://github.com/Sinity/polylogue/issues/1846)) ([#2162](https://github.com/Sinity/polylogue/issues/2162)) ([2ad1fe0](https://github.com/Sinity/polylogue/commit/2ad1fe0036a7745d543064cdde40324de973741b))
* **web:** execute context-pack workbench reads ([#1846](https://github.com/Sinity/polylogue/issues/1846)) ([#2160](https://github.com/Sinity/polylogue/issues/2160)) ([9af1edc](https://github.com/Sinity/polylogue/commit/9af1edc033b0a5e52a0e76a9f32419be33197d95))
* **web:** execute neighbor and correlation read views ([#1846](https://github.com/Sinity/polylogue/issues/1846)) ([#2164](https://github.com/Sinity/polylogue/issues/2164)) ([38cde7c](https://github.com/Sinity/polylogue/commit/38cde7c060252a7cab69e06f93e555e4a0aac42f)), closes [#1847](https://github.com/Sinity/polylogue/issues/1847)
* **web:** execute workbench read views ([#1846](https://github.com/Sinity/polylogue/issues/1846)) ([#2154](https://github.com/Sinity/polylogue/issues/2154)) ([9a2a394](https://github.com/Sinity/polylogue/commit/9a2a394d442abe07087e208d96e749e9dc58f708))
* **web:** expose recovery evidence in reader ([#2115](https://github.com/Sinity/polylogue/issues/2115)) ([9929705](https://github.com/Sinity/polylogue/commit/992970523f7119cba22bba430a7212f2d49edea7)), closes [#1846](https://github.com/Sinity/polylogue/issues/1846) [#1847](https://github.com/Sinity/polylogue/issues/1847)
* **work-packets:** cite commit refs from git tools ([#2117](https://github.com/Sinity/polylogue/issues/2117)) ([d901784](https://github.com/Sinity/polylogue/commit/d9017840cbcf33a3fba23aab56167b7d39cf53c4)), closes [#1838](https://github.com/Sinity/polylogue/issues/1838) [#1845](https://github.com/Sinity/polylogue/issues/1845)


### Fixed

* **agent:** resolve devloop repo roots dynamically ([#2505](https://github.com/Sinity/polylogue/issues/2505)) ([79290e6](https://github.com/Sinity/polylogue/commit/79290e6e6ea718aad5de0bb5d015eeb91c42699a))
* align raw materialization with v8 schema ([#2388](https://github.com/Sinity/polylogue/issues/2388)) ([a49e842](https://github.com/Sinity/polylogue/commit/a49e8424c6f704118f8758fa28774b693d9970aa)), closes [#2308](https://github.com/Sinity/polylogue/issues/2308)
* **archive:** report evidence and convergence honestly ([#2502](https://github.com/Sinity/polylogue/issues/2502)) ([7120bde](https://github.com/Sinity/polylogue/commit/7120bde619a1ffaaef6a82afcba13de904adf80f))
* **assertions:** finish candidate judgment catalogs ([#2182](https://github.com/Sinity/polylogue/issues/2182)) ([#2205](https://github.com/Sinity/polylogue/issues/2205)) ([1fb8ace](https://github.com/Sinity/polylogue/commit/1fb8aceaf5f1c282e805a6fd289f3bf94e98e850))
* **backup:** report missing referenced blob debt ([#2422](https://github.com/Sinity/polylogue/issues/2422)) ([49e1ddf](https://github.com/Sinity/polylogue/commit/49e1ddfa91e71753fbbaac5394939260d308c938))
* bound daemon archive read contention ([#2416](https://github.com/Sinity/polylogue/issues/2416)) ([f3e64a2](https://github.com/Sinity/polylogue/commit/f3e64a2fb7fdf3dddb0b1b8d918896b241056ab2))
* **browser-capture:** fetch ChatGPT native payloads ([#2375](https://github.com/Sinity/polylogue/issues/2375)) ([703d434](https://github.com/Sinity/polylogue/commit/703d4348acb533b309f5cee6c71ada0b737c234d))
* **browser-capture:** hide local paths from receiver DTOs ([#1847](https://github.com/Sinity/polylogue/issues/1847)) ([#2153](https://github.com/Sinity/polylogue/issues/2153)) ([b03d4ea](https://github.com/Sinity/polylogue/commit/b03d4ea395ce60143706676ab9a966a6ed3f6ccc))
* **browser-capture:** prefer native Claude payloads ([#2371](https://github.com/Sinity/polylogue/issues/2371)) ([c4b3c08](https://github.com/Sinity/polylogue/commit/c4b3c08b6f2f47b0c65a6c223941540326260314)), closes [#2308](https://github.com/Sinity/polylogue/issues/2308) [#2248](https://github.com/Sinity/polylogue/issues/2248)
* **browser-capture:** preserve temporary chats and attachments ([#2376](https://github.com/Sinity/polylogue/issues/2376)) ([80b3af4](https://github.com/Sinity/polylogue/commit/80b3af456ceb7b80bb1e91ecffcc23b80bb84837))
* **browser-capture:** recapture existing provider tabs ([#2377](https://github.com/Sinity/polylogue/issues/2377)) ([42a7555](https://github.com/Sinity/polylogue/commit/42a755504e9fb00b9a68515acf73daab164f4533))
* **browser:** avoid duplicate capture id provider prefixes ([d93acd1](https://github.com/Sinity/polylogue/commit/d93acd1adb8218cd8b3449445c2a08d771c5efbb))
* **browser:** harden capture status evidence ([#2326](https://github.com/Sinity/polylogue/issues/2326)) ([d981009](https://github.com/Sinity/polylogue/commit/d9810096ae3a3bafbbdd92e8967c615df3543c74)), closes [#2308](https://github.com/Sinity/polylogue/issues/2308)
* **browser:** harden live provider proof harness ([#2346](https://github.com/Sinity/polylogue/issues/2346)) ([a12dfb8](https://github.com/Sinity/polylogue/commit/a12dfb8d8a5a47b75cfc94f4f6658b7927abb9ea))
* **browser:** inject scripts from popup capture ([#2348](https://github.com/Sinity/polylogue/issues/2348)) ([02779ca](https://github.com/Sinity/polylogue/commit/02779cae6f3013dbf894fe1ae4a211ea35309f4a))
* **browser:** repair live capture convergence ([#2345](https://github.com/Sinity/polylogue/issues/2345)) ([ddf4f3e](https://github.com/Sinity/polylogue/commit/ddf4f3efc7060f202ec9c7bfe4a7aaf4af514216))
* **browser:** type temporary capture sessions ([#2393](https://github.com/Sinity/polylogue/issues/2393)) ([788e263](https://github.com/Sinity/polylogue/commit/788e26373b51b775bd8bd8faf79ca22b023a48eb)), closes [#2308](https://github.com/Sinity/polylogue/issues/2308)
* **cli:** accept delete action JSON format ([#2359](https://github.com/Sinity/polylogue/issues/2359)) ([d904285](https://github.com/Sinity/polylogue/commit/d904285af9abf09ab03240ccc467e9f31df1247d))
* **cli:** accept post-verb json output alias ([#2366](https://github.com/Sinity/polylogue/issues/2366)) ([ea58cf0](https://github.com/Sinity/polylogue/commit/ea58cf02f8424e0d0e68efe5d6cfc6340b88c742))
* **cli:** bound delete dry-run previews ([#2357](https://github.com/Sinity/polylogue/issues/2357)) ([ef19c65](https://github.com/Sinity/polylogue/commit/ef19c650709fc6b1b413bb3e27345108f48f806a))
* **cli:** defer expensive facet families by default ([#2343](https://github.com/Sinity/polylogue/issues/2343)) ([43f5a7c](https://github.com/Sinity/polylogue/commit/43f5a7ca37d47c78c24589dc4e68f4ab8f3fd4c6)), closes [#2317](https://github.com/Sinity/polylogue/issues/2317)
* **cli:** emit mark mutation payloads ([#2355](https://github.com/Sinity/polylogue/issues/2355)) ([3e5fda5](https://github.com/Sinity/polylogue/commit/3e5fda5a33721df69b48a6562686add11f379de1))
* **cli:** enforce read-view format profiles ([#1838](https://github.com/Sinity/polylogue/issues/1838)) ([#2099](https://github.com/Sinity/polylogue/issues/2099)) ([371657a](https://github.com/Sinity/polylogue/commit/371657aee64c2269491e7f3c3cb6e6e3107e1483))
* **cli:** explain query terminal actions ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2286](https://github.com/Sinity/polylogue/issues/2286)) ([e94b345](https://github.com/Sinity/polylogue/commit/e94b3453d194c112e3534dc07b960da6c06be97c))
* **cli:** make onboarding surfaces honest ([#2340](https://github.com/Sinity/polylogue/issues/2340)) ([e4a4c8b](https://github.com/Sinity/polylogue/commit/e4a4c8b6ed9260621d246d46fcf244a9fe5359c2))
* **cli:** parse quoted shell completion words ([#2337](https://github.com/Sinity/polylogue/issues/2337)) ([13d18ff](https://github.com/Sinity/polylogue/commit/13d18ff5b2b6b03f5347c533450ba702a99c99ac))
* **cli:** point root status to ops status ([#2332](https://github.com/Sinity/polylogue/issues/2332)) ([c28ee9a](https://github.com/Sinity/polylogue/commit/c28ee9a61fb3cbb6f60fb34c455a94f725c80bee))
* **cli:** read exact refs as selected sessions ([#2342](https://github.com/Sinity/polylogue/issues/2342)) ([06dd766](https://github.com/Sinity/polylogue/commit/06dd7667a18689dc3e6b71aa0119cd6c06582a11)), closes [#2317](https://github.com/Sinity/polylogue/issues/2317)
* **cli:** reject ambiguous plain select ([#2356](https://github.com/Sinity/polylogue/issues/2356)) ([5ab1904](https://github.com/Sinity/polylogue/commit/5ab19043629b98539892eece726458b254894d38))
* **cli:** reject read options for unrelated views ([#2219](https://github.com/Sinity/polylogue/issues/2219)) ([1644326](https://github.com/Sinity/polylogue/commit/1644326f659b82a2615d0cfaab484598e3fd91a7))
* **cli:** render run query-unit rows ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2214](https://github.com/Sinity/polylogue/issues/2214)) ([5909e7d](https://github.com/Sinity/polylogue/commit/5909e7d586d4b1448b030728cd2d970fe1ca1620))
* **cli:** require explicit delete confirmation ([#2354](https://github.com/Sinity/polylogue/issues/2354)) ([1261a4c](https://github.com/Sinity/polylogue/commit/1261a4c0a7386c99231983ce26e8fed2a863509b))
* **cli:** require singleton continue actions ([#2353](https://github.com/Sinity/polylogue/issues/2353)) ([03af293](https://github.com/Sinity/polylogue/commit/03af29383a3dffb60a24092d8bcd3afb297a4f97))
* **cli:** restore command-surface baseline coverage ([#2253](https://github.com/Sinity/polylogue/issues/2253)) ([#2283](https://github.com/Sinity/polylogue/issues/2283)) ([838bc0e](https://github.com/Sinity/polylogue/commit/838bc0e3a0b60b859537870c250106bf2abcb530))
* **cli:** route recovery reads through archive facade ([#1880](https://github.com/Sinity/polylogue/issues/1880)) ([#2057](https://github.com/Sinity/polylogue/issues/2057)) ([0cea5c3](https://github.com/Sinity/polylogue/commit/0cea5c37c79e4be7e00b09f85be6110aa44bd1b3))
* **cli:** support first-match read actions ([#2352](https://github.com/Sinity/polylogue/issues/2352)) ([115ebca](https://github.com/Sinity/polylogue/commit/115ebca0f42e8d06807c8440c082fab28458ca9f))
* **config:** audit env-provided path diagnostics ([#2344](https://github.com/Sinity/polylogue/issues/2344)) ([d70b0d3](https://github.com/Sinity/polylogue/commit/d70b0d3fb7eeec2606d8456368611dfeedee2e5c))
* **config:** report effective configuration debt ([#2334](https://github.com/Sinity/polylogue/issues/2334)) ([fc70163](https://github.com/Sinity/polylogue/commit/fc70163adaa2d188150cb8748a168cfbfdc0fdb0))
* **cost:** bill subscription output credits at 5x input rate ([#2484](https://github.com/Sinity/polylogue/issues/2484)) ([3c1bbb3](https://github.com/Sinity/polylogue/commit/3c1bbb3f3248a2013edb3d8933539add2587723c))
* **cost:** count Codex cumulative tokens per session, not per model ([#2489](https://github.com/Sinity/polylogue/issues/2489)) ([22d8bb5](https://github.com/Sinity/polylogue/commit/22d8bb50789d6ebec0fe59ae26eec16920c506d0))
* **cost:** flag paid models missing a catalog cache rate instead of $0 ([#2487](https://github.com/Sinity/polylogue/issues/2487)) ([e851b33](https://github.com/Sinity/polylogue/commit/e851b33deed3a79f560704c00104349098fcd6e1))
* **daemon:** apply message projection flags in reader API ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2043](https://github.com/Sinity/polylogue/issues/2043)) ([9f23294](https://github.com/Sinity/polylogue/commit/9f23294828e7319430dc7c0b6d499a621e844380))
* **daemon:** attribute append ingest storage route ([#2360](https://github.com/Sinity/polylogue/issues/2360)) ([9e6b259](https://github.com/Sinity/polylogue/commit/9e6b259e29c96b5ba2f44d7a2ac6931560770c95))
* **daemon:** preserve inbox import suffixes ([#1843](https://github.com/Sinity/polylogue/issues/1843)) ([#1953](https://github.com/Sinity/polylogue/issues/1953)) ([23d9ae7](https://github.com/Sinity/polylogue/commit/23d9ae77c4bdd1cb697939f9ad3836877d08e677))
* **daemon:** preserve live stage telemetry payloads ([#2331](https://github.com/Sinity/polylogue/issues/2331)) ([63309f6](https://github.com/Sinity/polylogue/commit/63309f6a806befb0d221c8bc91fc09f5ea7f6829))
* **daemon:** report live ingest read amplification ([#2323](https://github.com/Sinity/polylogue/issues/2323)) ([6cef737](https://github.com/Sinity/polylogue/commit/6cef7374015512b8fe3a36d4de191cf9536f0e5d)), closes [#2308](https://github.com/Sinity/polylogue/issues/2308)
* **daemon:** restore Drive source catch-up ([#1994](https://github.com/Sinity/polylogue/issues/1994)) ([f3bbd26](https://github.com/Sinity/polylogue/commit/f3bbd26fc409bbe11691b34c61fec59c0a961fb1))
* **daemon:** resume large live files from archived prefixes ([#2167](https://github.com/Sinity/polylogue/issues/2167)) ([#2168](https://github.com/Sinity/polylogue/issues/2168)) ([232ab92](https://github.com/Sinity/polylogue/commit/232ab92e4d0c62b9f8735aca5b6ac8a24a340351))
* debounce same-path watcher events ([#2408](https://github.com/Sinity/polylogue/issues/2408)) ([3b823de](https://github.com/Sinity/polylogue/commit/3b823de5224d4cfc1e299c10926b1558c308818a))
* **dev-loop:** prefer Chromium for extension smokes ([#2373](https://github.com/Sinity/polylogue/issues/2373)) ([2bfbe32](https://github.com/Sinity/polylogue/commit/2bfbe32fb116bee488104b4bf14d31c3eaa7844f))
* **dev-loop:** preserve provider smoke browser evidence ([#2372](https://github.com/Sinity/polylogue/issues/2372)) ([7b59d4d](https://github.com/Sinity/polylogue/commit/7b59d4dfbc925daee749b935d5370dd24311eaaa))
* **dev:** compose dev-loop browser artifacts ([#2248](https://github.com/Sinity/polylogue/issues/2248)) ([#2279](https://github.com/Sinity/polylogue/issues/2279)) ([c6f0820](https://github.com/Sinity/polylogue/commit/c6f0820c268b7d74df4ca41db458af82bccebf0b))
* **devtools:** align browser plan with Chromium smokes ([#2374](https://github.com/Sinity/polylogue/issues/2374)) ([c6c1aa6](https://github.com/Sinity/polylogue/commit/c6c1aa659633d22600b5f206f3de4a3c21d589b1))
* **devtools:** bound workload diagnostics and cleanup pytest temps ([#2163](https://github.com/Sinity/polylogue/issues/2163)) ([3e8ebc2](https://github.com/Sinity/polylogue/commit/3e8ebc2a921c42cc7699cff580dc71d615d9856b))
* **devtools:** clear stale pytest reports on verify timeout ([#1807](https://github.com/Sinity/polylogue/issues/1807)) ([#2090](https://github.com/Sinity/polylogue/issues/2090)) ([4f37816](https://github.com/Sinity/polylogue/commit/4f37816892b2f2b084c7f8a930876e8ee1dd2a8c))
* **devtools:** detect squash-equivalent worktrees ([#2501](https://github.com/Sinity/polylogue/issues/2501)) ([f41e078](https://github.com/Sinity/polylogue/commit/f41e078903edd6327459a609364aeb9755f6a6ac))
* **devtools:** expose live pytest verify progress ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2074](https://github.com/Sinity/polylogue/issues/2074)) ([28f7903](https://github.com/Sinity/polylogue/commit/28f79030e033bb1ff00d9672eb7a4f5ef0f4d596))
* **devtools:** expose pytest collection and phase timing ([#2109](https://github.com/Sinity/polylogue/issues/2109)) ([fd7afbe](https://github.com/Sinity/polylogue/commit/fd7afbe72111cd481ad4038b80738619bd1ff697))
* **devtools:** expose pytest selection during verify ([#2107](https://github.com/Sinity/polylogue/issues/2107)) ([dfe57dc](https://github.com/Sinity/polylogue/commit/dfe57dc0393929e87a592c911158546e9a500299))
* **devtools:** make verify runs observable and bounded ([#2110](https://github.com/Sinity/polylogue/issues/2110)) ([#2111](https://github.com/Sinity/polylogue/issues/2111)) ([4b41f13](https://github.com/Sinity/polylogue/commit/4b41f1306185619fa2ccb876ee0eac3bd04a559d))
* **devtools:** pin dev-loop daemon to worktree source ([90f04a1](https://github.com/Sinity/polylogue/commit/90f04a1a781bbc272ac72bfbbd4e7fc8fa184b5d))
* **devtools:** recognize repeated browser captures ([#2347](https://github.com/Sinity/polylogue/issues/2347)) ([5327c12](https://github.com/Sinity/polylogue/commit/5327c1279ce1eff7832bdb60bb2f9391cfd15e9e))
* **devtools:** reject stale testmon verify seeds ([#2103](https://github.com/Sinity/polylogue/issues/2103)) ([6b117cb](https://github.com/Sinity/polylogue/commit/6b117cb616f3f642a76e2331dfd0b72926acad2e))
* **devtools:** report active pytest node during verify ([#2100](https://github.com/Sinity/polylogue/issues/2100)) ([28e95ab](https://github.com/Sinity/polylogue/commit/28e95ab26d5e1db9a01b313eb9f662898d603607))
* **devtools:** run verify baseline from grouped command ([#2149](https://github.com/Sinity/polylogue/issues/2149)) ([26991ba](https://github.com/Sinity/polylogue/commit/26991ba79fabf9c8816be1dd3c659d63106d6497))
* **devtools:** stream pytest progress during verify ([#2073](https://github.com/Sinity/polylogue/issues/2073)) ([43e088e](https://github.com/Sinity/polylogue/commit/43e088e014ab67a3893c931f94462dcd7a4ba821))
* **devtools:** treat completed browser artifacts as smoke success ([#2349](https://github.com/Sinity/polylogue/issues/2349)) ([3198f22](https://github.com/Sinity/polylogue/commit/3198f22d71572af6ac4355d3f7bffc0341b9d4e8))
* **devtools:** trust indexed browser captures in smoke ([#2428](https://github.com/Sinity/polylogue/issues/2428)) ([62f3e53](https://github.com/Sinity/polylogue/commit/62f3e532e9664906baac8ec533c36175e65845b2))
* **devtools:** use block rows for FTS gap plans ([#2362](https://github.com/Sinity/polylogue/issues/2362)) ([17a017c](https://github.com/Sinity/polylogue/commit/17a017c8e3b981e7ee82ad194c453856d0078f72))
* **devtools:** validate release PR evidence bodies ([#2119](https://github.com/Sinity/polylogue/issues/2119)) ([214702b](https://github.com/Sinity/polylogue/commit/214702b3576dd31c28e7ca57482aaf82f217641d))
* **docs:** repair generated pages links ([#2500](https://github.com/Sinity/polylogue/issues/2500)) ([a9985f4](https://github.com/Sinity/polylogue/commit/a9985f4115d057c7de0842c1a23414492b3c1efe))
* **embed:** expose preflight min-message floor ([#2506](https://github.com/Sinity/polylogue/issues/2506)) ([12e7cdf](https://github.com/Sinity/polylogue/commit/12e7cdfd2fa877b8c094d5a4560c68e3ceb7dc94))
* **embed:** keep backfill windows under message cap ([#2516](https://github.com/Sinity/polylogue/issues/2516)) ([18df4c4](https://github.com/Sinity/polylogue/commit/18df4c457445c016d29c7a1ceecbcfcfa1c4624c))
* **embed:** keep partial vectors retrieval-ready ([#2515](https://github.com/Sinity/polylogue/issues/2515)) ([aa13e1f](https://github.com/Sinity/polylogue/commit/aa13e1f73f85eefe347d9f8869d0a2e98825c25b))
* **embed:** treat completed sessions as embedded ([#2514](https://github.com/Sinity/polylogue/issues/2514)) ([6cd054d](https://github.com/Sinity/polylogue/commit/6cd054d652cffcdf27860e769e9a15451cddd08b))
* ensure runtime indexes in batch ingest ([#2418](https://github.com/Sinity/polylogue/issues/2418)) ([64eeb41](https://github.com/Sinity/polylogue/commit/64eeb4151e975dceb387905bfa9f8367acfd9fd4))
* fail deployment smoke on route errors ([#2411](https://github.com/Sinity/polylogue/issues/2411)) ([4fe7f10](https://github.com/Sinity/polylogue/commit/4fe7f10abbcbdb913ab09bb1dff166aa927fad67))
* **import:** preflight unsupported sources truthfully ([#1815](https://github.com/Sinity/polylogue/issues/1815)) ([#1969](https://github.com/Sinity/polylogue/issues/1969)) ([0c4a348](https://github.com/Sinity/polylogue/commit/0c4a34842214fbaea49eef0ef81c8d0d40757188))
* **insights:** keep runtime user rows out of authored analysis ([#2327](https://github.com/Sinity/polylogue/issues/2327)) ([23cbf87](https://github.com/Sinity/polylogue/commit/23cbf877d28aeab7786dd6d84b17b320613f993e)), closes [#2318](https://github.com/Sinity/polylogue/issues/2318)
* **insights:** restore facade thread rebuild parity ([#2191](https://github.com/Sinity/polylogue/issues/2191)) ([861a84d](https://github.com/Sinity/polylogue/commit/861a84dbdfe7426313cadfc39402a4c55f952fb1)), closes [#2188](https://github.com/Sinity/polylogue/issues/2188)
* **insights:** upsert run-projection rows on cross-session ref collisions ([#2464](https://github.com/Sinity/polylogue/issues/2464)) ([15f4f21](https://github.com/Sinity/polylogue/commit/15f4f21b63f662f13fb4f16b5379d7dfaad17627))
* **maintenance:** classify blob reference debt ([#2423](https://github.com/Sinity/polylogue/issues/2423)) ([8616670](https://github.com/Sinity/polylogue/commit/86166704e61dbd4eec02a6540a46a12552684803))
* **maintenance:** ignore parsed raw sidecars ([#2511](https://github.com/Sinity/polylogue/issues/2511)) ([1d5d112](https://github.com/Sinity/polylogue/commit/1d5d1126943648d18bfcb2c55520942e0f807347))
* **maintenance:** plan raw-backed blob recovery ([#2426](https://github.com/Sinity/polylogue/issues/2426)) ([2820f3c](https://github.com/Sinity/polylogue/commit/2820f3c75a296c8466810317d99106631198b03a))
* **maintenance:** quarantine stale orphan blob refs ([#2425](https://github.com/Sinity/polylogue/issues/2425)) ([674bd07](https://github.com/Sinity/polylogue/commit/674bd076794df322b9edbc9c346d9fc430608256))
* **maintenance:** replace stale raw blobs from source ([#2427](https://github.com/Sinity/polylogue/issues/2427)) ([688abdf](https://github.com/Sinity/polylogue/commit/688abdf7fe7586b6ee07001a24276fec16ee3859))
* **maintenance:** replay parsed raw rows after index reset ([#2508](https://github.com/Sinity/polylogue/issues/2508)) ([d1bbb3e](https://github.com/Sinity/polylogue/commit/d1bbb3e02bfb09d55ba0e2ef6a1f69be5db37408))
* **maintenance:** resume interrupted raw materialization ([#2510](https://github.com/Sinity/polylogue/issues/2510)) ([b55ea52](https://github.com/Sinity/polylogue/commit/b55ea528e9c3d856c9ef2bf144b8ccba5a1e01b3))
* make deployed Polylogue smoke trustworthy ([#2303](https://github.com/Sinity/polylogue/issues/2303)) ([94a8bfb](https://github.com/Sinity/polylogue/commit/94a8bfb949508b01abd85d10d94dbaf2c3eecc3a))
* make raw materialization replay auditable ([#2387](https://github.com/Sinity/polylogue/issues/2387)) ([3f48928](https://github.com/Sinity/polylogue/commit/3f48928c2b16e192ed773af4df8bb57cb1738a56)), closes [#2308](https://github.com/Sinity/polylogue/issues/2308)
* **mcp:** apply projection flags to message reads ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#1954](https://github.com/Sinity/polylogue/issues/1954)) ([8d46985](https://github.com/Sinity/polylogue/commit/8d46985b2ea88595311047ee4c6be8a8d0415ac4))
* **mcp:** expose shared error envelope core ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2044](https://github.com/Sinity/polylogue/issues/2044)) ([f666000](https://github.com/Sinity/polylogue/commit/f666000f1d4ca7ff34d2dccf34aaf2ebeacb4b8c))
* **ops:** restore runtime materialization trust ([#2351](https://github.com/Sinity/polylogue/issues/2351)) ([373f9e3](https://github.com/Sinity/polylogue/commit/373f9e3c34fc747ee83604f3e75bf6e43e983a1f))
* **ops:** surface raw materialization debt ([4fe0400](https://github.com/Sinity/polylogue/commit/4fe0400f17ac59dc52be4d51c67267b801dc044d))
* package diagnostics runtime surface ([#2320](https://github.com/Sinity/polylogue/issues/2320)) ([b974548](https://github.com/Sinity/polylogue/commit/b974548e64fe7207b7c68a45b5a72d4e3d6fac0f))
* prevent silent raw materialization gaps ([#2321](https://github.com/Sinity/polylogue/issues/2321)) ([c38b6b2](https://github.com/Sinity/polylogue/commit/c38b6b29e3e6bd554cd231134c6887347b5f1248)), closes [#2308](https://github.com/Sinity/polylogue/issues/2308)
* **query:** accept session refs in find workflows ([#2330](https://github.com/Sinity/polylogue/issues/2330)) ([b2fa65e](https://github.com/Sinity/polylogue/commit/b2fa65e8f07b44c1aa7dbdce466a5f5410f7607f)), closes [#2305](https://github.com/Sinity/polylogue/issues/2305)
* **query:** bind runtime unit fields to descriptors ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2272](https://github.com/Sinity/polylogue/issues/2272)) ([68dbaf5](https://github.com/Sinity/polylogue/commit/68dbaf516e4b149f775596e6f1687ffca40d333f))
* **query:** dispatch unit envelopes from descriptors ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2223](https://github.com/Sinity/polylogue/issues/2223)) ([4ebab7c](https://github.com/Sinity/polylogue/commit/4ebab7c98ed103c604f233e0c30a31ee2456400c))
* **query:** enforce adapter query parity across MCP and daemon ([#1825](https://github.com/Sinity/polylogue/issues/1825)) ([#2042](https://github.com/Sinity/polylogue/issues/2042)) ([1524fd5](https://github.com/Sinity/polylogue/commit/1524fd5a26d04b1192df31cb00e301b3b1235027))
* **query:** execute shell-quoted unit-source actions ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2288](https://github.com/Sinity/polylogue/issues/2288)) ([6455a9d](https://github.com/Sinity/polylogue/commit/6455a9d3ba69d31122f5f63ade613d80b73792f7))
* **query:** explain runtime terminal unit lowering ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2165](https://github.com/Sinity/polylogue/issues/2165)) ([eece58e](https://github.com/Sinity/polylogue/commit/eece58e726c4ef908e44cd2ceb1df62f50d68362))
* **query:** preserve exclusive comparison operators ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2273](https://github.com/Sinity/polylogue/issues/2273)) ([9500110](https://github.com/Sinity/polylogue/commit/9500110b46767742f8d3d0e38fbb77d4fa804fbb))
* **query:** reject terminal-only unit selectors ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2161](https://github.com/Sinity/polylogue/issues/2161)) ([05feeed](https://github.com/Sinity/polylogue/commit/05feeed00887e52ea9f53e2ad9a35682bfe10354))
* **query:** require bound field refs during execution ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2285](https://github.com/Sinity/polylogue/issues/2285)) ([7824a59](https://github.com/Sinity/polylogue/commit/7824a599e5147f9cfe55877c9b50d5d57bc5858b))
* **query:** route unit execution by descriptor lowerer ([#2006](https://github.com/Sinity/polylogue/issues/2006)) ([#2221](https://github.com/Sinity/polylogue/issues/2221)) ([24aa877](https://github.com/Sinity/polylogue/commit/24aa877064ebe922ba470f2512489af2b184035f))
* reacquire missing raw blob evidence ([#2390](https://github.com/Sinity/polylogue/issues/2390)) ([cedb481](https://github.com/Sinity/polylogue/commit/cedb481b1571de09660ee5c43383e3aa00c74dc8))
* retarget query DSL planning and restore quick gate ([#2007](https://github.com/Sinity/polylogue/issues/2007)) ([123c84e](https://github.com/Sinity/polylogue/commit/123c84e28ae9ae3c43db991d0e611b335996732c))
* **scenarios:** keep demo fixture tool paths relative ([#1843](https://github.com/Sinity/polylogue/issues/1843)) ([#1937](https://github.com/Sinity/polylogue/issues/1937)) ([716c19c](https://github.com/Sinity/polylogue/commit/716c19cf15e6690e292b828334bde1d6b93378c1))
* **schema:** scope action result joins by session ([#2247](https://github.com/Sinity/polylogue/issues/2247)) ([e42cd49](https://github.com/Sinity/polylogue/commit/e42cd49df598d8336ebda98af94bcc70eb3c1888))
* snapshot exact FTS readiness checks ([#2389](https://github.com/Sinity/polylogue/issues/2389)) ([346e166](https://github.com/Sinity/polylogue/commit/346e166040880577a8d4e2601b555e10a0df32f8)), closes [#2308](https://github.com/Sinity/polylogue/issues/2308)
* stabilize live Polylogue dogfooding loop ([#2319](https://github.com/Sinity/polylogue/issues/2319)) ([6049c77](https://github.com/Sinity/polylogue/commit/6049c778615a12825eb6fd878652533521a9ef52))
* **storage:** compose forks on paginated, batch, and streaming reads ([#2485](https://github.com/Sinity/polylogue/issues/2485)) ([8664ee8](https://github.com/Sinity/polylogue/commit/8664ee872286811f5094e125cbcf78559c6b7499)), closes [#2470](https://github.com/Sinity/polylogue/issues/2470)
* **storage:** distrust FTS freshness without triggers ([#2333](https://github.com/Sinity/polylogue/issues/2333)) ([2afc28b](https://github.com/Sinity/polylogue/commit/2afc28b14e6f77ec659c301bc2b74be93bb4f80f))
* **storage:** remove blob refs with retained raw snapshots ([#2424](https://github.com/Sinity/polylogue/issues/2424)) ([71b395c](https://github.com/Sinity/polylogue/commit/71b395c3d0d91031f44d9447c938e810e35213b3))
* **test:** refresh status snapshot for index schema v10 ([#2430](https://github.com/Sinity/polylogue/issues/2430)) ([52532aa](https://github.com/Sinity/polylogue/commit/52532aa3fa013b7d1d959e1bd9fc4a0d7c9e3da5))
* **test:** refresh terminal snapshots for raw_materialization + facets ([#2432](https://github.com/Sinity/polylogue/issues/2432)) ([1c18ed2](https://github.com/Sinity/polylogue/commit/1c18ed29c4c536f1a7541b964925a9b8a17a2ecb))
* **tests:** restore baseline after surface moves ([#2151](https://github.com/Sinity/polylogue/issues/2151)) ([bea1bb4](https://github.com/Sinity/polylogue/commit/bea1bb42e690f0364ee8ab816ce570615f57db47))
* **tests:** stop default pytest basetemps from using shm ([#1995](https://github.com/Sinity/polylogue/issues/1995)) ([8647ceb](https://github.com/Sinity/polylogue/commit/8647cebb9a4386333877be004f0724b55aaa4fcb))
* **usage:** expose codex zero-token projection debt ([#2336](https://github.com/Sinity/polylogue/issues/2336)) ([df0f8f8](https://github.com/Sinity/polylogue/commit/df0f8f82cc465129b6b7260e49096c8d32397ecb))
* **web:** defer expensive archive facets ([44938c3](https://github.com/Sinity/polylogue/commit/44938c3752b354dfe41ff6b88e4b6971f59d4604))
* **web:** stop archive session queries after client abort ([#2335](https://github.com/Sinity/polylogue/issues/2335)) ([cd67316](https://github.com/Sinity/polylogue/commit/cd67316cef2f1a5781090687585a772a033ac688))


### Changed

* avoid full-session active-leaf churn ([#2394](https://github.com/Sinity/polylogue/issues/2394)) ([b4aa678](https://github.com/Sinity/polylogue/commit/b4aa67801e8d5895d328ce2b630dd0bfa12120f0)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* classify agent control actions ([#2407](https://github.com/Sinity/polylogue/issues/2407)) ([fb04b3a](https://github.com/Sinity/polylogue/commit/fb04b3a46c80cf1d2f7e26ce1cf35ef928994f6a))
* **daemon:** allow skipping configured source catch-up ([#2404](https://github.com/Sinity/polylogue/issues/2404)) ([f84ee53](https://github.com/Sinity/polylogue/commit/f84ee53f22ea8898c98166d19cbfac6317a97fc8))
* **daemon:** bound metrics route counting ([#2363](https://github.com/Sinity/polylogue/issues/2363)) ([a584bd2](https://github.com/Sinity/polylogue/commit/a584bd246e31ce5034389a3d505ca29f4df5e9a4))
* **daemon:** bound SQLite cache pressure ([#2172](https://github.com/Sinity/polylogue/issues/2172)) ([#2173](https://github.com/Sinity/polylogue/issues/2173)) ([0227d0f](https://github.com/Sinity/polylogue/commit/0227d0fe7bbfc8a01a48b072be9ba13979846249))
* **devtools:** bound pytest harness memory pressure ([#2170](https://github.com/Sinity/polylogue/issues/2170)) ([#2171](https://github.com/Sinity/polylogue/issues/2171)) ([d150845](https://github.com/Sinity/polylogue/commit/d15084515ca75a5c417550b87edd743589376de2))
* **embed:** avoid exact scans during window selection ([#2512](https://github.com/Sinity/polylogue/issues/2512)) ([ba4655a](https://github.com/Sinity/polylogue/commit/ba4655afc2040bf0f59e015331032a487511cba5))
* **embed:** constrain archive block text lookup ([#2513](https://github.com/Sinity/polylogue/issues/2513)) ([da18955](https://github.com/Sinity/polylogue/commit/da18955963ac32bc0c4f88443a0926d7a4abdb82))
* **embed:** index authored prose selection ([#2507](https://github.com/Sinity/polylogue/issues/2507)) ([cdd0081](https://github.com/Sinity/polylogue/commit/cdd008167b4cfe03d08231b0200420b4353df957))
* expose append ingest stage timings ([#2395](https://github.com/Sinity/polylogue/issues/2395)) ([0adb353](https://github.com/Sinity/polylogue/commit/0adb353d53853263317849be508ca95206b17416)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* expose deep append index timings ([#2397](https://github.com/Sinity/polylogue/issues/2397)) ([8e94c6d](https://github.com/Sinity/polylogue/commit/8e94c6d1504214c8b650a026574e2bf02f4eb444)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* expose full ingest stage timings ([#2399](https://github.com/Sinity/polylogue/issues/2399)) ([34a68df](https://github.com/Sinity/polylogue/commit/34a68dfab02bfc709fccbd467caf4aca9f480e4d))
* expose insight profile timing ([#2403](https://github.com/Sinity/polylogue/issues/2403)) ([947c8e1](https://github.com/Sinity/polylogue/commit/947c8e1ffaeeee3ef785e3dc87d0143d50402104))
* expose insight refresh stage timings ([#2402](https://github.com/Sinity/polylogue/issues/2402)) ([f9ae43d](https://github.com/Sinity/polylogue/commit/f9ae43dfcca695966abc97f8e77ad29123efe5c1)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* expose live ingest stage timings ([#2392](https://github.com/Sinity/polylogue/issues/2392)) ([5c82385](https://github.com/Sinity/polylogue/commit/5c82385ec6adf834ad3fe7b21ca30750f3989abf)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* expose ops stage timing diagnostics ([#2401](https://github.com/Sinity/polylogue/issues/2401)) ([45a2c0b](https://github.com/Sinity/polylogue/commit/45a2c0bda331851362bc071249aea31cda3be1dc)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* expose semantic fact stage timings ([#2405](https://github.com/Sinity/polylogue/issues/2405)) ([5a18cf9](https://github.com/Sinity/polylogue/commit/5a18cf9880bb488675dfebf090e1ab16b5fd6807))
* fold provider usage on append ([#2414](https://github.com/Sinity/polylogue/issues/2414)) ([4a0cfdc](https://github.com/Sinity/polylogue/commit/4a0cfdca4b9307ddbb057a58aec413e62fcd6303))
* increment session counts on append ([#2412](https://github.com/Sinity/polylogue/issues/2412)) ([ea85fb1](https://github.com/Sinity/polylogue/commit/ea85fb1dfb433362aef1bcdae116e55888445a8c))
* index append active-leaf clearing ([#2417](https://github.com/Sinity/polylogue/issues/2417)) ([01a51bb](https://github.com/Sinity/polylogue/commit/01a51bbdc1793a932dde9659bdaba1d2fa0a7a21))
* **ingest:** batch commits by accumulated message count ([#2492](https://github.com/Sinity/polylogue/issues/2492)) ([eaa4ad7](https://github.com/Sinity/polylogue/commit/eaa4ad7e23ac9238291b9d1902f9c13ec7963432)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* **insights:** bound insight-rebuild WAL via per-chunk commits (Ref [#2458](https://github.com/Sinity/polylogue/issues/2458)) ([#2466](https://github.com/Sinity/polylogue/issues/2466)) ([2eee22a](https://github.com/Sinity/polylogue/commit/2eee22a9fbfe9f177bc895f64c3271917044a91b))
* **maintenance:** batch raw materialization replay ([#2509](https://github.com/Sinity/polylogue/issues/2509)) ([4a39c0c](https://github.com/Sinity/polylogue/commit/4a39c0c31fa07d76e314773cf69a7e191ff474d9))
* reduce profile cost projection work ([#2406](https://github.com/Sinity/polylogue/issues/2406)) ([b55b904](https://github.com/Sinity/polylogue/commit/b55b904c494af5a5b4a5d5e0e81708b8188228ab))
* scope append attachment refcounts ([#2398](https://github.com/Sinity/polylogue/issues/2398)) ([af63ee9](https://github.com/Sinity/polylogue/commit/af63ee9911bb9d60d7b0a458e204796f40a26d9d)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* skip append cost aggregate without tokens ([#2415](https://github.com/Sinity/polylogue/issues/2415)) ([b0ad5d9](https://github.com/Sinity/polylogue/commit/b0ad5d93bf78a430ac63df28413697ccd13caeb1))
* skip provider usage rollup on plain appends ([#2410](https://github.com/Sinity/polylogue/issues/2410)) ([283d213](https://github.com/Sinity/polylogue/commit/283d213080eef5016ff26d678b56514e3be66a13))
* split append storage timing ([#2396](https://github.com/Sinity/polylogue/issues/2396)) ([77af593](https://github.com/Sinity/polylogue/commit/77af593a24dc73cc03e54078a4f69c88763078ab)), closes [#2391](https://github.com/Sinity/polylogue/issues/2391)
* split full replace ingest timings ([#2400](https://github.com/Sinity/polylogue/issues/2400)) ([faaafa0](https://github.com/Sinity/polylogue/commit/faaafa08d0580db3daa2d41c04be251bba0c831e))
* **storage:** index message role facets ([#2420](https://github.com/Sinity/polylogue/issues/2420)) ([a13215b](https://github.com/Sinity/polylogue/commit/a13215b7cd070bec1fd8cf5cbe194e29e3642c5b))
* use active-path index for append leaf clear ([#2413](https://github.com/Sinity/polylogue/issues/2413)) ([ad7dd46](https://github.com/Sinity/polylogue/commit/ad7dd469d0b45f96fe9a77a43267d8b2ff96f993))
* use indexed thread refresh predicate ([#2409](https://github.com/Sinity/polylogue/issues/2409)) ([583c107](https://github.com/Sinity/polylogue/commit/583c107dad8048a6c497fa5aa3e97e838dfeade5))
* **watcher:** skip tail reads for stat-stable cursors ([#2364](https://github.com/Sinity/polylogue/issues/2364)) ([a1fb28b](https://github.com/Sinity/polylogue/commit/a1fb28bf7a5980f9acf10faa62599aa903a94d20))

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
