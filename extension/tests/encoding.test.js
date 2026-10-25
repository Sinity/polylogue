import { describe, expect, it } from "vitest";
import { sortedJson, deriveAccountScope } from "../src/backfill/encoding.js";

describe("local checkpoint encoding", () => {
  it("orders Unicode scalar keys and rejects ambiguous normalized keys and numbers", () => {
    expect(sortedJson({ "\u{10000}": 2, "\uE000": 1 })).toBe('{"\uE000":1,"\u{10000}":2}');
    expect(() => sortedJson({ "é": 1, "e\u0301": 2 })).toThrow("key_collision");
    expect(() => sortedJson({ value: 1.5 })).toThrow("number");
  });
  it("isolates local provider accounts and profiles without exposing account handles", async () => {
    const scope = await deriveAccountScope("profile", "chatgpt", "neutral@example.test");
    expect(scope).toMatch(/^h1:[A-Za-z0-9_-]{43}$/);
    expect(scope).not.toContain("neutral");
    expect(await deriveAccountScope("other-profile", "chatgpt", "neutral@example.test")).not.toBe(scope);
    expect(await deriveAccountScope("profile", "chatgpt", "other@example.test")).not.toBe(scope);
  });
});
