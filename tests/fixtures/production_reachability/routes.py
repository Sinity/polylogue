"""Small import/call graph fixture with one live and one unreachable helper."""

from __future__ import annotations


def production_entrypoint() -> str:
    return live_helper()


def live_helper() -> str:
    return "live"


def dead_helper() -> str:
    return "dead"


def route_with_nested() -> str:
    def nested() -> str:
        return dead_helper()

    return live_helper()


def route_accepts_helper(helper: object) -> str:
    return live_helper()


def shadowed_route() -> str:
    def live_helper() -> str:
        return "shadowed"

    return live_helper()


class Runner:
    def run(self) -> str:
        return live_helper()


def class_route() -> str:
    return Runner().run()
