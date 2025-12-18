from __future__ import annotations

import click


class OptionalValueChoiceOption(click.Option):
    """A Click option that supports `--opt` (flag_value) or `--opt <value>`.

    Click does not support optional option values out of the box. We use a
    flag-style option (nargs=0) and, when the option is seen, opportunistically
    consume the following token as the value when it does not look like another
    option.
    """

    def add_to_parser(self, parser, ctx) -> None:  # type: ignore[override]
        super().add_to_parser(parser, ctx)

        # Patch the parser option handlers so `--opt value` works even when the
        # option is registered as a flag (nargs=0).
        for opt in self.opts:
            parser_opt = parser._long_opt.get(opt) or parser._short_opt.get(opt)  # type: ignore[attr-defined]
            if parser_opt is None:
                continue

            def process(value, state, *, self=self):  # noqa: ANN001
                if value is not None:
                    actual = value
                elif getattr(state, "rargs", None) and state.rargs and not state.rargs[0].startswith("-"):
                    actual = state.rargs.pop(0)
                else:
                    actual = self.flag_value
                state.opts[self.name] = actual

            parser_opt.process = process
