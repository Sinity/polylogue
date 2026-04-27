"""Anti-omission laws over the verification catalog.

Each test in this package is a single named law expressed as a parametrized
pytest test over the compiled obligation set. The intent is to make the
catalog's structural invariants explicit and enforceable rather than
implicit in catalog construction:

    completeness/         laws asserting required coverage exists
    anti_dead_code/       laws asserting nothing is built but unused

Adding a law here is the mechanism by which the catalog grows beyond
"renderable" into "structurally complete." A law that fires names the
specific subject(s) or claim(s) that violate the rule.
"""
