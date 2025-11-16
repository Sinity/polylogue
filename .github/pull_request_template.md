## Summary

<!-- Brief description of what this PR does -->

## Changes

<!-- List the main changes in this PR -->

-
-
-

## Checklist

### Code Quality

- [ ] All commits are clean and meaningful (no "WIP", "fix typo", "oops" commits)
- [ ] Commit messages follow [conventional commits](https://www.conventionalcommits.org/) format
- [ ] Each commit is atomic and could be cherry-picked independently
- [ ] Tests pass (`pytest`)
- [ ] No unused imports or debug code

### Git Hygiene

**Before requesting review, clean your branch:**

```bash
# Interactive rebase to squash/fixup messy commits
git rebase -i main

# Example - turn this:
#   - fix: implement feature X
#   - WIP trying something
#   - oops typo
#   - actually fix it
# Into this:
#   - fix: implement feature X
#   - feat: add validation for edge cases
```

- [ ] Feature branch has been rebased/cleaned (see [CONTRIBUTING.md](CONTRIBUTING.md#cleaning-your-branch))
- [ ] No merge commits FROM main in this branch (use `git rebase main`, not `git merge main`)
- [ ] Branch is up to date with main

### Documentation

- [ ] README updated if user-facing changes
- [ ] AGENTS.md updated if development workflow changes
- [ ] Added tests for new functionality

## Testing

<!-- How did you test this? -->

```bash
# Example:
pytest tests/test_new_feature.py -v
python3 polylogue.py <new-command> --dry-run
```

## Additional Notes

<!-- Any context reviewers should know -->

---

**Note:** This repository uses **merge commits with discipline**. Your PR will be merged with `--no-ff` to preserve branch structure. Please ensure your commits are clean before requesting review.
