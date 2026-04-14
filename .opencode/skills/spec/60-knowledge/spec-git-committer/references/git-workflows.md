# Git Workflow Patterns

## Branching Strategies

### 1. GitHub Flow (Simple)

Best for: Continuous deployment, small teams

```
main ──●────●────●────●────●────●──►
        \      /  \      /
feature  ●────●    ●────●
```

Rules:
- `main` is always deployable
- Create feature branches from `main`
- Open PR, get review, merge
- Deploy immediately after merge

```bash
# Start feature
git checkout -b feature/add-login

# Work, commit
git commit -m "feat(auth): add login form"
git commit -m "feat(auth): add validation"

# Push and create PR
git push -u origin feature/add-login
gh pr create

# After approval, merge
gh pr merge --squash
```

### 2. GitFlow (Complex)

Best for: Scheduled releases, large teams

```
main     ──●─────────────────●─────────●──►
            \               /         /
release      ●─────●──────●         /
              \   /              /
develop  ──●───●─●───●───●───●───●──►
            \     /       \     /
feature      ●───●         ●───●
```

Branches:
- `main`: Production releases only
- `develop`: Integration branch
- `feature/*`: New features
- `release/*`: Release preparation
- `hotfix/*`: Production fixes

```bash
# Start feature
git checkout develop
git checkout -b feature/add-search

# Finish feature
git checkout develop
git merge --no-ff feature/add-search

# Start release
git checkout -b release/1.2.0

# Finish release
git checkout main
git merge --no-ff release/1.2.0
git tag -a v1.2.0 -m "Release 1.2.0"
git checkout develop
git merge --no-ff release/1.2.0
```

### 3. Trunk-Based Development

Best for: Experienced teams, high automation

```
main  ──●──●──●──●──●──●──●──●──►
         |     |     |
       short-lived branches (< 1 day)
```

Rules:
- Branch for < 1 day
- Feature flags for incomplete work
- Merge to main multiple times/day

```bash
# Short-lived branch
git checkout -b add-button
# ... few hours of work ...
git commit -m "feat: add submit button (behind flag)"
git push origin add-button
# Create PR, merge same day
```

## Merge Strategies

### Merge Commit (--no-ff)
```bash
git merge --no-ff feature/add-login
```
```
     A---B---C feature
    /         \
---D-----------E main (merge commit)
```
Pros: Preserves history, easy to revert
Cons: Cluttered history

### Squash Merge
```bash
git merge --squash feature/add-login
git commit -m "feat(auth): add login functionality"
```
```
     A---B---C feature
    /
---D-----------E main (single commit with all changes)
```
Pros: Clean history, atomic features
Cons: Loses individual commits

### Rebase
```bash
git checkout feature/add-login
git rebase main
git checkout main
git merge feature/add-login
```
```
Before:      A---B feature
            /
       D---E main

After:       D---E---A'---B' main
```
Pros: Linear history
Cons: Rewrites history, can be dangerous

## Commit Hygiene

### Before PR: Interactive Rebase
```bash
# Clean up last 5 commits
git rebase -i HEAD~5
```

Options:
- `pick`: Keep commit as-is
- `reword`: Change commit message
- `edit`: Amend commit
- `squash`: Combine with previous
- `fixup`: Combine, discard message
- `drop`: Delete commit

### Fixup Workflow
```bash
# Commit that fixes previous commit
git commit --fixup=abc123

# Later, auto-squash
git rebase -i --autosquash main
```

### Amend Last Commit
```bash
# Add forgotten file
git add forgotten-file.py
git commit --amend --no-edit

# Change message
git commit --amend -m "feat(auth): add login (fixed)"
```

## Conflict Resolution

### Standard Resolution
```bash
git merge feature-branch
# CONFLICT in file.py

# 1. Open file, find markers
<<<<<<< HEAD
current code
=======
incoming code
>>>>>>> feature-branch

# 2. Edit to resolve
# 3. Stage and commit
git add file.py
git commit -m "merge: resolve conflict in file.py"
```

### Using Theirs/Ours
```bash
# Accept all changes from branch being merged
git checkout --theirs file.py

# Keep all current changes
git checkout --ours file.py
```

### Abort If Needed
```bash
git merge --abort
git rebase --abort
```

## Tags and Releases

### Semantic Versioning
```
v1.2.3
│ │ │
│ │ └── Patch: Bug fixes
│ └──── Minor: New features (backward compatible)
└────── Major: Breaking changes
```

### Creating Tags
```bash
# Lightweight tag
git tag v1.2.3

# Annotated tag (recommended)
git tag -a v1.2.3 -m "Release 1.2.3: Add login feature"

# Push tags
git push origin v1.2.3
git push --tags
```

### Release Workflow
```bash
# 1. Create release branch
git checkout -b release/1.2.3

# 2. Bump version, update changelog
vim VERSION
vim CHANGELOG.md
git commit -m "chore: prepare release 1.2.3"

# 3. Merge to main
git checkout main
git merge --no-ff release/1.2.3

# 4. Tag
git tag -a v1.2.3 -m "Release 1.2.3"

# 5. Push
git push origin main --tags

# 6. Merge back to develop
git checkout develop
git merge --no-ff release/1.2.3
```

## Emergency Procedures

### Revert a Merge
```bash
# Find merge commit
git log --oneline

# Revert merge (keep first parent)
git revert -m 1 abc123
```

### Reset to Previous State
```bash
# Soft: Keep changes staged
git reset --soft HEAD~1

# Mixed: Keep changes unstaged
git reset HEAD~1

# Hard: Discard all changes (DANGEROUS)
git reset --hard HEAD~1
```

### Recover Lost Commits
```bash
# Find lost commits
git reflog

# Restore
git checkout abc123
# or
git cherry-pick abc123
```
