# Commit Message Conventions

## Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

## Types

| Type | When to Use |
|------|-------------|
| `feat` | New feature for the user |
| `fix` | Bug fix for the user |
| `docs` | Documentation only changes |
| `style` | Formatting, missing semi-colons, etc. |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding missing tests or correcting existing tests |
| `build` | Changes to build process or dependencies |
| `ci` | Changes to CI configuration |
| `chore` | Other changes that don't modify src or test |
| `revert` | Reverts a previous commit |

## Scope

Scope is the module/component affected:
- `auth`, `api`, `db`, `ui`, `config`, `deps`
- Or feature name: `login`, `checkout`, `search`

## Subject Line Rules

1. **Imperative mood**: "add feature" not "added feature"
2. **No period at the end**
3. **Max 50 characters** (72 hard limit)
4. **Capitalize first letter**
5. **What, not how**

### Good Examples
```
feat(auth): add password reset via email
fix(api): handle null user in profile endpoint
docs(readme): update installation instructions
refactor(checkout): extract payment validation logic
perf(search): add index on product_name column
test(auth): add tests for token refresh flow
```

### Bad Examples
```
❌ fixed bug                     # Too vague
❌ feat: Added new login page.   # Past tense, period
❌ updated stuff                  # Meaningless
❌ WIP                            # Never commit WIP
❌ fix(auth): fixing the bug that was causing issues with the login flow when users... # Too long
```

## Body Guidelines

Use body to explain:
- **What** changed and **why**
- **Not how** (that's in the code)
- Context that helps reviewers

```
fix(checkout): prevent double-charge on retry

When payment fails and user retries, we were creating a new
PaymentIntent instead of reusing the existing one. This caused
double charges when the first payment eventually succeeded.

Now we store the PaymentIntent ID in session and reuse it.
```

## Footer Conventions

### Breaking Changes
```
feat(api)!: change user endpoint response format

BREAKING CHANGE: The /api/users endpoint now returns a paginated
response. Clients must handle the new { data: [], meta: {} } format.
```

### Issue References
```
fix(auth): validate token expiry correctly

Fixes #123
Closes #456
Refs #789
```

### Co-authors
```
feat(search): implement fuzzy matching

Co-authored-by: Jane Doe <jane@example.com>
Co-authored-by: Bob Smith <bob@example.com>
```

## Multi-line Commits

```
feat(orders): implement order cancellation

- Add cancel endpoint POST /orders/{id}/cancel
- Send cancellation email to customer
- Refund payment if already charged
- Update inventory levels

This completes the order lifecycle management feature.

Closes #234
```

## Atomic Commits

Each commit should be:
1. **One logical change** - don't mix refactor with feature
2. **Buildable** - code compiles/runs after this commit
3. **Testable** - tests pass after this commit

### Split Large Changes

Instead of:
```
feat: implement user management with auth and profile
```

Do:
```
feat(auth): add user registration endpoint
feat(auth): add user login endpoint
feat(profile): add user profile endpoint
feat(profile): add profile photo upload
```

## Commit Templates

### .gitmessage template
```
# <type>(<scope>): <subject>
# |<----  Using a Maximum Of 50 Characters  ---->|

# Explain why this change is being made
# |<----   Try To Limit Each Line to a Maximum Of 72 Characters   ---->|

# Provide links or keys to any relevant tickets, articles or other resources
# Example: Fixes #23

# --- COMMIT END ---
# Type can be:
#    feat     (new feature)
#    fix      (bug fix)
#    refactor (refactoring production code)
#    style    (formatting, missing semi colons, etc)
#    docs     (changes to documentation)
#    test     (adding or refactoring tests)
#    chore    (updating grunt tasks etc)
# --------------------
# Remember to:
#    Capitalize the subject line
#    Use the imperative mood in the subject line
#    Do not end the subject line with a period
#    Separate subject from body with a blank line
#    Use the body to explain what and why vs. how
#    Can use multiple lines with "-" for bullet points in body
# --------------------
```

Set up:
```bash
git config --global commit.template ~/.gitmessage
```

## Conventional Commits Tooling

### commitlint
```bash
npm install -g @commitlint/cli @commitlint/config-conventional

# commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'scope-enum': [2, 'always', ['auth', 'api', 'ui', 'db']],
    'subject-case': [2, 'always', 'lower-case']
  }
};
```

### commitizen
```bash
npm install -g commitizen
commitizen init cz-conventional-changelog --save-dev

# Use:
git cz
# Instead of git commit
```

### husky + pre-commit
```bash
npm install husky --save-dev
npx husky install
npx husky add .husky/commit-msg 'npx commitlint --edit $1'
```
