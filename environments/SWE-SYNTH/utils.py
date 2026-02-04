"""
Shared utilities for SWE-SYNTH environment.
"""

# Source code file extensions for git diff
# Used to filter out non-code files when extracting patches
DIFF_EXTENSIONS = (
    "'*.js' '*.ts' '*.jsx' '*.tsx' '*.py' '*.java' '*.go' "
    "'*.c' '*.cpp' '*.h' '*.rs' '*.rb' '*.php' '*.cs' "
    "'*.swift' '*.kt' '*.scala' '*.vue' '*.svelte'"
)

# Git history sanitization script
# Used to prevent cheating by removing commit history that could reveal the fix
SANITIZE_GIT_SCRIPT = """
cd /app
git config user.email "agent@swe-synth.local"
git config user.name "SWE-SYNTH Agent"
git checkout --orphan sanitized_branch
git add -A
git commit -m "Initial state"
git branch -D main 2>/dev/null || git branch -D master 2>/dev/null || true
git branch -m main
rm -rf .git/logs
rm -rf .git/refs/original
git reflog expire --expire=now --all 2>/dev/null || true
git gc --prune=now 2>/dev/null || true
echo "Git history sanitized"
"""
