# Contributing to Affinetes

Thank you for your interest in contributing to Affinetes! This document provides guidelines for contributing to the project, with special attention to security practices.

## Table of Contents

- [Security Guidelines](#security-guidelines)
- [Code Standards](#code-standards)
- [Development Setup](#development-setup)
- [Submitting Changes](#submitting-changes)

## Security Guidelines

### Cryptographically Secure Random Number Generation

**CRITICAL:** This project has been hardened to use cryptographically secure random number generation throughout. When contributing code, you **MUST** follow these practices:

#### ✅ DO: Use `secrets` module for random generation

```python
import secrets

# Generate random integers
random_id = secrets.randbelow(100_000_000)  # Range: 0 to 99,999,999
task_seed = secrets.randbelow(2**32)        # Range: 0 to 4,294,967,295

# Use SystemRandom for random operations
rng = secrets.SystemRandom()
value = rng.randint(1, 100)
choice = rng.choice([1, 2, 3])
shuffled = list(range(10))
rng.shuffle(shuffled)
```

#### ❌ DON'T: Use predictable random generation

```python
import random

# NEVER use these - they are predictable and insecure:
random.randint(0, 100)           # ❌ Predictable
random.Random(seed)              # ❌ Deterministic with seed
rng = random.Random(task_id)     # ❌ Can be predicted
```

### Why This Matters

1. **Security**: `random.randint()` and `random.Random(seed)` use predictable pseudo-random algorithms that can be exploited
2. **Task Integrity**: Even when seeds are provided for task generation, the actual random operations must remain unpredictable
3. **Cryptographic Strength**: `secrets.SystemRandom()` uses the operating system's cryptographically secure random source

### When to Use Each Approach

| Use Case | Method | Example |
|----------|--------|---------|
| Generate task ID | `secrets.randbelow(N)` | `task_id = secrets.randbelow(100_000_000)` |
| Generate seed | `secrets.randbelow(2**32)` | `seed = secrets.randbelow(2**32)` |
| Random operations | `secrets.SystemRandom()` | `rng = secrets.SystemRandom()` |
| Select from list | `rng.choice()` | `rng.choice([1, 2, 3])` |
| Generate range | `rng.randint()` | `rng.randint(1, 100)` |

## Code Standards

### Environment Structure

Each environment follows this structure:

```
environments/<domain>/<env_name>/
├── env.py              # Main environment actor
├── <task_name>.py      # Task generator/verifier
└── ...
```

### Task Generation Pattern

All task generators should follow this secure pattern:

```python
import secrets
from base.data import Data

class TaskGenerator:
    def generate(self, seed: int = None, **kwargs) -> Data:
        """
        Generate a task with cryptographically secure randomness
        
        Args:
            seed: Optional seed (accepted for API compatibility, 
                  but does NOT make generation deterministic)
        """
        # Generate secure seed if not provided
        if seed is None:
            seed = secrets.randbelow(100_000_000)
        
        # Use cryptographically secure RNG
        rng = secrets.SystemRandom()
        
        # Generate task using secure random operations
        difficulty = rng.choice(['easy', 'medium', 'hard'])
        value = rng.randint(1, 100)
        
        return Data(
            question=f"Task with difficulty {difficulty}",
            answer=str(value),
            metadata={"seed": seed, "difficulty": difficulty}
        )
```

### Migration Checklist

When updating or creating new task generators:

- [ ] Import `secrets` module
- [ ] Replace `random.randint()` with `secrets.randbelow()`
- [ ] Replace `random.Random(seed)` with `secrets.SystemRandom()`
- [ ] Remove any `rng.seed()` calls
- [ ] Ensure no predictable patterns based on seeds
- [ ] Test that repeated calls with same seed produce different results

## Development Setup

### Prerequisites

- Python 3.12+
- Virtual environment recommended

### Installation

```bash
cd affinetes
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific environment tests
pytest environments/primeintellect/lgc-v2/

# Verify security (no predictable random usage)
grep -r "random\.randint" environments/ --include="*.py" | grep -v ".venv"
grep -r "random\.Random(" environments/ --include="*.py" | grep -v ".venv"
```

## Submitting Changes

### Pull Request Process

1. **Fork & Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow security guidelines above
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Thoroughly**
   ```bash
   # Security check
   python -c "import secrets; rng = secrets.SystemRandom(); print(rng.randint(1,100))"
   
   # Run tests
   pytest
   ```

4. **Commit with Clear Messages**
   ```bash
   git commit -m "feat: add secure task generator for X"
   git commit -m "security: replace random.randint with secrets.randbelow"
   ```

5. **Submit PR**
   - Describe what changed and why
   - Reference any related issues
   - Confirm security guidelines followed

### Code Review Criteria

Your PR will be reviewed for:

✅ **Security**: No use of predictable random generation  
✅ **Functionality**: Code works as intended  
✅ **Testing**: Adequate test coverage  
✅ **Documentation**: Clear comments and docstrings  
✅ **Style**: Follows Python PEP 8 guidelines  

### Security Review

All PRs undergo automatic security scanning for:

- Use of `random.randint()`
- Use of `random.Random(seed)`
- Predictable seed-based generation
- Missing `secrets` imports

## Recent Security Improvements

### January 2026 Security Hardening

The entire codebase was audited and hardened with the following changes:

1. **Replaced `random.randint()` with `secrets.randbelow()`**
   - All seed generation now cryptographically secure
   - 40+ files updated across all environments

2. **Replaced `random.Random(seed)` with `secrets.SystemRandom()`**
   - Even with seeds, generation is now unpredictable
   - 15+ game generators in lgc-v2 updated
   - Config files updated to use secure RNG

3. **Files Updated:**
   - All environment actors (`env.py`)
   - All task generators (`*_task.py`)
   - All game generators in `lgc-v2/games/`
   - Configuration files using random generation

### Security Benefits

- **Unpredictable**: No one can predict task generation patterns
- **Cryptographic**: Uses OS-level secure random sources
- **Seed-Independent**: Seeds don't create deterministic outputs
- **Attack-Resistant**: Resistant to RNG prediction attacks

## Questions?

If you have questions about:

- Security practices: Review this document's security section
- Code structure: Check existing environment implementations
- Contributing process: Open an issue for discussion

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Remember:** Security is not optional. All contributions must follow the cryptographically secure random generation practices outlined above.
