# Validate an Affinetes Environment

Run the fail-closed lifecycle and task-identity gate against a local environment
directory or an already-built image.

## Usage

```bash
afs validate [ENV_DIR | --image IMAGE] \
  --task-id ID --task-id ID \
  [--expect-failure-task-id ID \
   --expected-failure-error-code CODE] \
  [--model MODEL --base-url LOOPBACK_URL] \
  [--allow-remote-model-endpoint] \
  [--api-key-env NAME] [--output DIR]
```

The command also supports `--task-id-start` with `--num-tests` or
`--task-id-end`. Explicit `--task-id` values cannot be combined with a range.

## Safety boundary

- The default release gate must use a loopback model stub and an already-built
  local image. Do not use production credentials, production model endpoints or
  an online configuration store.
- A non-loopback model URL is rejected before Docker access unless the owner
  explicitly supplies `--allow-remote-model-endpoint`. Never use that opt-in in
  the local release gate.
- Validation refuses non-default Docker contexts, SSH Docker endpoints and
  non-loopback Docker daemons before build, pull or container mutation.
- Registry-digest validation is an owner-authorized clean-host action. The host's
  Docker daemon must still be local.
- Read credentials through `--api-key-env`; never place them in the command line.
- A per-run owner label, exact container ID and local image ID bind cleanup. A
  pre-existing or identity-mismatched container is never replaced or deleted.

## What it validates

1. Every valid task runs twice with different model seeds and returns
   `success=true` plus a finite numeric score.
2. A task's prompt hash is invariant across model seeds; different task IDs have
   different prompt hashes.
3. Expected-failure IDs are rejected twice with the exact configured structured
   error code.
4. The owned container is cleaned up and its absence is verified.
5. The atomic `summary.json` is secret-safe and contains no full prompts or model
   responses.

## Local example

```bash
export MODEL_API_KEY=local-test-secret
afs validate --image instruction-gym:local \
  --task-id 0 \
  --task-id 1 \
  --expect-failure-task-id 102636151 \
  --expected-failure-error-code invalid_task_id \
  --model local-stub \
  --base-url http://127.0.0.1:18080/v1 \
  --api-key-env MODEL_API_KEY \
  --output /tmp/affinetes-validation
```

The only generated result file is `summary.json`. The command exits non-zero if
any evaluation, identity, report-safety or cleanup check fails.
