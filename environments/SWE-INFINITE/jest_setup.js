// setupFilesAfterEnv: runs in the test vm AFTER expect/it are installed but
// BEFORE the test file (and the impl source it requires) loads. Snapshot the
// runner globals; verify at teardown they were not replaced in-process.
const fs = require('fs');
const integ = (process.env._SWE_ORACLE_OUT || '/workspace/oracle.jsonl') + '.integrity';
const _expect = global.expect;
const _it = global.it;
const _test = global.test;
afterAll(() => {
  const reasons = [];
  if (global.expect !== _expect) reasons.push('expect_replaced');
  if (global.it !== _it) reasons.push('it_replaced');
  if (global.test !== _test) reasons.push('test_replaced');
  try { fs.appendFileSync(integ, JSON.stringify({ ok: reasons.length === 0, reasons }) + '\n'); } catch (e) {}
});
