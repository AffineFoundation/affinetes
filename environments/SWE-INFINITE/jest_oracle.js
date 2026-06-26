// NODE_OPTIONS=--require preload. In the jest CLI process, inject our reporter
// (outcomes, main process) + setupFilesAfterEnv (integrity, test vm) via argv.
try {
  const fs = require('fs');
  const a1 = process.argv[1] || '';
  const isJestCli = /(\/|^)jest(\.js)?$/.test(a1) || a1.endsWith('jest');
  if (isJestCli && !process.env._SWE_ORACLE_ARGV_DONE) {
    process.env._SWE_ORACLE_ARGV_DONE = '1';
    const out = process.env._SWE_ORACLE_OUT || '/workspace/oracle.jsonl';
    try { fs.writeFileSync(out, ''); } catch (e) {}
    try { fs.writeFileSync(out + '.integrity', ''); } catch (e) {}
    process.argv.push(
      '--reporters', 'default',
      '--reporters', '/opt/oracle/jest_reporter.js',
      '--setupFilesAfterEnv', '/opt/oracle/jest_setup.js'
    );
  }
} catch (e) {}
