const fs = require('fs');
const out = process.env._SWE_ORACLE_OUT || '/workspace/oracle.jsonl';
const integ = out + '.integrity';
class OracleReporter {
  onTestResult(_test, result) {
    let fp = result.testFilePath || '';
    if (fp.startsWith('/app/')) fp = fp.slice(5);
    for (const r of result.testResults) {
      const outcome = r.status === 'passed' ? 'passed' : 'failed';
      try { fs.appendFileSync(out, JSON.stringify({ event: 'test', nodeid: fp + '::' + r.fullName, outcome }) + '\n'); } catch (e) {}
    }
  }
  onRunComplete() {
    let ok = true; const reasons = [];
    try {
      const lines = fs.readFileSync(integ, 'utf8').trim().split('\n').filter(Boolean);
      for (const ln of lines) { const j = JSON.parse(ln); if (!j.ok) { ok = false; reasons.push(...j.reasons); } }
    } catch (e) {}
    try { fs.appendFileSync(out, JSON.stringify({ event: 'finish', integrity_ok: ok, reasons }) + '\n'); } catch (e) {}
  }
}
module.exports = OracleReporter;
