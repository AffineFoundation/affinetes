# Trusted out-of-repo RSpec oracle (verifier Layer 2, Ruby).
#
# Loaded via SPEC_OPTS="-r /opt/oracle/rspec_oracle.rb -f RSpecOracleFormatter"
# AFTER rspec-core is loaded but BEFORE the spec files (and the miner's
# implementation .rb they require) are loaded. It:
#   1. captures the real File.open + snapshots RSpec::Core::Example#run at load
#      (before any model code runs);
#   2. records the authoritative per-example outcome out-of-band to a private
#      file the miner cannot reach (defeats fabricated stdout);
#   3. at suite close, verifies Example#run was not redefined (Ruby in-process
#      monkeypatch of the runner) and writes the integrity verdict.
#
# Same JSONL contract as the other languages' oracles:
#   {"event":"test","nodeid":"<file>::<full description>","outcome":"passed|failed"}
#   {"event":"finish","integrity_ok":true|false,"reasons":[...]}
require 'json'

$SWE_REAL_FILE_OPEN = File.method(:open)
begin
  $SWE_ORIG_EXAMPLE_RUN = RSpec::Core::Example.instance_method(:run)
rescue StandardError
  $SWE_ORIG_EXAMPLE_RUN = nil
end

class RSpecOracleFormatter
  if defined?(RSpec::Core::Formatters)
    RSpec::Core::Formatters.register self, :example_passed, :example_failed, :example_pending, :close
  end

  def initialize(_output)
    path = ENV['_SWE_ORACLE_OUT'] || '/workspace/oracle.jsonl'
    @io = $SWE_REAL_FILE_OPEN.call(path, 'w')
  end

  def _emit(hash)
    @io.write(JSON.generate(hash) + "\n")
    @io.flush
  rescue StandardError
    nil
  end

  def _record(notification, outcome)
    ex = notification.example
    nodeid = "#{ex.metadata[:file_path]}::#{ex.full_description}"
    _emit('event' => 'test', 'nodeid' => nodeid, 'outcome' => outcome)
  end

  def example_passed(notification)
    _record(notification, 'passed')
  end

  def example_failed(notification)
    _record(notification, 'failed')
  end

  def example_pending(notification)
    _record(notification, 'failed')
  end

  def close(_notification)
    ok = true
    reasons = []
    if $SWE_ORIG_EXAMPLE_RUN.nil?
      ok = false
      reasons << 'rspec_internals_unavailable'
    elsif RSpec::Core::Example.instance_method(:run) != $SWE_ORIG_EXAMPLE_RUN
      ok = false
      reasons << 'Example#run_replaced'
    end
    _emit('event' => 'finish', 'integrity_ok' => ok, 'reasons' => reasons)
    @io.close
  rescue StandardError
    nil
  end
end
