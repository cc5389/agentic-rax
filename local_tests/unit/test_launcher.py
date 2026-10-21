"""Host-run Claude Code launcher: workspace confinement, sandbox, credentials and resume."""
import json
from pathlib import Path
import time

import pytest

from conftest import configuration
from harness.claude_cli import launcher


SITE = ('/opt/robot-controller', '/srv/robot-sdk-env')  # an operator's robot stack


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    monkeypatch.setenv('RAX_PROTECTED_PATHS', ':'.join(SITE))
    monkeypatch.setenv('FRANKA_CALIBRATION_DIR', str(tmp_path/'calibration'))
    key = tmp_path/'anthropic.key'
    key.write_text('sk-ant-api-secret')
    key.chmod(0o600)
    binary = tmp_path/'bin/claude'
    binary.parent.mkdir()
    binary.write_text('')
    config = configuration(tmp_path, provider='anthropic', subagents=0,
                           backend_options={'protected_paths': ['/opt/second-robot-config.yaml']})
    return config, launcher.deploy(config, key_file=key, claude=binary)


def test_deploy_builds_the_workspace_and_private_session_files(deployed):
    config, d = deployed
    workspace = d['workspace']
    assert (workspace/'AGENTS.md').is_file() and (workspace/'CLAUDE.md').read_text() == '@AGENTS.md\n'
    assert (workspace/'.claude/skills').resolve() == (workspace/'.agents/skills').resolve()
    assert 'Stack the red cup' in (workspace/'TASK.md').read_text()
    assert (workspace/'api/runtime.py').is_file()
    for name in ('configuration.json', 'model-relay.json'):
        assert (config.root/name).stat().st_mode & 0o077 == 0
    assert 'sk-ant' not in (config.root/'model-relay.json').read_text()
    assert not list(workspace.rglob('*.key'))


def test_settings_confine_tools_to_the_workspace_and_sandbox_bash(deployed):
    config, d = deployed
    settings = json.loads((d['claude_home']/'settings.json').read_text())
    workspace = str(d['workspace'])
    permissions, sandbox = settings['permissions'], settings['sandbox']
    assert permissions['defaultMode'] == 'dontAsk'
    assert f'Edit(/{workspace}/**)' in permissions['allow'] and 'mcp__robot' in permissions['allow']
    assert 'Agent' not in permissions['allow'] and 'WebFetch' in permissions['allow'] and 'WebSearch' in permissions['deny']
    assert sandbox['enabled'] and sandbox['failIfUnavailable'] and not sandbox['allowUnsandboxedCommands']
    assert sandbox['filesystem']['allowWrite'] == [workspace]
    denied = sandbox['filesystem']['denyRead']
    site = [Path(p) for p in (*SITE, '/opt/second-robot-config.yaml', config.root.parent/'calibration')]
    for protected in (launcher.PROJECT, Path.home()/'.claude', Path.home()/'.ssh', *site,
                      config.root/'model-relay.json', config.root/'mcp.jsonl'):
        assert str(protected) in denied
        assert f'Read(/{protected})' in permissions['deny']
    assert 'Read(//opt/robot-controller/**)' in permissions['deny']  # a directory: everything below too
    assert sandbox['network'] == {'allowAllUnixSockets': True, 'allowLocalBinding': False, 'allowedDomains': ['*']}
    for hidden in ('/run/docker.sock', '/var/run/docker.sock', f'/run/user/{launcher.UID}'):
        assert hidden in denied  # Unix sockets that would bypass the robot service.
    assert settings['env']['RAX_MCP_SOCKET'] == str(d['socket'])
    assert settings['env']['PYTHONPATH'] == workspace
    assert len(str(d['socket'])) < 100


def test_relative_protected_paths_are_refused(tmp_path, monkeypatch):
    monkeypatch.setenv('RAX_PROTECTED_PATHS', '/opt/robot-controller:robot-sdk')
    with pytest.raises(ValueError, match="absolute: 'robot-sdk'"):
        launcher.protected_paths(configuration(tmp_path))
    monkeypatch.delenv('RAX_PROTECTED_PATHS')
    monkeypatch.delenv('FRANKA_CALIBRATION_DIR', raising=False)
    assert launcher.site_paths(configuration(tmp_path)) == []


def test_mcp_config_bridges_to_the_robot_service_socket(deployed):
    _, d = deployed
    server = json.loads((d['claude_home']/'mcp.json').read_text())['mcpServers']['robot']
    assert server['type'] == 'stdio' and server['args'] == [
        str(launcher.PROJECT/'harness/claude_cli/mcp_bridge.py'), str(d['socket'])]


def test_cli_arguments_and_environment_hold_no_credential(deployed):
    _, d = deployed
    args = launcher.claude_arguments(d, prompt='go')
    assert args[:2] == ['-p', '--model'] and args[-2:] == ['--', 'go']
    assert args[args.index('--permission-mode')+1] == 'dontAsk' and 'Agent' in args
    assert args[args.index('--effort')+1] == 'xhigh'
    environment = launcher.agent_environment(d, ('127.0.0.1', 4242))
    assert environment['ANTHROPIC_BASE_URL'] == 'http://127.0.0.1:4242'
    assert environment['HOME'] == environment['CLAUDE_CONFIG_DIR'] == str(d['claude_home'])
    assert 'sk-ant' not in json.dumps(environment) + ' '.join(args)


def test_a_used_session_root_is_refused(deployed, tmp_path):
    config, _ = deployed
    with pytest.raises(FileExistsError):
        launcher.deploy(config, key_file=tmp_path/'anthropic.key', claude=tmp_path/'bin/claude')


def test_resume_picks_the_newest_top_level_conversation(tmp_path):
    project = tmp_path/'projects/-ws'
    (project/'old/subagents').mkdir(parents=True)
    (project/'old.jsonl').write_text('{}')
    time.sleep(.01)
    (project/'new.jsonl').write_text('{}')
    (project/'old/subagents/agent-x.jsonl').write_text('{}')
    assert launcher.latest_claude_session(tmp_path) == 'new'


def test_resume_budget_excludes_usage_waits(tmp_path):
    record = tmp_path/'agent-lifecycle.json'
    record.write_text(json.dumps({'started_at': 100.0, 'finished_at': 400.0, 'usage_wait_seconds': 100.0,
                                  'attempts': [{'started_at': 0.0, 'finished_at': 50.0}]}))
    assert launcher.attempt_seconds(launcher.resumable_lifecycle(record)) == 250.0


def test_denied_hosts_reach_the_sandbox(tmp_path, monkeypatch):
    from harness.claude_cli import launcher
    assert launcher.denied_hosts() == {}
    monkeypatch.setenv('RAX_DENIED_HOSTS', '10.0.0.2:10.0.0.3')
    assert launcher.denied_hosts() == {'deniedDomains': ['10.0.0.2', '10.0.0.3']}
