"""Run Claude Code on the host as the robot agent, separated from the robot infrastructure.

The launcher (trusted, operator-run) prepares a session under the artifact root:

    <session>/
      configuration.json     operator configuration (services.controller.config)
      model-relay.json       provider, models and the credential *path* (never the token)
      agent-workspace/       the agent's cwd and its only writable directory
      claude-home/           CLAUDE_CONFIG_DIR and HOME of the agent: settings.json, mcp.json
      state.json, mcp.jsonl  robot-service ledger and audit log (unreadable to the agent)
      agent-lifecycle.json   launches, usage waits and exit status

and per launch starts two host services the agent cannot read or modify:

- the Messages relay on 127.0.0.1:<port>, which holds the model credential;
- the robot service (services.controller.frontend --socket), the only robot owner.

Claude Code runs with ``--permission-mode dontAsk`` and an explicit allow list, so
the Read/Edit/Write tools reach only the workspace, and every Bash command runs in
Claude Code's bubblewrap sandbox: writes only in the workspace, no network, the robot
service socket as the only Unix socket, and the repository, robot-controller checkouts,
credentials and the session's private files unreadable. The robot is reachable only
through the MCP tools.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from harness.claude_cli.messages_relay import Credential, MessagesRelay, PROVIDERS
from services.controller.config import Configuration
from services.json_io import write_private_json

PROJECT = Path(__file__).resolve().parents[2]
HOME_NAME = 'claude-home'
WORKSPACE_NAME = 'agent-workspace'
MCP_SERVER = 'robot'
# Tools the robot agent never needs: scheduling, orchestration and worktrees. WebSearch is a
# server tool the Messages relay refuses; WebFetch runs locally and is allowed.
DENIED_TOOLS = ('WebSearch', 'Workflow', 'CronCreate', 'CronDelete', 'CronList', 'ScheduleWakeup',
                'RemoteTrigger', 'SendMessage', 'ListAgents', 'EnterWorktree', 'ExitWorktree', 'PushNotification')
SUBAGENT_TOOLS = ('Agent', 'Task')  # 'Task' is the tool's former name.
RELAY_TOKEN = 'host-relay'  # Placeholder the relay replaces; not a credential.
# Session files only the launcher and the robot service read.
PRIVATE_FILES = ('configuration.json', 'model-relay.json', 'state.json', 'mcp.jsonl', 'frontend.log',
                 'model-usage.jsonl', 'model-calls.json', 'agent-lifecycle.json', 'workspace-manifest.json')
# This repository and credentials: the agent's tools must never read them.
HOME = Path.home()
PROTECTED_PATHS = (PROJECT, HOME/'.config/rax', HOME/'.claude', HOME/'.claude.json', HOME/'.codex', HOME/'.ssh',
                   HOME/'.aws', HOME/'.config/gcloud', HOME/'.netrc', HOME/'.git-credentials')
# The site's robot stack (controller checkouts, robot SDK environments and configurations, camera
# calibration): every path that grants direct robot access, listed by the operator. Colon-separated,
# like PATH; the backend option `protected_paths` adds more per session.
SITE_PATHS_ENVIRONMENT = 'RAX_PROTECTED_PATHS'
# The site's robot hosts off loopback (controller PC, Desk), colon-separated: the sandbox proxy refuses them
# although every other domain is allowed.
DENIED_HOSTS_ENVIRONMENT = 'RAX_DENIED_HOSTS'
# On Linux the Bash sandbox can only allow all Unix sockets or none (seccomp cannot filter
# by path), and the agent's scripts need the robot service socket. These hosts sockets
# would bypass the robot service (docker controls the robot containers) or leak the
# desktop session, so they are hidden from the sandbox.
UID = os.getuid()
HIDDEN_SOCKETS = (Path('/run/docker.sock'), Path('/var/run/docker.sock'), Path('/run/containerd'),
                  Path('/run/dbus'), Path(f'/run/user/{UID}'), Path('/tmp/.X11-unix'),
                  Path(f'/tmp/rax-cam-{UID}'))  # the camera server's private socket
MAX_USAGE_WAITS = 10
# OpenRouter key for gemini_generate, read only by the host robot service (never the agent).
DEFAULT_GEMINI_KEY = Path.home()/'.config/rax/openrouter.key'
USAGE_STOP_SECONDS = 600  # Let Claude Code end its own turn first; then stop it.
RESUME_PROMPT = ("This session was interrupted and has now been resumed in the same workspace. Reread "
                 "memory/PROGRESS.md, call status, and continue from the live state. Do not repeat finished "
                 "work.")
USAGE_PROMPT = ("The model usage limit paused this session until it reset. Call status, reread "
                "memory/PROGRESS.md, and continue from the live state.")


def claude_binary(path=None):
    binary = Path(path or shutil.which('claude') or '').resolve()
    if not binary.is_file():
        raise RuntimeError('Claude Code binary is required (install claude on the host)')
    return binary


def socket_path(root):
    """Short, per-session Unix socket path (Linux limits socket paths to 107 bytes)."""
    directory = Path(f'/tmp/rax-{os.getuid()}')
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    return directory/(hashlib.sha256(str(root).encode()).hexdigest()[:16] + '.sock')


def agent_python(config):
    """Interpreter directory put first on the agent's PATH (operator choice)."""
    configured = config.backend_options.get('agent_python')
    return Path(configured or sys.executable).parent


def site_paths(config):
    """The operator's robot-stack paths: RAX_PROTECTED_PATHS, the session's `protected_paths` and
    the camera calibration directory (FRANKA_CALIBRATION_DIR)."""
    listed = [*os.environ.get(SITE_PATHS_ENVIRONMENT, '').split(':'),
              *config.backend_options.get('protected_paths', ()), os.environ.get('FRANKA_CALIBRATION_DIR', '')]
    paths = []
    for entry in filter(None, listed):
        path = Path(entry).expanduser()
        if not path.is_absolute():
            raise ValueError(f'Protected paths must be absolute: {entry!r}')
        paths.append(path)
    return paths


def denied_hosts():
    hosts = [host for host in os.environ.get(DENIED_HOSTS_ENVIRONMENT, '').split(':') if host]
    return {'deniedDomains': hosts} if hosts else {}


def protected_paths(config):
    return [*PROTECTED_PATHS, *site_paths(config), *(config.root/name for name in PRIVATE_FILES)]


def settings(config, workspace, socket):
    """Claude Code settings: dontAsk permissions confined to the workspace, sandboxed Bash."""
    ws = f'/{workspace}'  # Permission rules spell absolute paths //like/this.
    allow = [f'Read({ws}/**)', f'Edit({ws}/**)',  # Edit rules cover every file-editing tool.
             f'Read(/{config.root}/stream/**)',  # the read-only live recording (runtime/stream)
             'Bash', 'Glob', 'Grep', 'TodoWrite', 'WebFetch', f'mcp__{MCP_SERVER}']
    if config.subagents:
        allow.append('Agent')
    protected = protected_paths(config)
    deny = [*DENIED_TOOLS]
    for path in protected:
        for tool in ('Read', 'Edit'):
            deny.append(f'{tool}(/{path})')
            if not path.suffix:  # Directories (or not yet created): everything below too.
                deny.append(f'{tool}(/{path}/**)')
    timeout_ms = (config.session_seconds + 300) * 1000
    return {
        'permissions': {'defaultMode': 'dontAsk', 'allow': allow, 'deny': deny},
        'sandbox': {
            'enabled': True, 'failIfUnavailable': True, 'autoAllowBashIfSandboxed': True,
            'allowUnsandboxedCommands': False,
            'filesystem': {'allowWrite': [str(workspace)],
                           'denyRead': [str(p) for p in (*protected, *HIDDEN_SOCKETS)]},
            # Internet over the sandbox's HTTP(S) proxy; loopback (the robot bridges) stays closed.
            'network': {'allowAllUnixSockets': True, 'allowLocalBinding': False, 'allowedDomains': ['*'],
                        **denied_hosts()},
        },
        'autoUpdates': False, 'includeCoAuthoredBy': False, 'cleanupPeriodDays': 3650,
        'env': {
            'RAX_MCP_SOCKET': str(socket),
            'PYTHONPATH': str(workspace), 'PYTHONDONTWRITEBYTECODE': '1',
            'PATH': f'{agent_python(config)}:/usr/local/bin:/usr/bin:/bin',
            'API_TIMEOUT_MS': str(timeout_ms), 'MCP_TOOL_TIMEOUT': str(timeout_ms), 'MCP_TIMEOUT': '60000',
            'BASH_DEFAULT_TIMEOUT_MS': '600000', 'BASH_MAX_TIMEOUT_MS': str(config.session_seconds * 1000),
            'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY': '1',
            'CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS': '0',
            # Every MCP tool stays a local tool definition (the relay refuses server tools).
            'ENABLE_TOOL_SEARCH': 'false'}}


def populate_workspace(config, workspace, *, gemini=True):
    """Template, skills, TASK.md and the contract for a new agent workspace."""
    from services.workspace import Setup, build, recipe_for
    for name in ('code', 'memory', 'output', 'runtime/frames', 'tmp'):
        (workspace/name).mkdir(parents=True, exist_ok=True)
    # The live camera/robot recording lives outside the workspace: readable, never writable.
    (config.root/'stream').mkdir(mode=0o755, exist_ok=True)
    (workspace/'runtime'/'stream').symlink_to(config.root/'stream', target_is_directory=True)
    built = build(recipe_for(), Setup.from_config(config, gemini=gemini), workspace)
    (workspace/'TASK.md').write_text(f'# Task: {config.task}\n\n{config.task_description.strip()}\n')
    (workspace/'CLAUDE.md').write_text('@AGENTS.md\n')
    (workspace/'.claude').mkdir(exist_ok=True)
    (workspace/'.claude/skills').symlink_to('../.agents/skills', target_is_directory=True)
    return built


def deploy(config, *, key_file=None, gemini_key_file=None, claude=None):
    """Prepare a new session without starting the robot service or a model."""
    binary = claude_binary(claude)
    root = config.root
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    workspace, home = root/WORKSPACE_NAME, root/HOME_NAME
    if (root/'state.json').exists() or workspace.exists():
        raise FileExistsError('Use a fresh session root')
    provider = config.agent_provider
    model = config.agent_model or PROVIDERS[provider]['model']
    account = None
    if provider == 'claude-login':
        from harness.claude_cli import experiment_login
        key_file = key_file or experiment_login.token_file()  # Never the personal ~/.claude login.
        account = experiment_login.status()
    Credential(provider, key_file=key_file)  # Validate before provisioning.
    write_private_json(root/'model-relay.json', {
        'provider': provider, 'base_url': PROVIDERS[provider]['base_url'], 'model': model,
        'models': sorted({model, PROVIDERS[provider]['auditor'], PROVIDERS[provider]['fast']}),
        'key_file': str(key_file) if key_file else None,
        'gemini_key_file': str(gemini_key_file) if gemini_key_file else None,
        # Identity only (no token): which dedicated account this session bills.
        'account': {k: account.get(k) for k in ('email', 'organization')} if account else None})
    write_private_json(root/'configuration.json', config.to_dict())
    workspace.mkdir(mode=0o700)
    home.mkdir(mode=0o700)
    write_private_json(root/'workspace-manifest.json', populate_workspace(config, workspace,
                                                                          gemini=bool(gemini_key_file)))
    socket = socket_path(root)
    (home/'settings.json').write_text(json.dumps(settings(config, workspace, socket), indent=2))
    (home/'.claude.json').write_text(json.dumps({
        'hasCompletedOnboarding': True,
        'projects': {str(workspace): {'hasTrustDialogAccepted': True, 'hasCompletedProjectOnboarding': True}}}))
    (home/'mcp.json').write_text(json.dumps({'mcpServers': {MCP_SERVER: {
        'type': 'stdio', 'command': sys.executable,
        'args': [str(PROJECT/'harness/claude_cli/mcp_bridge.py'), str(socket)]}}}))
    return deployed_paths(config, binary, model)


def deployed_paths(config, binary, model):
    root = config.root
    return {'root': root, 'workspace': root/WORKSPACE_NAME, 'claude_home': root/HOME_NAME, 'claude': binary,
            'model': model, 'configuration': root/'configuration.json',
            'relay_configuration': root/'model-relay.json', 'socket': socket_path(root),
            'subagents': config.subagents, 'effort': config.agent_effort, 'resume_session': None}


def redeploy(config, claude=None):
    """Reattach an existing session for a resumed launch."""
    relay_file = config.root/'model-relay.json'
    deployed = deployed_paths(config, claude_binary(claude), json.loads(relay_file.read_text())['model'])
    for key in ('workspace', 'claude_home', 'configuration', 'relay_configuration'):
        if not deployed[key].exists():
            raise FileNotFoundError(f'Cannot resume: {deployed[key]} is missing')
    deployed['resume_session'] = latest_claude_session(deployed['claude_home'])
    return deployed


def latest_claude_session(home):
    """Id of the newest top-level Claude Code conversation (subagent transcripts live below it)."""
    transcripts = sorted(Path(home, 'projects').glob('*/*.jsonl'), key=lambda p: p.stat().st_mtime)
    if not transcripts:
        raise RuntimeError('No Claude Code conversation to resume in '+str(home))
    return transcripts[-1].stem


def claude_arguments(deployed, *, prompt, interactive=False, json_output=False):
    denied = DENIED_TOOLS + (() if deployed.get('subagents') else SUBAGENT_TOOLS)
    home = deployed['claude_home']
    args = ['--model', deployed['model'], '--effort', deployed.get('effort', 'xhigh'),
            '--settings', str(home/'settings.json'),
            '--mcp-config', str(home/'mcp.json'), '--strict-mcp-config',
            '--permission-mode', 'dontAsk', '--disallowedTools', *denied]
    if deployed.get('resume_session'):
        args += ['--resume', deployed['resume_session']]  # Continue the session's own conversation.
    if not interactive:
        args = ['-p', *args, '--output-format', 'stream-json' if json_output else 'text', '--verbose']
    return [*args, '--', prompt]


def agent_environment(deployed, relay_address):
    """The agent CLI's process environment: nothing inherited but the terminal basics."""
    home = deployed['claude_home']
    environment = {k: os.environ[k] for k in ('TERM', 'LANG', 'LC_ALL', 'COLORTERM') if k in os.environ}
    environment.update({
        'HOME': str(home), 'CLAUDE_CONFIG_DIR': str(home),
        'PATH': f'{deployed["claude"].parent}:/usr/local/bin:/usr/bin:/bin',
        'ANTHROPIC_BASE_URL': f'http://{relay_address[0]}:{relay_address[1]}',
        'ANTHROPIC_AUTH_TOKEN': RELAY_TOKEN, 'RAX_MCP_SOCKET': str(deployed['socket'])})
    return environment


def relay_credential(deployed):
    relay = json.loads(deployed['relay_configuration'].read_text())
    return relay, Credential(relay['provider'], key_file=relay['key_file'])


def start_relay(config, deployed, *, max_model_calls):
    relay, credential = relay_credential(deployed)
    model = MessagesRelay(('127.0.0.1', 0), credential, base_url=relay['base_url'],
                          models=relay.get('models') or [relay['model']], max_calls=max_model_calls,
                          calls_file=config.root/'model-calls.json', client_write_timeout=config.session_seconds,
                          usage_log=config.root/'model-usage.jsonl')
    model.start()
    return model


def start_robot_service(config, deployed, log, *, started_at, timeout=120):
    """The robot service: sole robot owner, serving the agent CLI and the agent's scripts."""
    relay = json.loads(deployed['relay_configuration'].read_text())
    socket = deployed['socket']
    socket.unlink(missing_ok=True)
    args = [sys.executable, '-m', 'services.controller.frontend', '--config', str(deployed['configuration']),
            '--socket', str(socket), '--started-at', str(started_at)]
    if relay.get('gemini_key_file'):
        args += ['--key-file', relay['gemini_key_file']]
    environment = {**os.environ, 'PYTHONPATH': str(PROJECT)}
    server = subprocess.Popen(args, cwd=PROJECT, env=environment, stdout=log, stderr=log)
    deadline = time.monotonic() + timeout
    while not socket.exists():
        if server.poll() is not None or time.monotonic() >= deadline:
            stop(server)
            raise RuntimeError(f'Robot service startup failed; inspect {config.root}/frontend.log')
        time.sleep(.1)
    return server


def stop(process, grace=10):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def watch(agent, model, timeout, *, stop_on_limit=True):
    """Wait for the agent; stop it once the usage window has been exhausted for a while."""
    deadline, limited = time.monotonic() + timeout, None
    while agent.poll() is None:
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(agent.args, timeout)
        if stop_on_limit and model.limited_until > time.time():
            limited = limited or time.monotonic()
            if time.monotonic() - limited > USAGE_STOP_SECONDS:
                stop(agent, grace=30)
        time.sleep(1)


def until_usage_resets(run, model, *, deployed, prompt, budget, interactive, on_wait=None, sleep=time.sleep):
    """Run the agent. If the subscription usage window ran out during a run, wait for the reset
    (off the session clock; the robot service and relay stay up), then continue the agent's own
    conversation."""
    began, waited, waits = time.monotonic(), 0.0, 0
    while True:
        run_started = time.time()
        returncode = run(prompt, budget - (time.monotonic() - began - waited))
        if interactive or model.limited_until <= run_started or waits >= MAX_USAGE_WAITS:
            return returncode
        pause = max(0.0, model.limited_until + 60 - time.time())
        if on_wait:
            on_wait(model.limited_until, waited)
        sleep(pause)
        waited, waits = waited + pause, waits + 1
        if on_wait:
            on_wait(None, waited)
        deployed['resume_session'] = latest_claude_session(deployed['claude_home'])
        prompt = USAGE_PROMPT


def launch(config, deployed, *, prompt, interactive=True, seconds=None, max_model_calls=None, on_wait=None):
    """Start the relay and robot service, run Claude Code in the workspace, clean up."""
    relay_credential(deployed)  # Fail before starting anything.
    started_at = time.time()
    server = model = agent = None
    with (config.root/'frontend.log').open('ab') as log:
        try:
            model = start_relay(config, deployed, max_model_calls=max_model_calls or config.max_model_calls)
            server = start_robot_service(config, deployed, log, started_at=started_at)
            environment = agent_environment(deployed, model.address)
            budget = seconds or config.session_seconds

            def run(prompt, remaining):
                nonlocal agent
                command = [str(deployed['claude']), *claude_arguments(deployed, prompt=prompt, interactive=interactive)]
                agent = subprocess.Popen(command, cwd=deployed['workspace'], env=environment)
                watch(agent, model, remaining, stop_on_limit=not interactive)
                return agent.returncode
            return until_usage_resets(run, model, deployed=deployed, prompt=prompt, budget=budget,
                                      interactive=interactive, on_wait=on_wait)
        finally:
            stop(agent)
            stop(server, grace=30)
            if model:
                model.close()


def attempt_seconds(attempts):
    return sum(max(0.0, a['finished_at'] - a['started_at'] - a.get('usage_wait_seconds', 0.0)) for a in attempts
               if isinstance(a.get('started_at'), (int, float)) and isinstance(a.get('finished_at'), (int, float)))


def resumable_lifecycle(record):
    """Earlier attempts of a session, or a reason it cannot be resumed."""
    if not record.is_file():
        raise ValueError('Nothing to resume: this session was never launched')
    previous = json.loads(record.read_text())
    if previous.get('status') == 'prepared-only':
        raise ValueError('The session was only prepared; launch it instead')
    if not previous.get('finished_at'):
        # The caller holds launch.lock, so the previous launcher died without cleanup.
        logs = [p.stat().st_mtime for p in (record.parent/'frontend.log', record.parent/'state.json') if p.exists()]
        previous.update(status='crashed', finished_at=max(logs, default=previous.get('started_at', time.time())))
    return [*previous.pop('attempts', []), previous]


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True, help='Operator configuration JSON')
    parser.add_argument('--key-file', type=Path, help='Model credential (openrouter/anthropic key, or the '
                        'claude-login experiment token; default for claude-login: the experiment token)')
    parser.add_argument('--gemini-key-file', type=Path,
                        help=f'OpenRouter key enabling gemini_generate (default: {DEFAULT_GEMINI_KEY} if present)')
    parser.add_argument('--no-gemini', action='store_true', help='Run without gemini_generate')
    parser.add_argument('--prepare-only', action='store_true', help='Build the session; start no service or model')
    parser.add_argument('--exec-prompt', help='Run non-interactively with this prompt')
    parser.add_argument('--non-interactive', action='store_true', help='Run START_PROMPT.md without a terminal UI')
    parser.add_argument('--resume', action='store_true', help="Continue a stopped session's own conversation")
    args = parser.parse_args()
    config = Configuration.load(args.config)
    signal.signal(signal.SIGTERM, _interrupt)
    config.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (config.root/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = config.root/'agent-lifecycle.json'
        lifecycle = {'pid': os.getpid(), 'started_at': time.time(), 'status': 'preparing'}
        seconds = config.session_seconds
        if args.resume:
            try:
                attempts = resumable_lifecycle(record)
            except ValueError as exc:
                raise SystemExit(str(exc))
            seconds -= int(attempt_seconds(attempts))
            if seconds < 60:
                raise SystemExit('The session has no wall-clock budget left to resume')
            lifecycle.update(resumed=True, attempts=attempts)
        elif record.exists():
            raise FileExistsError('This session was already launched; use --resume or a fresh configuration')
        write_private_json(record, lifecycle)
        try:
            if config.agent_provider == 'claude-login' and not args.key_file:
                from harness.claude_cli import experiment_login
                experiment_login.check()  # Fail before provisioning if the token is missing or revoked.
            gemini_key = None if args.no_gemini else args.gemini_key_file or (
                DEFAULT_GEMINI_KEY if DEFAULT_GEMINI_KEY.is_file() else None)
            if args.resume:
                deployed = redeploy(config)
            else:
                deployed = deploy(config, key_file=args.key_file, gemini_key_file=gemini_key)
            if args.prepare_only:
                print(json.dumps({k: str(v) for k, v in deployed.items()}, indent=2))
                lifecycle['status'] = 'prepared-only'
                return
            interactive = not (args.non_interactive or args.exec_prompt)
            prompt = args.exec_prompt or (RESUME_PROMPT if args.resume else
                                          (deployed['workspace']/'START_PROMPT.md').read_text().strip())
            base_wait = lifecycle.get('usage_wait_seconds', 0.0)

            def usage_wait(until, waited):
                if until is None:
                    lifecycle.update(status='running', usage_wait_seconds=base_wait + waited)
                    lifecycle.pop('usage_limited_until', None)
                else:
                    lifecycle.update(status='usage_wait', usage_limited_until=until)
                write_private_json(record, lifecycle)
            lifecycle['status'] = 'running'
            write_private_json(record, lifecycle)
            returncode = launch(config, deployed, prompt=prompt, interactive=interactive, seconds=seconds,
                                on_wait=usage_wait)
            lifecycle.update(status='exited', returncode=returncode)
            raise SystemExit(returncode)
        except KeyboardInterrupt:
            lifecycle['status'] = 'stopped'
            raise SystemExit(130)
        except Exception as exc:
            lifecycle.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            lifecycle['finished_at'] = time.time()
            write_private_json(record, lifecycle)


if __name__ == '__main__':
    main()
