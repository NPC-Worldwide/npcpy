
import platform
import shutil
import sys

from npcpy.llm_funcs import get_llm_response

import subprocess
import os
import tempfile

from typing import Any

JOBS_DIR = os.path.expanduser("~/.npcsh/jobs")
LOGS_DIR = os.path.expanduser("~/.npcsh/logs")


def _npc_bin_path():
    """Full path to the ``npc`` binary in the current environment."""
    candidate = os.path.join(os.path.dirname(sys.executable), 'npc')
    if os.path.isfile(candidate):
        return candidate
    return shutil.which('npc') or 'npc'


def _plist_path(job_name):
    return os.path.expanduser(
        '~/Library/LaunchAgents/com.npcsh.job.' + job_name + '.plist'
    )


def _cron_tag(job_name):
    return '# npcsh:' + job_name


# ── Core scheduling primitives ──────────────────────────────────────

def compile_job_script(command, job_name):
    """Turn *command* into a self-contained executable bash script.

    The command is a jinx name + args. The compiled script calls the ``npc``
    binary with its full absolute path so it works in a minimal cron environment.

    Returns the path to the generated script (``~/.npcsh/jobs/<name>.sh``).
    """
    os.makedirs(JOBS_DIR, exist_ok=True)
    script_path = os.path.join(JOBS_DIR, job_name + '.sh')
    npc = _npc_bin_path()
    with open(script_path, 'w') as f:
        f.write('#!/bin/bash\n# npcsh job: ' + job_name
                + '\nset -euo pipefail\n\n'
                + npc + ' ' + command.lstrip('/') + '\n')
    os.chmod(script_path, 0o755)
    return script_path


def schedule_job(schedule, command, job_name):
    """Compile *command* and register it with the OS scheduler.

    Returns ``(success: bool, message: str)``.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    script_path = compile_job_script(command, job_name)
    log_path = os.path.join(LOGS_DIR, job_name + '.log')
    system = platform.system()

    if system == 'Darwin':
        return _schedule_launchd(script_path, schedule, job_name, log_path)
    elif system == 'Windows':
        return _schedule_windows(script_path, schedule, job_name, log_path)
    return _schedule_crontab(script_path, schedule, job_name, log_path)


def unschedule_job(job_name):
    """Remove a scheduled job. Returns ``(success, message)``."""
    system = platform.system()
    if system == 'Darwin':
        return _unschedule_launchd(job_name)
    elif system == 'Windows':
        return _unschedule_windows(job_name)
    return _unschedule_crontab(job_name)


def list_jobs():
    """Return a list of ``{'name': …, 'active': bool}`` for all npcsh jobs."""
    system = platform.system()
    jobs = []
    if system == 'Darwin':
        agents = os.path.expanduser('~/Library/LaunchAgents/')
        if os.path.isdir(agents):
            for f in sorted(os.listdir(agents)):
                if f.startswith('com.npcsh.job.') and f.endswith('.plist'):
                    name = f.replace('com.npcsh.job.', '').replace('.plist', '')
                    r = subprocess.run(
                        ['launchctl', 'list', 'com.npcsh.job.' + name],
                        capture_output=True, text=True,
                    )
                    jobs.append({'name': name, 'active': r.returncode == 0})
    elif system == 'Windows':
        r = subprocess.run(
            ['schtasks', '/query', '/fo', 'CSV', '/nh'],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                if 'NPCSH_' in line:
                    parts = line.split(',')
                    if parts:
                        name = parts[0].strip('"').replace('\\NPCSH_', '').replace('NPCSH_', '')
                        jobs.append({'name': name, 'active': True})
    else:
        r = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                tag_pos = line.find('# npcsh:')
                if tag_pos >= 0:
                    name = line[tag_pos + 8:].strip()
                    jobs.append({'name': name, 'active': True})
    return jobs


def job_is_active(job_name):
    """Quick check whether *job_name* is currently scheduled."""
    system = platform.system()
    if system == 'Darwin':
        return os.path.exists(_plist_path(job_name))
    elif system == 'Windows':
        r = subprocess.run(
            ['schtasks', '/query', '/tn', 'NPCSH_' + job_name],
            capture_output=True, text=True,
        )
        return r.returncode == 0
    r = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if r.returncode == 0:
        return any(_cron_tag(job_name) in l for l in r.stdout.splitlines())
    return False


def job_status(job_name):
    """Detailed status dict for a job."""
    log_path = os.path.join(LOGS_DIR, job_name + '.log')
    info = {
        'name': job_name,
        'active': job_is_active(job_name),
        'log': log_path,
        'recent_log': [],
    }
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                info['recent_log'] = f.readlines()[-10:]
        except OSError:
            pass
    return info


# ── macOS (launchd) ─────────────────────────────────────────────────

def _schedule_launchd(script_path, schedule, job_name, log_path):
    ppath = _plist_path(job_name)
    os.makedirs(os.path.dirname(ppath), exist_ok=True)

    parts = schedule.split()
    plist = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
        ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n<dict>\n'
        '  <key>Label</key>\n'
        '  <string>com.npcsh.job.' + job_name + '</string>\n'
        '  <key>ProgramArguments</key>\n  <array>\n'
        '    <string>' + script_path + '</string>\n'
        '  </array>\n'
    )

    if len(parts) == 5:
        plist += '  <key>StartCalendarInterval</key>\n  <dict>\n'
        for key, idx in [('Minute', 0), ('Hour', 1), ('Day', 2), ('Month', 3), ('Weekday', 4)]:
            if parts[idx] != '*':
                plist += '    <key>' + key + '</key>\n    <integer>' + parts[idx] + '</integer>\n'
        plist += '  </dict>\n'
    else:
        # Treat as interval in seconds
        try:
            plist += '  <key>StartInterval</key>\n  <integer>' + str(int(schedule)) + '</integer>\n'
        except ValueError:
            pass

    plist += (
        '  <key>StandardOutPath</key>\n  <string>' + log_path + '</string>\n'
        '  <key>StandardErrorPath</key>\n  <string>' + log_path + '</string>\n'
        '</dict>\n</plist>\n'
    )

    with open(ppath, 'w') as f:
        f.write(plist)
    subprocess.run(['launchctl', 'load', ppath], capture_output=True)
    return True, 'Scheduled "' + job_name + '": ' + schedule


def _unschedule_launchd(job_name):
    ppath = _plist_path(job_name)
    if os.path.exists(ppath):
        subprocess.run(['launchctl', 'unload', ppath], capture_output=True)
        os.remove(ppath)
        return True, 'Removed "' + job_name + '"'
    return False, 'Job "' + job_name + '" not found.'


# ── Linux (crontab) ─────────────────────────────────────────────────

def _schedule_crontab(script_path, schedule, job_name, log_path):
    r = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    existing = r.stdout if r.returncode == 0 else ''
    entry = schedule + ' ' + script_path + ' >> ' + log_path + ' 2>&1 ' + _cron_tag(job_name)
    new_crontab = existing.rstrip('\n') + '\n' + entry + '\n'
    p = subprocess.run(['crontab', '-'], input=new_crontab, capture_output=True, text=True)
    if p.returncode == 0:
        return True, 'Scheduled "' + job_name + '": ' + schedule
    return False, 'Failed: ' + p.stderr


def _unschedule_crontab(job_name):
    r = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
    if r.returncode == 0:
        tag = _cron_tag(job_name)
        lines = [l for l in r.stdout.splitlines() if tag not in l]
        p = subprocess.run(['crontab', '-'], input='\n'.join(lines) + '\n', capture_output=True, text=True)
        if p.returncode == 0:
            return True, 'Removed "' + job_name + '"'
        return False, 'Failed: ' + p.stderr
    return False, 'No crontab found.'


# ── Windows (Task Scheduler) ────────────────────────────────────────

def _schedule_windows(script_path, schedule, job_name, log_path):
    task_name = 'NPCSH_' + job_name
    schedule_params = schedule.split()
    cmd = (
        ['schtasks', '/create', '/tn', task_name, '/tr',
         'powershell -NoProfile -ExecutionPolicy Bypass -File ' + script_path]
        + schedule_params + ['/f']
    )
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return True, 'Scheduled "' + job_name + '": ' + schedule
    return False, 'Failed: ' + r.stderr


def _unschedule_windows(job_name):
    task_name = 'NPCSH_' + job_name
    r = subprocess.run(
        ['schtasks', '/delete', '/tn', task_name, '/f'],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        return True, 'Removed "' + job_name + '"'
    return False, 'Failed: ' + r.stderr


# ── LLM-driven /plan command ────────────────────────────────────────

def execute_plan_command(
    command,
    **kwargs,
):
    parts = command.split(maxsplit=1)
    if len(parts) < 2:
        return {
            "messages": kwargs.get('messages'),
            "output": "Usage: /plan <command and schedule description>",
        }

    request = parts[1]
    platform_system = platform.system()

    linux_request = f"""Convert this scheduling request into a crontab-based script:
    Request: {request}

    """

    linux_prompt_static = """Example for "record CPU usage every 10 minutes":
    {
        "script": "
set -euo pipefail
IFS=$'\\n\\t'

LOGFILE=\"$HOME/.npcsh/logs/cpu_usage.log\"

log_info() {
    echo \"[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*\" >> \"$LOGFILE\"
}

log_error() {
    echo \"[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*\" >> \"$LOGFILE\"
}

record_cpu() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local cpu_usage=$(top -bn1 | grep 'Cpu(s)' | awk '{print $2}')
    log_info \"CPU Usage: $cpu_usage%\"
}

record_cpu",
        "schedule": "*/10 * * * *",
        "description": "Record CPU usage every 10 minutes",
        "name": "record_cpu_usage"
    }

    Your response must be valid json with the following keys:
    - script: The shell script content with proper functions and error handling. special characters must be escaped to ensure python json.loads will work correctly.
    - schedule: Crontab expression (5 fields: minute hour day month weekday)
    - description: A human readable description
    - name: A unique name for the job

    Do not include any additional markdown formatting in your response or leading ```json tags."""

    mac_request = f"""Convert this scheduling request into a launchd-compatible script:
    Request: {request}

    """

    mac_prompt_static = """Example for "record CPU usage every 10 minutes":
    {
        "script": "
set -euo pipefail
IFS=$'\\n\\t'

LOGFILE=\"$HOME/.npcsh/logs/cpu_usage.log\"

log_info() {
    echo \"[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*\" >> \"$LOGFILE\"
}

log_error() {
    echo \"[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*\" >> \"$LOGFILE\"
}

record_cpu() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local cpu_usage=$(top -l 1 | grep 'CPU usage' | awk '{print $3}' | tr -d '%')
    log_info \"CPU Usage: $cpu_usage%\"
}

record_cpu",
        "schedule": "600",
        "description": "Record CPU usage every 10 minutes",
        "name": "record_cpu_usage"
    }

    Your response must be valid json with the following keys:
    - script: The shell script content with proper functions and error handling. special characters must be escaped to ensure python json.loads will work correctly.
    - schedule: Interval in seconds (e.g. 600 for 10 minutes)
    - description: A human readable description
    - name: A unique name for the job

    Do not include any additional markdown formatting in your response or leading ```json tags."""

    windows_request = f"""Convert this scheduling request into a PowerShell script with Task Scheduler parameters:
    Request: {request}

    """

    windows_prompt_static = """Example for "record CPU usage every 10 minutes":
    {
        "script": "$ErrorActionPreference = 'Stop'

$LogFile = \"$HOME\\.npcsh\\logs\\cpu_usage.log\"

function Write-Log {
    param($Message, $Type = 'INFO')
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    \"[$timestamp] [$Type] $Message\" | Out-File -FilePath $LogFile -Append
}

function Get-CpuUsage {
    try {
        $cpu = (Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples.CookedValue
        Write-Log \"CPU Usage: $($cpu)%\"
    } catch {
        Write-Log $_.Exception.Message 'ERROR'
        throw
    }
}

Get-CpuUsage",
        "schedule": "/sc minute /mo 10",
        "description": "Record CPU usage every 10 minutes",
        "name": "record_cpu_usage"
    }

    Your response must be valid json with the following keys:
    - script: The PowerShell script content with proper functions and error handling. special characters must be escaped to ensure python json.loads will work correctly.
    - schedule: Task Scheduler parameters (e.g. /sc minute /mo 10)
    - description: A human readable description
    - name: A unique name for the job

    Do not include any additional markdown formatting in your response or leading ```json tags."""

    prompts = {
        "Linux": linux_request + linux_prompt_static,
        "Darwin": mac_request + mac_prompt_static,
        "Windows": windows_request + windows_prompt_static,
    }

    prompt = prompts[platform_system]
    response = get_llm_response(
        prompt, format="json", **kwargs
    )
    schedule_info = response.get("response")
    print("Received schedule info:", schedule_info)

    job_name = f"job_{schedule_info['name']}"
    sched = schedule_info['schedule']

    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if platform_system == "Windows":
        script_path = os.path.join(JOBS_DIR, f"{job_name}.ps1")
    else:
        script_path = os.path.join(JOBS_DIR, f"{job_name}.sh")

    log_path = os.path.join(LOGS_DIR, f"{job_name}.log")

    # Write the LLM-generated script directly (not via compile_job_script,
    # since the LLM already produced the full script content)
    with open(script_path, "w") as f:
        f.write(schedule_info["script"])
    os.chmod(script_path, 0o755)

    # Register with OS scheduler using the shared primitives
    if platform_system == "Linux":
        ok, msg = _schedule_crontab(script_path, sched, job_name, log_path)
    elif platform_system == "Darwin":
        ok, msg = _schedule_launchd(script_path, sched, job_name, log_path)
    elif platform_system == "Windows":
        ok, msg = _schedule_windows(script_path, sched, job_name, log_path)

    output = f"""Job created successfully:
- Description: {schedule_info['description']}
- Schedule: {sched}
- Script: {script_path}
- Log: {log_path}"""

    return {"messages": kwargs.get('messages'), "output": output}
