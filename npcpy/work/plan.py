
import platform

from npcpy.llm_funcs import get_llm_response

import subprocess
import os
import tempfile

from typing import Any

def execute_plan_command(
    command, 
    **kwargs,
):
    parts = command.split(maxsplit=1)
    if len(parts) < 2:
        return {
            "messages": messages,
            "output": "Usage: /plan <command and schedule description>",
        }

    request = parts[1]
    platform_system = platform.system()

    
    jobs_dir = os.path.expanduser("~/.npcsh/jobs")
    logs_dir = os.path.expanduser("~/.npcsh/logs")
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    
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
        prompt,  format="json", **kwargs
    )
    schedule_info = response.get("response")
    print("Received schedule info:", schedule_info)

    job_name = f"job_{schedule_info['name']}"

    if platform_system == "Windows":
        script_path = os.path.join(jobs_dir, f"{job_name}.ps1")
    else:
        script_path = os.path.join(jobs_dir, f"{job_name}.sh")

    log_path = os.path.join(logs_dir, f"{job_name}.log")

    
    with open(script_path, "w") as f:
        f.write(schedule_info["script"])
    os.chmod(script_path, 0o755)

    if platform_system == "Linux":
        try:
            current_crontab = subprocess.check_output(["crontab", "-l"], text=True)
        except subprocess.CalledProcessError:
            current_crontab = ""

        crontab_line = f"{schedule_info['schedule']} {script_path} >> {log_path} 2>&1"
        new_crontab = current_crontab.strip() + "\n" + crontab_line + "\n"

        with tempfile.NamedTemporaryFile(mode="w") as tmp:
            tmp.write(new_crontab)
            tmp.flush()
            subprocess.run(["crontab", tmp.name], check=True)

        output = f"""Job created successfully:
- Description: {schedule_info['description']}
- Schedule: {schedule_info['schedule']}
- Script: {script_path}
- Log: {log_path}
- Crontab entry: {crontab_line}"""

    elif platform_system == "Darwin":
        plist_dir = os.path.expanduser("~/Library/LaunchAgents")
        os.makedirs(plist_dir, exist_ok=True)
        plist_path = os.path.join(plist_dir, f"com.npcsh.{job_name}.plist")

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.npcsh.{job_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{script_path}</string>
    </array>
    <key>StartInterval</key>
    <integer>{schedule_info['schedule']}</integer>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>"""

        with open(plist_path, "w") as f:
            f.write(plist_content)

        subprocess.run(["launchctl", "unload", plist_path], check=False)
        subprocess.run(["launchctl", "load", plist_path], check=True)

        output = f"""Job created successfully:
- Description: {schedule_info['description']}
- Schedule: Every {schedule_info['schedule']} seconds
- Script: {script_path}
- Log: {log_path}
- Launchd plist: {plist_path}"""

    elif platform_system == "Windows":
        task_name = f"NPCSH_{job_name}"

        
        schedule_params = schedule_info["schedule"].split()

        cmd = (
            [
                "schtasks",
                "/create",
                "/tn",
                task_name,
                "/tr",
                f"powershell -NoProfile -ExecutionPolicy Bypass -File {script_path}",
            ]
            + schedule_params
            + ["/f"]
        )  

        subprocess.run(cmd, check=True)

        output = f"""Job created successfully:
- Description: {schedule_info['description']}
- Schedule: {schedule_info['schedule']}
- Script: {script_path}
- Log: {log_path}
- Task name: {task_name}"""

    return {"messages": kwargs.get('messages'), "output": output}

