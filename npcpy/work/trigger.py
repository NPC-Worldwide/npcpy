import os 
import platform
import subprocess
from npcpy.llm_funcs import get_llm_response
def execute_trigger_command(
    command, npc=None, model=None, provider=None, messages=None, api_url=None
):
    parts = command.split(maxsplit=1)
    if len(parts) < 2:
        return {
            "messages": messages,
            "output": "Usage: /trigger <trigger condition and action description>",
        }

    request = parts[1]
    platform_system = platform.system()

    linux_request = f"""Convert this trigger request into a single event-monitoring daemon script:
    Request: {request}

    """

    linux_prompt_static = """Example for "Move PDFs from Downloads to Documents/PDFs":
    {
        "script": "#!/bin/bash\\nset -euo pipefail\\nIFS=$'\\n\\t'\\n\\nLOGFILE=\\\"$HOME/.npcsh/logs/pdf_mover.log\\\"\\nSOURCE=\\\"$HOME/Downloads\\\"\\nTARGET=\\\"$HOME/Documents/PDFs\\\"\\n\\nlog_info() {\\n    echo \\\"[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*\\\" >> \\\"$LOGFILE\\\"\\n}\\n\\nlog_error() {\\n    echo \\\"[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*\\\" >> \\\"$LOGFILE\\\"\\n}\\n\\ninotifywait -m -q -e create --format '%w%f' \\\"$SOURCE\\\" | while read filepath; do\\n    if [[ \\\"$filepath\\\" =~ \\\\.pdf$ ]]; then\\n        mv \\\"$filepath\\\" \\\"$TARGET/\\\" && log_info \\\"Moved $filepath to $TARGET\\\" || log_error \\\"Failed to move $filepath\\\"\\n    fi\\ndone",
        "name": "pdf_mover",
        "description": "Move PDF files from Downloads to Documents/PDFs folder"
    }

    The script MUST:
    - Use inotifywait -m -q -e create --format '%w%f' to get full paths
    - Double quote ALL file operations: "$SOURCE/$FILE"
    - Use $HOME for absolute paths
    - Echo both success and failure messages to log

    Your response must be valid json with the following keys:
    - script: The shell script content with proper functions and error handling
    - name: A unique name for the trigger
    - description: A human readable description

    Do not include any additional markdown formatting in your response."""

    mac_request = f"""Convert this trigger request into a single event-monitoring daemon script:
    Request: {request}

    """

    mac_prompt_static = """Example for "Move PDFs from Downloads to Documents/PDFs":
    {
        "script": "#!/bin/bash\\nset -euo pipefail\\nIFS=$'\\n\\t'\\n\\nLOGFILE=\\\"$HOME/.npcsh/logs/pdf_mover.log\\\"\\nSOURCE=\\\"$HOME/Downloads\\\"\\nTARGET=\\\"$HOME/Documents/PDFs\\\"\\n\\nlog_info() {\\n    echo \\\"[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*\\\" >> \\\"$LOGFILE\\\"\\n}\\n\\nlog_error() {\\n    echo \\\"[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*\\\" >> \\\"$LOGFILE\\\"\\n}\\n\\nfswatch -0 -r -e '.*' --event Created --format '%p' \\\"$SOURCE\\\" | while read -d '' filepath; do\\n    if [[ \\\"$filepath\\\" =~ \\\\.pdf$ ]]; then\\n        mv \\\"$filepath\\\" \\\"$TARGET/\\\" && log_info \\\"Moved $filepath to $TARGET\\\" || log_error \\\"Failed to move $filepath\\\"\\n    fi\\ndone",
        "name": "pdf_mover",
        "description": "Move PDF files from Downloads to Documents/PDFs folder"
    }

    The script MUST:
    - Use fswatch -0 -r -e '.*' --event Created --format '%p' to get full paths
    - Double quote ALL file operations: "$SOURCE/$FILE"
    - Use $HOME for absolute paths
    - Echo both success and failure messages to log

    Your response must be valid json with the following keys:
    - script: The shell script content with proper functions and error handling
    - name: A unique name for the trigger
    - description: A human readable description

    Do not include any additional markdown formatting in your response."""

    windows_request = f"""Convert this trigger request into a single event-monitoring daemon script:
    Request: {request}

    """

    windows_prompt_static = """Example for "Move PDFs from Downloads to Documents/PDFs":
    {
        "script": "$ErrorActionPreference = 'Stop'\\n\\n$LogFile = \\\"$HOME\\.npcsh\\logs\\pdf_mover.log\\\"\\n$Source = \\\"$HOME\\Downloads\\\"\\n$Target = \\\"$HOME\\Documents\\PDFs\\\"\\n\\nfunction Write-Log {\\n    param($Message, $Type = 'INFO')\\n    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'\\n    \\\"[$timestamp] [$Type] $Message\\\" | Out-File -FilePath $LogFile -Append\\n}\\n\\n$watcher = New-Object System.IO.FileSystemWatcher\\n$watcher.Path = $Source\\n$watcher.Filter = \\\"*.pdf\\\"\\n$watcher.IncludeSubdirectories = $true\\n$watcher.EnableRaisingEvents = $true\\n\\n$action = {\\n    $path = $Event.SourceEventArgs.FullPath\\n    try {\\n        Move-Item -Path $path -Destination $Target\\n        Write-Log \\\"Moved $path to $Target\\\"\\n    } catch {\\n        Write-Log $_.Exception.Message 'ERROR'\\n    }\\n}\\n\\nRegister-ObjectEvent $watcher 'Created' -Action $action\\n\\nwhile ($true) { Start-Sleep 1 }",
        "name": "pdf_mover",
        "description": "Move PDF files from Downloads to Documents/PDFs folder"
    }

    The script MUST:
    - Use FileSystemWatcher for monitoring
    - Double quote ALL file operations: "$Source\\$File"
    - Use $HOME for absolute paths
    - Echo both success and failure messages to log

    Your response must be valid json with the following keys:
    - script: The PowerShell script content with proper functions and error handling
    - name: A unique name for the trigger
    - description: A human readable description

    Do not include any additional markdown formatting in your response."""

    prompts = {
        "Linux": linux_request + linux_prompt_static,
        "Darwin": mac_request + mac_prompt_static,
        "Windows": windows_request + windows_prompt_static,
    }

    prompt = prompts[platform_system]
    response = get_llm_response(
        prompt, npc=npc, model=model, provider=provider, format="json"
    )
    trigger_info = response.get("response")
    print("Trigger info:", trigger_info)

    from npcpy.npc_sysenv import get_triggers_dir, get_logs_dir
    triggers_dir = get_triggers_dir()
    logs_dir = get_logs_dir()
    os.makedirs(triggers_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    trigger_name = f"trigger_{trigger_info['name']}"
    log_path = os.path.join(logs_dir, f"{trigger_name}.log")

    if platform_system == "Linux":
        script_path = os.path.join(triggers_dir, f"{trigger_name}.sh")

        with open(script_path, "w") as f:
            f.write(trigger_info["script"])
        os.chmod(script_path, 0o755)

        service_dir = os.path.expanduser("~/.config/systemd/user")
        os.makedirs(service_dir, exist_ok=True)
        service_path = os.path.join(service_dir, f"npcsh-{trigger_name}.service")

        service_content = f"""[Unit]
Description={trigger_info['description']}
After=network.target

[Service]
Type=simple
ExecStart={script_path}
Restart=always
StandardOutput=append:{log_path}
StandardError=append:{log_path}

[Install]
WantedBy=default.target
"""

        with open(service_path, "w") as f:
            f.write(service_content)

        subprocess.run(["systemctl", "--user", "daemon-reload"])
        subprocess.run(
            ["systemctl", "--user", "enable", f"npcsh-{trigger_name}.service"]
        )
        subprocess.run(
            ["systemctl", "--user", "start", f"npcsh-{trigger_name}.service"]
        )

        status = subprocess.run(
            ["systemctl", "--user", "status", f"npcsh-{trigger_name}.service"],
            capture_output=True,
            text=True,
        )

        output = f"""Trigger service created:
- Description: {trigger_info['description']}
- Script: {script_path}
- Service: {service_path}
- Log: {log_path}

Status:
{status.stdout}"""

    elif platform_system == "Darwin":
        script_path = os.path.join(triggers_dir, f"{trigger_name}.sh")

        with open(script_path, "w") as f:
            f.write(trigger_info["script"])
        os.chmod(script_path, 0o755)

        plist_dir = os.path.expanduser("~/Library/LaunchAgents")
        os.makedirs(plist_dir, exist_ok=True)
        plist_path = os.path.join(plist_dir, f"com.npcsh.{trigger_name}.plist")

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.npcsh.{trigger_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{script_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
</dict>
</plist>"""

        with open(plist_path, "w") as f:
            f.write(plist_content)

        subprocess.run(["launchctl", "unload", plist_path], check=False)
        subprocess.run(["launchctl", "load", plist_path], check=True)

        output = f"""Trigger service created:
- Description: {trigger_info['description']}
- Script: {script_path}
- Launchd plist: {plist_path}
- Log: {log_path}"""

    elif platform_system == "Windows":
        script_path = os.path.join(triggers_dir, f"{trigger_name}.ps1")

        with open(script_path, "w") as f:
            f.write(trigger_info["script"])

        task_name = f"NPCSH_{trigger_name}"

        
        cmd = [
            "schtasks",
            "/create",
            "/tn",
            task_name,
            "/tr",
            f"powershell -NoProfile -ExecutionPolicy Bypass -File {script_path}",
            "/sc",
            "onstart",
            "/ru",
            "System",
            "/f",  
        ]

        subprocess.run(cmd, check=True)

        
        subprocess.run(["schtasks", "/run", "/tn", task_name])

        output = f"""Trigger service created:
- Description: {trigger_info['description']}
- Script: {script_path}
- Task name: {task_name}
- Log: {log_path}"""

    return {"messages": messages, "output": output}
