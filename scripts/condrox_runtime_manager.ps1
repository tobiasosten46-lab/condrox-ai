#Requires -Version 5.1
<#
.SYNOPSIS
    Condrox AI local runtime orchestrator (no Docker).
.DESCRIPTION
    Boots the Condrox AI stack on Windows:
      - API server (uvicorn + FastAPI) on port 8000
      - Dashboard (Vite + React) on port 3000
      - This control terminal with start/stop/restart/status/logs commands
.PARAMETER Command
    One of: start, stop, restart, status, logs, api, dash, shell, test, help.
.EXAMPLE
    .\scripts\condrox_runtime_manager.ps1 start
#>
param(
    [Parameter(Position = 0)]
    [string]$Command = "start"
)

$ErrorActionPreference = "Stop"

# ─── Configuration ───────────────────────────────────────────────────────────
$ProjectRoot = (Split-Path -Parent $PSScriptRoot | Resolve-Path).Path
$RuntimeDir  = Join-Path $ProjectRoot "runtime"
$ApiPort     = if ($env:CONDROX_API_PORT) { $env:CONDROX_API_PORT } else { 8000 }
$DashPort    = if ($env:CONDROX_DASH_PORT) { $env:CONDROX_DASH_PORT } else { 3000 }
$Python      = if ($env:CONDROX_PYTHON) { $env:CONDROX_PYTHON } else { "python" }
$Npm         = if ($env:CONDROX_NPM) { $env:CONDROX_NPM } else { "npm" }

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null

function Get-PidFile { param($name) Join-Path $RuntimeDir "$name.pid" }
function Get-LogFile { param($name) Join-Path $RuntimeDir "$name.log" }

# ─── Banner ──────────────────────────────────────────────────────────────────
function Show-Banner {
    Clear-Host
    Write-Host ""
    Write-Host "   ____             _                         _    ___ ____ " -ForegroundColor Cyan
    Write-Host "  / ___|___ ___  __| | ___ _ __ ___  ___  ___| |  |_ _|  _ \" -ForegroundColor Cyan
    Write-Host " | |   / __/ _ \\/ _` |/ _ \ '__/ _ \/ _ \/ __| |   | || |_) |" -ForegroundColor Cyan
    Write-Host " | |__| (_| (_) | (_| |  __| | |  __/  __/\\__ \ |  | ||  __/ " -ForegroundColor Cyan
    Write-Host "  \\____\\___\\___/ \\__,_|\\___|_|  \\___|\\___||___/ | |___|_|    " -ForegroundColor Cyan
    Write-Host "                                                  Local Runtime" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Project: $ProjectRoot" -ForegroundColor Gray
    Write-Host "  API:     http://localhost:$ApiPort" -ForegroundColor Gray
    Write-Host "  Dashboard: http://localhost:$DashPort" -ForegroundColor Gray
    Write-Host ""
}

# ─── Service helpers ─────────────────────────────────────────────────────────
function Start-ServiceProcess {
    param(
        [string]$Name,
        [string]$WorkingDirectory,
        [string]$ArgumentList,
        [string]$WindowTitle
    )
    $pidFile = Get-PidFile $Name
    $logFile = Get-LogFile $Name
    if (Test-Path $pidFile) {
        $oldPid = Get-Content $pidFile -Raw
        try {
            $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
            if ($proc -and !$proc.HasExited) {
                Write-Host "  $Name is already running (PID $oldPid)." -ForegroundColor Yellow
                return
            }
        } catch {}
    }

    # Write a temporary launch script to avoid complex nested quoting.
    $tempScript = Join-Path $RuntimeDir "$Name-service.ps1"
    $serviceScript = @"
Set-Location '$WorkingDirectory'
`$host.ui.RawUI.WindowTitle = '$WindowTitle'
`$env:CONDROX_API_PORT = '$ApiPort'
$ArgumentList | Tee-Object -FilePath '$logFile'
"@
    $serviceScript | Out-File -FilePath $tempScript -Encoding UTF8 -Force

    $proc = Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoExit", "-File", $tempScript `
        -WorkingDirectory $WorkingDirectory `
        -PassThru
    $proc.Id | Out-File -FilePath $pidFile -Force
    Write-Host "  Started $Name (PID $($proc.Id)) -> $logFile" -ForegroundColor Green
}

function Stop-ServiceProcess {
    param([string]$Name)
    $pidFile = Get-PidFile $Name
    if (-not (Test-Path $pidFile)) {
        Write-Host "  $Name is not running (no PID file)." -ForegroundColor Yellow
        return
    }
    $id = Get-Content $pidFile -Raw
    try {
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($proc -and !$proc.HasExited) {
            Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
            Write-Host "  Stopped $Name (PID $id)." -ForegroundColor Green
        } else {
            Write-Host "  $Name process already exited." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  Could not stop $Name (PID $id): $_" -ForegroundColor Red
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

function Get-ServiceStatus {
    param([string]$Name)
    $pidFile = Get-PidFile $Name
    if (-not (Test-Path $pidFile)) { return "stopped" }
    $id = Get-Content $pidFile -Raw
    try {
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($proc -and !$proc.HasExited) { return "running (PID $id)" }
        return "dead (stale PID $id)"
    } catch { return "dead (stale PID $id)" }
}

# ─── Health checks ───────────────────────────────────────────────────────────
function Wait-ForApi {
    param([int]$TimeoutSeconds = 30)
    $url = "http://localhost:$ApiPort/health"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        try {
            $r = Invoke-RestMethod -Uri $url -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($r) { return $true }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    return $false
}

# ─── Commands ────────────────────────────────────────────────────────────────
function Start-Stack {
    Show-Banner
    Write-Host "Starting Condrox AI stack..." -ForegroundColor Cyan
    Start-ServiceProcess -Name "api" -WorkingDirectory $ProjectRoot -ArgumentList "$Python -m uvicorn api.main:app --host 0.0.0.0 --port $ApiPort" -WindowTitle "Condrox AI - API Server"
    if (Wait-ForApi -TimeoutSeconds 30) {
        Write-Host "  API health check passed." -ForegroundColor Green
    } else {
        Write-Host "  API health check timed out - check runtime/api.log" -ForegroundColor Red
    }
    Start-ServiceProcess -Name "dashboard" -WorkingDirectory (Join-Path $ProjectRoot "dashboard") -ArgumentList "$Npm run dev" -WindowTitle "Condrox AI - Dashboard"
    Show-Status
    Enter-ControlLoop
}

function Stop-Stack {
    Write-Host "Stopping Condrox AI stack..." -ForegroundColor Cyan
    Stop-ServiceProcess "dashboard"
    Stop-ServiceProcess "api"
}

function Restart-Stack {
    param([string]$Target = "all")
    if ($Target -eq "all" -or $Target -eq "api") {
        Stop-ServiceProcess "api"
        Start-ServiceProcess -Name "api" -WorkingDirectory $ProjectRoot -ArgumentList "$Python -m uvicorn api.main:app --host 0.0.0.0 --port $ApiPort" -WindowTitle "Condrox AI - API Server"
    }
    if ($Target -eq "all" -or $Target -eq "dashboard") {
        Stop-ServiceProcess "dashboard"
        Start-ServiceProcess -Name "dashboard" -WorkingDirectory (Join-Path $ProjectRoot "dashboard") -ArgumentList "$Npm run dev" -WindowTitle "Condrox AI - Dashboard"
    }
    Show-Status
}

function Show-Status {
    Write-Host ""
    Write-Host "Condrox AI Runtime Status" -ForegroundColor Cyan
    Write-Host "  API:       $(Get-ServiceStatus 'api')"
    Write-Host "  Dashboard: $(Get-ServiceStatus 'dashboard')"
    Write-Host "  API URL:   http://localhost:$ApiPort"
    Write-Host "  Dash URL:  http://localhost:$DashPort"
    Write-Host ""
}

function Invoke-HealthCheck {
    Write-Host ""
    Write-Host "Condrox Runtime Health Monitor (CROSC)" -ForegroundColor Cyan

    $apiStatus = "UNKNOWN"
    try {
        $resp = Invoke-RestMethod -Uri "http://localhost:$ApiPort/health" -TimeoutSec 3 -ErrorAction Stop
        $apiStatus = if ($resp.status -eq "healthy") { "HEALTHY" } else { "DEGRADED" }
    } catch {
        $apiStatus = "UNREACHABLE"
    }
    $colorApi = switch ($apiStatus) { "HEALTHY" { "Green" } "DEGRADED" { "Yellow" } default { "Red" } }
    Write-Host "  API /health: $apiStatus" -ForegroundColor $colorApi

    $dashStatus = "UNKNOWN"
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $dashStatus = if ($tcp.ConnectAsync('localhost', $DashPort).Wait(1000)) { "LISTENING" } else { "UNREACHABLE" }
        $tcp.Close()
    } catch {
        $dashStatus = "UNREACHABLE"
    }
    $colorDash = if ($dashStatus -eq "LISTENING") { "Green" } else { "Red" }
    Write-Host "  Dashboard port ${DashPort}: $dashStatus" -ForegroundColor $colorDash

    $apiProc = Get-ServiceStatus 'api'
    $dashProc = Get-ServiceStatus 'dashboard'
    Write-Host "  API process: $apiProc"
    Write-Host "  Dashboard process: $dashProc"

    $overall = if ($apiStatus -eq "HEALTHY" -and $dashStatus -eq "LISTENING") { "OK" } else { "ISSUES DETECTED" }
    $colorOverall = if ($overall -eq "OK") { "Green" } else { "Red" }
    Write-Host ""
    Write-Host "  Overall: $overall" -ForegroundColor $colorOverall
    Write-Host ""
}

function Show-Logs {
    param([string]$Name = "api", [int]$Tail = 50)
    $logFile = Get-LogFile $Name
    if (-not (Test-Path $logFile)) {
        Write-Host "No log file for $Name." -ForegroundColor Yellow
        return
    }
    Write-Host "--- $Name log (last $Tail lines) ---" -ForegroundColor Cyan
    Get-Content $logFile -Tail $Tail -ErrorAction SilentlyContinue | Write-Host
}

function Open-Browser {
    param([string]$Url)
    Start-Process $Url
}

function Invoke-SmokeTest {
    Write-Host "Running import smoke check..." -ForegroundColor Cyan
    $env:CONDROX_SKIP_KNOWLEDGE_LOAD = "1"
    $tempPy = Join-Path $RuntimeDir "smoke_test.py"
    @"
import sys
sys.path.insert(0, r'$ProjectRoot')
from core import BrainController
bc = BrainController()
print('BOOT:', bc.boot_system().success)
"@ | Out-File -FilePath $tempPy -Encoding UTF8 -Force
    & $Python $tempPy
}

function Invoke-QualityCheck {
    Write-Host "Running Condrox Code Quality System..." -ForegroundColor Cyan
    & $Python "$ProjectRoot\scripts\code_quality\run_quality_check.py" --format text | Select-Object -First 20
    Write-Host "`nRun 'python scripts/code_quality/run_quality_check.py' for full JSON output." -ForegroundColor Gray
}

function Invoke-SystemVerification {
    Write-Host "Running Condrox AI System Verification..." -ForegroundColor Cyan
    & $Python "$ProjectRoot\scripts\verify_system.py" --format text | Select-Object -First 40
    Write-Host "`nRun 'python scripts/verify_system.py --format json' for full JSON output." -ForegroundColor Gray
}

function Invoke-EvidenceGeneration {
    Write-Host "Running Condrox Evidence Engine..." -ForegroundColor Cyan
    & $Python "$ProjectRoot\scripts\generate_evidence.py"
}

function Invoke-EvidencePortal {
    Write-Host "Opening Condrox Evidence Portal..." -ForegroundColor Cyan
    & $Python "$ProjectRoot\scripts\generate_evidence.py" *> $null
    $portal = "$ProjectRoot\Reports\evidence\index.html"
    if (Test-Path $portal) {
        Start-Process $portal
        Write-Host "Portal opened: $portal" -ForegroundColor Green
    } else {
        Write-Host "Portal not found. Run 'condrox evidence' first." -ForegroundColor Red
    }
}

function Enter-ControlLoop {
    Write-Host "Entering AI control terminal. Type 'help' for commands." -ForegroundColor Cyan
    while ($true) {
        $line = Read-Host ">>>"
        $parts = $line -split '\s+', 2
        $cmd = $parts[0].ToLower()
        $arg = if ($parts.Length -gt 1) { $parts[1] } else { "" }
        switch ($cmd) {
            "help" {
                Write-Host "Commands: start, stop, restart [api|dashboard|all], status, health, verify, evidence, evidence-portal, logs [api|dashboard|control] [N], api, dash, shell, test, quality, exit" -ForegroundColor Yellow
            }
            "verify" { Invoke-SystemVerification }
            "evidence" { Invoke-EvidenceGeneration }
            "evidence-portal" { Invoke-EvidencePortal }
            "quality" { Invoke-QualityCheck }
            "health" { Invoke-HealthCheck }
            "start" { Show-Status }
            "stop" { Stop-Stack; return }
            "restart" { Restart-Stack -Target ($arg -or "all") }
            "status" { Show-Status }
            "logs" {
                $a = $arg -split '\s+'
                $n = if ($a[1] -match '^\d+$') { [int]$a[1] } else { 50 }
                Show-Logs -Name ($a[0] -or "api") -Tail $n
            }
            "api" { Open-Browser "http://localhost:$ApiPort/docs" }
            "dash" { Open-Browser "http://localhost:$DashPort" }
            "shell" { Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$ProjectRoot'; $Python" }
            "test" { Invoke-SmokeTest }
            "exit" { Stop-Stack; return }
            default { Write-Host "Unknown command. Type 'help'." -ForegroundColor Red }
        }
    }
}

# ─── Entry point ─────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    "start" { Start-Stack }
    "boot" { Start-Stack }
    "stop" { Stop-Stack }
    "restart" { Restart-Stack -Target ($args[0] -or "all") }
    "status" { Show-Status }
    "health" { Invoke-HealthCheck }
    "logs" { Show-Logs -Name ($args[0] -or "api") -Tail ([int]($args[1] -or 50)) }
    "api" { Open-Browser "http://localhost:$ApiPort/docs" }
    "dash" { Open-Browser "http://localhost:$DashPort" }
    "test" { Invoke-SmokeTest }
    "quality" { Invoke-QualityCheck }
    "verify" { Invoke-SystemVerification }
    "evidence" { Invoke-EvidenceGeneration }
    "evidence-portal" { Invoke-EvidencePortal }
    default {
        Write-Host "Usage: condrox {start|stop|restart|status|health|verify|evidence|evidence-portal|logs|api|dash|test|quality|help}" -ForegroundColor Yellow
    }
}
