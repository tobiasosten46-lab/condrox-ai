#Requires -Version 5.1
<#
.SYNOPSIS
    Starts the Condrox Knowledge Ingestion Engine for a full pre-download.
.DESCRIPTION
    Runs scripts/knowledge_ingestion_engine.py with max_samples=0 and writes
    a log to data/knowledge\ingestion.log. The process is detached
    so it can continue after the IDE session ends.
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Config = "$ProjectRoot\config\knowledge_sources.json"
$Engine = "$ProjectRoot\scripts\knowledge_ingestion_engine.py"
$KnowledgeRoot = "data/knowledge"
$LogFile = "$KnowledgeRoot\ingestion.log"

New-Item -ItemType Directory -Path $KnowledgeRoot -Force | Out-Null

$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

Start-Process -FilePath "python" `
    -ArgumentList "`"$Engine`" --config `"$Config`" --root `"$KnowledgeRoot`" --max-samples 0" `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError "$KnowledgeRoot\ingestion.err.log" `
    -WindowStyle Hidden

Write-Host "Full ingestion started. Log: $LogFile"
