<#
.SYNOPSIS
  Transcribe audio/video locally with NVIDIA Parakeet TDT 0.6B v2 (GPU, inside WSL Ubuntu-24.04).
.EXAMPLE
  .\Transcribe-Parakeet.ps1 "C:\path\to\meeting.mp4"
.EXAMPLE
  .\Transcribe-Parakeet.ps1 "D:\Audio\talk.m4a" -OutDir "D:\Audio\transcripts"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Path,
    [string]$OutDir,
    [ValidateSet('auto', 'on', 'off')][string]$LocalAttn = 'auto',
    [double]$ChunkMin = 0,
    [switch]$Cpu,
    [switch]$CudaGraphs
)
$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = '1'
$Distro = 'Ubuntu-24.04'

function ConvertTo-WslPath([string]$p) {
    $ErrorActionPreference = 'Continue'
    $full = (Resolve-Path -LiteralPath $p -ErrorAction SilentlyContinue)
    if ($full) { $p = $full.ProviderPath } else { $p = [IO.Path]::GetFullPath($p) }
    $out = & wsl.exe -d $Distro -u root -e wslpath -a -u $p
    if ($LASTEXITCODE -ne 0 -or -not $out) { throw "wslpath could not convert: $p" }
    return ($out | Select-Object -First 1).Trim()
}

$wslArgs = @()
foreach ($p in $Path) {
    if (-not (Test-Path -LiteralPath $p)) { throw "File not found: $p" }
    $wslArgs += (ConvertTo-WslPath $p)
}
if ($OutDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $wslArgs += @('-o', (ConvertTo-WslPath $OutDir))
}
$wslArgs += @('--local-attn', $LocalAttn)
if ($ChunkMin -gt 0) { $wslArgs += @('--chunk-min', "$ChunkMin") }
if ($Cpu) { $wslArgs += '--cpu' }
if ($CudaGraphs) { $wslArgs += '--cuda-graphs' }

# NeMo prints progress/warnings on stderr; don't let PowerShell treat that as a failure.
$ErrorActionPreference = 'Continue'
& wsl.exe -d $Distro -u root -e /opt/parakeet/venv/bin/python /opt/parakeet/scripts/transcribe.py @wslArgs
exit $LASTEXITCODE
