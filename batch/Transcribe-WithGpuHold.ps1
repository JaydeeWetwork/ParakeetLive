<#
.SYNOPSIS
  Runs batch\Transcribe-Parakeet.ps1 (unchanged) while the Parakeet Live widget's model is held off the GPU.
.DESCRIPTION
  Added 2026-10-07 (log 52). The batch transcriber needs up to ~2.7 GB of the 4 GB GPU, and the live widget
  keeps its model (~1.4 GB) on the GPU by default, which does not fit together.
  Before the job: if the widget is running, ask it (window-message IPC to its tray window: same user and
  desktop, no network port) to move the model to RAM and hold it there. The lease is this PowerShell
  process: the widget keeps a handle to it and polls it every 2 s.
  After the job, including on errors and Ctrl+C (finally), release the hold: the widget's normal policy
  resumes (back on the GPU unless a game or another GPU-heavy app is running).
  If this process dies without releasing, the widget keeps the hold only while this process's children
  (the wsl.exe running transcribe.py) still run, then resumes by itself (6 h cap on any hold).
  If the widget is not running, or -Cpu is given, the batch runs exactly as before.
  All arguments are passed through unchanged.
.EXAMPLE
  .\Transcribe-WithGpuHold.ps1 "D:\Audio\talk.m4a" -OutDir "D:\Audio\transcripts"
#>
$real = Join-Path $PSScriptRoot 'Transcribe-Parakeet.ps1'
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $repo 'widget\.venv\Scripts\python.exe'
$app = Join-Path $repo 'widget\parakeet_live.pyw'

$held = $false
$wantGpu = -not ($args -contains '-Cpu')
if ($wantGpu -and (Test-Path -LiteralPath $py) -and (Test-Path -LiteralPath $app)) {
    try {
        & $py $app --cmd batchhold --pid $PID --wait 120 | Write-Host
        $rc = $LASTEXITCODE
        # 0 = model off the GPU, 4 = still finishing an utterance (it moves right after), 3 = widget not running
        if ($rc -eq 0 -or $rc -eq 4) { $held = $true }
    } catch {
        Write-Host ('Parakeet Live hold skipped: ' + $_.Exception.Message)
    }
}

$code = 1
try {
    & $real @args
    $code = $LASTEXITCODE
} finally {
    if ($held) {
        & $py $app --cmd batchrelease --pid $PID | Write-Host
    }
}
exit $code
