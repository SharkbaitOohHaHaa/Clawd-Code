param(
    [switch]$SelfTest
)

$ErrorActionPreference = 'SilentlyContinue'
$version = '0.1.0'
$timeoutSeconds = 3
$statePath = Join-Path $PSScriptRoot 'upstream-review-state.json'

function Write-Skipped([string]$reason) {
    Write-Host "Update check skipped ($reason)." -ForegroundColor DarkGray
}

function Read-ReviewState {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Test-ReviewState($state) {
    if (-not $state) { return $false }
    if ($state.repo -ne 'GPT-AGI/Clawd-Code') { return $false }
    if ($state.branch -ne 'main') { return $false }
    return [bool]([string]$state.reviewed_sha -match '^[0-9a-f]{40}$')
}
function Get-RemoteHead($state) {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if (-not $git) { return $null }

    $info = New-Object System.Diagnostics.ProcessStartInfo
    $info.FileName = $git.Source
    $info.Arguments = "ls-remote https://github.com/$($state.repo).git refs/heads/$($state.branch)"
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.EnvironmentVariables['GIT_TERMINAL_PROMPT'] = '0'

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $info
    try {
        if (-not $process.Start()) { return $null }
        if (-not $process.WaitForExit($timeoutSeconds * 1000)) {
            try { $process.Kill() } catch {}
            return $null
        }
        if ($process.ExitCode -ne 0) { return $null }
        $line = $process.StandardOutput.ReadLine()
        if (-not $line) { return $null }

        $sha = (($line -split '\s+')[0]).Trim()
        if ($sha -notmatch '^[0-9a-f]{40}$') { return $null }
        return $sha
    } finally {
        $process.Dispose()
    }
}

function Get-UpdateStatus([string]$reviewed, [string]$remote) {
    if ($reviewed -eq $remote) { return 'current' }
    return 'available'
}
if ($SelfTest) {
    $failures = @()
    $a = '1111111111111111111111111111111111111111'
    $b = '2222222222222222222222222222222222222222'
    if ((Get-UpdateStatus $a $a) -ne 'current') {
        $failures += 'equal SHA values must be current'
    }
    if ((Get-UpdateStatus $a $b) -ne 'available') {
        $failures += 'different SHA values must be available'
    }
    $valid = [pscustomobject]@{ repo='GPT-AGI/Clawd-Code'; branch='main'; reviewed_sha=$a }
    if (-not (Test-ReviewState $valid)) {
        $failures += 'valid review state was rejected'
    }
    $invalid = [pscustomobject]@{ repo='GPT-AGI/Clawd-Code'; branch='main'; reviewed_sha='bad' }
    if (Test-ReviewState $invalid) {
        $failures += 'invalid review state was accepted'
    }
    if ($failures.Count -gt 0) {
        $failures | ForEach-Object { Write-Error $_ }
        exit 1
    }
    Write-Host 'Startup updater self-test passed.'
    exit 0
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Skipped 'git unavailable'
    exit 0
}

$state = Read-ReviewState
if (-not (Test-ReviewState $state)) {
    Write-Skipped 'review state unavailable'
    exit 0
}

$remote = Get-RemoteHead $state
if (-not $remote) {
    Write-Skipped 'offline'
    exit 0
}

$status = Get-UpdateStatus ([string]$state.reviewed_sha) $remote
if ($status -eq 'current') {
    Write-Host "Clawd Codex v$version - upstream current." -ForegroundColor DarkGray
    exit 0
}

Write-Host "Clawd Codex v$version - upstream update available; review required." -ForegroundColor Yellow
Write-Host 'Run "Review Clawd Updates.cmd" when you want to inspect it.' -ForegroundColor DarkGray
exit 0
