param(
    [switch]$MarkReviewed,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Continue'
$statePath = Join-Path $PSScriptRoot 'upstream-review-state.json'
$referencePath = Join-Path $PSScriptRoot 'upstream-reference'
$clawdRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$liveRoot = Join-Path $clawdRoot 'Clawd Codex v0.1.0'

$auxSources = @(
    [pscustomobject]@{ Name='DeepSeek Harness'; Path=(Join-Path $PSScriptRoot 'provider-sources\deepseek-harness'); Repo='deepseek-ai/deepseek-harness'; Branch='master' },
    [pscustomobject]@{ Name='Qwen 3.8'; Path=(Join-Path $PSScriptRoot 'provider-sources\qwen3.8'); Repo='QwenLM/Qwen3.8'; Branch='main' },
    [pscustomobject]@{ Name='Anthropic Skills'; Path=(Join-Path $PSScriptRoot 'skill-sources\anthropics-skills'); Repo='anthropics/skills'; Branch='main' },
    [pscustomobject]@{ Name='Compound Engineering'; Path=(Join-Path $PSScriptRoot 'skill-sources\every-compound-engineering'); Repo='EveryInc/compound-engineering-plugin'; Branch='main' },
    [pscustomobject]@{ Name='Superpowers'; Path=(Join-Path $PSScriptRoot 'skill-sources\obra-superpowers'); Repo='obra/superpowers'; Branch='main' },
    [pscustomobject]@{ Name='Trail of Bits Skills'; Path=(Join-Path $PSScriptRoot 'skill-sources\trailofbits-skills-curated'); Repo='trailofbits/skills-curated'; Branch='main' }
)

function Read-ReviewState {
    if (-not (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        throw "Review state is missing: $statePath"
    }
    $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($state.repo -ne 'GPT-AGI/Clawd-Code' -or $state.branch -ne 'main') {
        throw 'Review state repository or branch is invalid.'
    }
    if ([string]$state.reviewed_sha -notmatch '^[0-9a-f]{40}$') {
        throw 'Review state SHA is invalid.'
    }
    return $state
}

function Write-ReviewState($state, [string]$sha) {
    $next = [ordered]@{
        repo = [string]$state.repo
        branch = [string]$state.branch
        reviewed_sha = $sha
        reviewed_at = (Get-Date).ToString('o')
    }
    $temp = "$statePath.tmp"
    $next | ConvertTo-Json | Set-Content -LiteralPath $temp -Encoding UTF8
    Move-Item -LiteralPath $temp -Destination $statePath -Force
}

function Get-PathClassification(
    [string]$repoPath,
    [string]$reviewedSha,
    [string]$root,
    [string]$changedPath
) {
    $spec = "${reviewedSha}:$changedPath"
    & git -C $repoPath cat-file -e $spec 2>$null
    $baselineExists = ($LASTEXITCODE -eq 0)
    $localPath = Join-Path $root ($changedPath -replace '/', '\')
    $localExists = Test-Path -LiteralPath $localPath -PathType Leaf

    if ($baselineExists -and -not $localExists) {
        return 'deleted'
    }
    if (-not $baselineExists) {
        if ($localExists) { return 'modified' }
        return 'untouched'
    }

    $baselineHash = (& git -C $repoPath rev-parse $spec 2>$null | Select-Object -First 1)
    if (-not $baselineHash) {
        throw "Unable to hash reviewed upstream file: $changedPath"
    }
    $localHash = (& git -C $repoPath hash-object "--path=$changedPath" -- $localPath 2>$null | Select-Object -First 1)
    if (-not $localHash) {
        throw "Unable to hash local file: $changedPath"
    }

    if ($baselineHash.Trim() -eq $localHash.Trim()) {
        return 'untouched'
    }
    return 'modified'
}

function Show-Category([string]$title, $items) {
    Write-Host ""
    Write-Host "$title ($($items.Count))" -ForegroundColor Cyan
    if ($items.Count -eq 0) {
        Write-Host '  (none)' -ForegroundColor DarkGray
        return
    }
    $items | Sort-Object | ForEach-Object { Write-Host "  $_" }
}
function Get-AuxRemoteHead($source) {
    $env:GIT_TERMINAL_PROMPT = '0'
    $line = & git ls-remote ("https://github.com/" + $source.Repo + ".git") ("refs/heads/" + $source.Branch) 2>$null |
        Select-Object -First 1
    if (-not $line) { return $null }
    $sha = (($line -split '\s+')[0]).Trim()
    if ($sha -notmatch '^[0-9a-f]{40}$') { return $null }
    return $sha
}

function Show-AuxiliaryStatus {
    Write-Host ""
    Write-Host 'Other configured source status' -ForegroundColor Cyan
    foreach ($source in $auxSources) {
        if (-not (Test-Path -LiteralPath (Join-Path $source.Path '.git'))) {
            Write-Host "  $($source.Name): local source missing" -ForegroundColor DarkGray
            continue
        }
        $local = (& git -C $source.Path rev-parse HEAD 2>$null | Select-Object -First 1)
        $remote = Get-AuxRemoteHead $source
        if (-not $remote) {
            Write-Host "  $($source.Name): check unavailable" -ForegroundColor DarkGray
        } elseif ($local -and $local.Trim() -eq $remote) {
            Write-Host "  $($source.Name): current" -ForegroundColor DarkGray
        } else {
            Write-Host "  $($source.Name): update available" -ForegroundColor Yellow
        }
    }
}
function Invoke-ClassificationSelfTest {
    $temp = Join-Path $env:TEMP ("clawd-update-review-selftest-" + [guid]::NewGuid().ToString('N'))
    $repo = Join-Path $temp 'upstream'
    $live = Join-Path $temp 'live'
    New-Item -ItemType Directory -Path $repo, $live -Force | Out-Null
    try {
        & git -C $repo init -q
        & git -C $repo config user.name 'Clawd Self Test'
        & git -C $repo config user.email 'clawd-selftest@example.invalid'
        Set-Content -LiteralPath (Join-Path $repo 'deleted.txt') -Value 'base'
        Set-Content -LiteralPath (Join-Path $repo 'modified.txt') -Value 'base'
        Set-Content -LiteralPath (Join-Path $repo 'untouched.txt') -Value 'same'
        & git -C $repo add .
        & git -C $repo commit -q -m baseline
        $sha = (& git -C $repo rev-parse HEAD).Trim()

        Set-Content -LiteralPath (Join-Path $live 'modified.txt') -Value 'local-change'
        Set-Content -LiteralPath (Join-Path $live 'untouched.txt') -Value 'same'
        Set-Content -LiteralPath (Join-Path $live 'local-collision.txt') -Value 'ours'

        $cases = [ordered]@{
            'deleted.txt' = 'deleted'
            'modified.txt' = 'modified'
            'untouched.txt' = 'untouched'
            'new-upstream.txt' = 'untouched'
            'local-collision.txt' = 'modified'
        }
        $failures = @()
        foreach ($entry in $cases.GetEnumerator()) {
            $actual = Get-PathClassification $repo $sha $live $entry.Key
            if ($actual -ne $entry.Value) {
                $failures += "$($entry.Key): expected=$($entry.Value) actual=$actual"
            }
        }
        if ($failures.Count -gt 0) {
            throw ($failures -join '; ')
        }
        Write-Host 'Review classifier self-test passed: 5 cases.'
    } finally {
        Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($SelfTest) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw 'Git is required for the review self-test.'
    }
    Invoke-ClassificationSelfTest
    exit 0
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error 'Git is required for explicit upstream review.'
    exit 1
}
if (-not (Test-Path -LiteralPath (Join-Path $referencePath '.git'))) {
    Write-Error "Upstream reference repository is missing: $referencePath"
    exit 1
}

$state = Read-ReviewState
$reviewed = ([string]$state.reviewed_sha).Trim()
$env:GIT_TERMINAL_PROMPT = '0'
Write-Host "Fetching $($state.repo):$($state.branch) into the isolated review repository..." -ForegroundColor DarkGray
& git -C $referencePath fetch origin $state.branch --no-tags --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Upstream review fetch failed. No review state or live files were changed.'
    exit 1
}

$remote = (& git -C $referencePath rev-parse ("origin/" + $state.branch) 2>$null | Select-Object -First 1)
if (-not $remote -or $remote.Trim() -notmatch '^[0-9a-f]{40}$') {
    Write-Error 'Unable to resolve the fetched upstream head.'
    exit 1
}
$remote = $remote.Trim()

& git -C $referencePath cat-file -e "${reviewed}^{commit}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Reviewed commit is unavailable in the reference repository: $reviewed"
    exit 1
}

if ($remote -eq $reviewed) {
    Write-Host "Clawd upstream is already reviewed at $($remote.Substring(0,12))." -ForegroundColor Green
    Show-AuxiliaryStatus
    exit 0
}

$changed = @(& git -C $referencePath diff --name-only --no-renames "$reviewed..$remote" 2>$null)
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Unable to compute the upstream review diff.'
    exit 1
}
$changed = @($changed | Where-Object { $_ -and $_.Trim() })
$deleted = @()
$modified = @()
$untouched = @()

foreach ($path in $changed) {
    switch (Get-PathClassification $referencePath $reviewed $liveRoot $path) {
        'deleted' { $deleted += $path }
        'modified' { $modified += $path }
        'untouched' { $untouched += $path }
        default { throw "Unknown classification for $path" }
    }
}

Write-Host ""
Write-Host "Clawd upstream review: $($reviewed.Substring(0,12)) -> $($remote.Substring(0,12))" -ForegroundColor Yellow
Write-Host "Changed upstream files: $($changed.Count)"
Show-Category 'Files we deleted' $deleted
Show-Category 'Files we modified' $modified
Show-Category "Files we haven't touched" $untouched

if ($MarkReviewed) {
    Write-ReviewState $state $remote
    Write-Host ""
    Write-Host "Marked $($remote.Substring(0,12)) as reviewed." -ForegroundColor Green
    Write-Host 'No live Clawd files were copied, merged, installed, or changed.' -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host 'Review only: no live Clawd files or review state were changed.' -ForegroundColor DarkGray
    Write-Host 'After review, re-run with -MarkReviewed to advance the reviewed SHA.' -ForegroundColor DarkGray
}

Show-AuxiliaryStatus
