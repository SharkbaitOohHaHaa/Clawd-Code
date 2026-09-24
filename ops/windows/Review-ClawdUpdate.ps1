param(
    [string]$MarkReviewed,
    [switch]$SelfTest
)

$ErrorActionPreference = 'Continue'
$utf8 = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
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

function Resolve-LivePath([string]$root, [string]$repoPath) {
    if (-not $repoPath -or [IO.Path]::IsPathRooted($repoPath)) {
        throw "Unsafe repository path: $repoPath"
    }
    $relative = $repoPath -replace '/', '\'
    $rootFull = [IO.Path]::GetFullPath($root).TrimEnd('\') + '\'
    $candidate = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $candidate.StartsWith($rootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Repository path escapes live root: $repoPath"
    }
    return $candidate
}

function Get-PathClassification(
    [string]$repoPath,
    [string]$reviewedSha,
    [string]$root,
    [string]$baselinePath,
    [string]$collisionPath = ''
) {
    $spec = "${reviewedSha}:$baselinePath"
    & git -C $repoPath cat-file -e $spec 2>$null
    $baselineExists = ($LASTEXITCODE -eq 0)
    $localPath = Resolve-LivePath $root $baselinePath
    $localExists = Test-Path -LiteralPath $localPath -PathType Leaf

    if ($collisionPath) {
        $collisionLocal = Resolve-LivePath $root $collisionPath
        if (Test-Path -LiteralPath $collisionLocal -PathType Leaf) {
            return 'modified'
        }
    }
    if ($baselineExists -and -not $localExists) {
        return 'deleted'
    }
    if (-not $baselineExists) {
        if ($localExists) { return 'modified' }
        return 'untouched'
    }

    $baselineHash = (& git -C $repoPath rev-parse $spec 2>$null | Select-Object -First 1)
    if (-not $baselineHash) {
        throw "Unable to hash reviewed upstream file: $baselinePath"
    }
    $localHash = (& git -C $repoPath hash-object "--path=$baselinePath" -- $localPath 2>$null | Select-Object -First 1)
    if (-not $localHash) {
        throw "Unable to hash local file: $baselinePath"
    }

    if ($baselineHash.Trim() -eq $localHash.Trim()) {
        return 'untouched'
    }
    return 'modified'
}

function Get-UpstreamChanges([string]$repoPath, [string]$fromSha, [string]$toSha) {
    $lines = @(& git -C $repoPath -c core.quotepath=false diff --name-status -M "$fromSha..$toSha" 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to compute the upstream review diff.'
    }

    $changes = @()
    foreach ($line in $lines) {
        if (-not $line) { continue }
        $parts = @($line -split "`t")
        $status = [string]$parts[0]
        if ($status -match '^R\d+$') {
            if ($parts.Count -ne 3) { throw "Unable to parse upstream rename: $line" }
            $changes += [pscustomobject]@{
                Status=$status
                BaselinePath=[string]$parts[1]
                CurrentPath=[string]$parts[2]
                Display="$($parts[1]) -> $($parts[2])"
            }
        } else {
            if ($parts.Count -ne 2) { throw "Unable to parse upstream change: $line" }
            $changes += [pscustomobject]@{
                Status=$status
                BaselinePath=[string]$parts[1]
                CurrentPath=[string]$parts[1]
                Display=[string]$parts[1]
            }
        }
    }
    return @($changes)
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

        foreach ($name in @('deleted.txt', 'modified.txt', 'untouched.txt', 'café.txt', '模型.md', 'rename_src.py', 'upstream_delete.py')) {
            Set-Content -LiteralPath (Join-Path $repo $name) -Value 'base' -Encoding UTF8
        }
        & git -C $repo add .
        & git -C $repo commit -q -m baseline
        $sha = (& git -C $repo rev-parse HEAD).Trim()

        Set-Content -LiteralPath (Join-Path $live 'modified.txt') -Value 'local-change' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $live 'untouched.txt') -Value 'base' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $live 'café.txt') -Value 'local-change' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $live 'rename_src.py') -Value 'local-change' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $live 'upstream_delete.py') -Value 'local-change' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $live 'local-collision.txt') -Value 'ours' -Encoding UTF8

        Set-Content -LiteralPath (Join-Path $repo 'modified.txt') -Value 'upstream-change' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $repo 'café.txt') -Value 'upstream-change' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $repo '模型.md') -Value 'upstream-change' -Encoding UTF8
        & git -C $repo mv rename_src.py rename_dst.py
        Remove-Item -LiteralPath (Join-Path $repo 'upstream_delete.py')
        Set-Content -LiteralPath (Join-Path $repo 'new-upstream.txt') -Value 'new' -Encoding UTF8
        Set-Content -LiteralPath (Join-Path $repo 'local-collision.txt') -Value 'upstream-new' -Encoding UTF8
        & git -C $repo add -A
        & git -C $repo commit -q -m update
        $target = (& git -C $repo rev-parse HEAD).Trim()

        $failures = @()
        $directCases = [ordered]@{
            'deleted.txt' = 'deleted'
            'modified.txt' = 'modified'
            'untouched.txt' = 'untouched'
            'new-upstream.txt' = 'untouched'
            'local-collision.txt' = 'modified'
            'café.txt' = 'modified'
            '模型.md' = 'deleted'
        }
        foreach ($entry in $directCases.GetEnumerator()) {
            $actual = Get-PathClassification $repo $sha $live $entry.Key
            if ($actual -ne $entry.Value) {
                $failures += "$($entry.Key): expected=$($entry.Value) actual=$actual"
            }
        }

        $changes = @(Get-UpstreamChanges $repo $sha $target)
        $rename = @($changes | Where-Object { $_.Status -match '^R\d+$' -and $_.BaselinePath -eq 'rename_src.py' })
        if ($rename.Count -ne 1 -or $rename[0].CurrentPath -ne 'rename_dst.py') {
            $failures += 'rename status/path parsing failed'
        } else {
            $renameClass = Get-PathClassification $repo $sha $live $rename[0].BaselinePath $rename[0].CurrentPath
            if ($renameClass -ne 'modified') { $failures += "rename classification expected=modified actual=$renameClass" }
        }

        $deletedUpstream = @($changes | Where-Object { $_.Status -eq 'D' -and $_.CurrentPath -eq 'upstream_delete.py' })
        if ($deletedUpstream.Count -ne 1) { $failures += 'upstream deletion status was not preserved' }
        if (-not ($changes | Where-Object { $_.CurrentPath -eq 'café.txt' })) { $failures += 'accented filename was not preserved' }
        if (-not ($changes | Where-Object { $_.CurrentPath -eq '模型.md' })) { $failures += 'Chinese filename was not preserved' }

        try {
            $null = Resolve-LivePath $live '../escape.txt'
            $failures += 'path traversal was not rejected'
        } catch {}

        if ($failures.Count -gt 0) {
            throw ($failures -join '; ')
        }
        Write-Host "Review classifier self-test passed: $($directCases.Count) direct cases + Unicode + rename + deletion + traversal."
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
$reviewed = ([string]$state.reviewed_sha).Trim().ToLowerInvariant()
$env:GIT_TERMINAL_PROMPT = '0'

if ($MarkReviewed -and $MarkReviewed -notmatch '^[0-9a-fA-F]{40}$') {
    Write-Error '-MarkReviewed requires the exact 40-character SHA that was reviewed.'
    exit 1
}

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
$remote = $remote.Trim().ToLowerInvariant()
$target = if ($MarkReviewed) { $MarkReviewed.Trim().ToLowerInvariant() } else { $remote }

foreach ($sha in @($reviewed, $target)) {
    & git -C $referencePath cat-file -e "${sha}^{commit}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Required commit is unavailable in the reference repository: $sha"
        exit 1
    }
}

& git -C $referencePath merge-base --is-ancestor $reviewed $target 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error 'The requested reviewed SHA is not descended from the current reviewed SHA.'
    exit 1
}
& git -C $referencePath merge-base --is-ancestor $target $remote 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error 'The requested reviewed SHA is not on the current upstream branch history.'
    exit 1
}

if ($target -eq $reviewed) {
    Write-Host "Clawd upstream target is already reviewed at $($target.Substring(0,12))." -ForegroundColor Green
    if ($remote -ne $target) {
        Write-Host "A newer upstream head remains available at $($remote.Substring(0,12))." -ForegroundColor Yellow
    }
    Show-AuxiliaryStatus
    exit 0
}

$changes = @(Get-UpstreamChanges $referencePath $reviewed $target)
$deleted = @()
$modified = @()
$untouched = @()

foreach ($change in $changes) {
    $collision = if ($change.Status -match '^R\d+$') { $change.CurrentPath } else { '' }
    $classification = Get-PathClassification $referencePath $reviewed $liveRoot $change.BaselinePath $collision
    $display = "[$($change.Status)] $($change.Display)"
    switch ($classification) {
        'deleted' { $deleted += $display }
        'modified' { $modified += $display }
        'untouched' { $untouched += $display }
        default { throw "Unknown classification for $($change.Display)" }
    }
}

Write-Host ""
Write-Host "Clawd upstream review: $($reviewed.Substring(0,12)) -> $($target.Substring(0,12))" -ForegroundColor Yellow
Write-Host "Changed upstream files: $($changes.Count)"
if ($remote -ne $target) {
    Write-Host "Current upstream head is newer: $($remote.Substring(0,12))" -ForegroundColor Yellow
}
Show-Category 'Files we deleted' $deleted
Show-Category 'Files we modified' $modified
Show-Category "Files we haven't touched" $untouched

if ($MarkReviewed) {
    Write-ReviewState $state $target
    Write-Host ""
    Write-Host "Marked exactly $($target.Substring(0,12)) as reviewed." -ForegroundColor Green
    if ($remote -ne $target) {
        Write-Host "Newer upstream commits remain unreviewed at $($remote.Substring(0,12))." -ForegroundColor Yellow
    }
    Write-Host 'No live Clawd files were copied, merged, installed, or changed.' -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host 'Review only: no live Clawd files or review state were changed.' -ForegroundColor DarkGray
    Write-Host "After review, re-run with -MarkReviewed $target to mark exactly this SHA." -ForegroundColor DarkGray
}

Show-AuxiliaryStatus
