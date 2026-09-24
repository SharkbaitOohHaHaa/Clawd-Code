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
$jobTypeSource = @'
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;

public sealed class ClawdUpdateKillJob : IDisposable
{
    private IntPtr handle;

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimitInfo
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimitInfo
    {
        public BasicLimitInfo BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll")]
    private static extern bool SetInformationJobObject(
        IntPtr job,
        int infoClass,
        ref ExtendedLimitInfo info,
        uint infoLength
    );

    [DllImport("kernel32.dll")]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);

    public ClawdUpdateKillJob()
    {
        handle = CreateJobObject(IntPtr.Zero, null);
        if (handle == IntPtr.Zero) throw new Win32Exception();

        var info = new ExtendedLimitInfo();
        info.BasicLimitInformation.LimitFlags = 0x00002000;
        if (!SetInformationJobObject(
            handle,
            9,
            ref info,
            (uint)Marshal.SizeOf(info)
        ))
        {
            var error = new Win32Exception();
            CloseHandle(handle);
            handle = IntPtr.Zero;
            throw error;
        }
    }

    public void Assign(Process process)
    {
        if (!AssignProcessToJobObject(handle, process.Handle))
            throw new Win32Exception();
    }

    public void Dispose()
    {
        if (handle != IntPtr.Zero)
        {
            CloseHandle(handle);
            handle = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }
}
'@
Add-Type -TypeDefinition $jobTypeSource

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
    $job = $null
    try {
        $job = New-Object ClawdUpdateKillJob
        if (-not $process.Start()) { return $null }
        $job.Assign($process)

        if (-not $process.WaitForExit($timeoutSeconds * 1000)) {
            $job.Dispose()
            $job = $null
            try { $null = $process.WaitForExit(1000) } catch {}
            return $null
        }
        if ($process.ExitCode -ne 0) { return $null }
        $line = $process.StandardOutput.ReadLine()
        if (-not $line) { return $null }

        $sha = (($line -split '\s+')[0]).Trim()
        if ($sha -notmatch '^[0-9a-f]{40}$') { return $null }
        return $sha
    } catch {
        return $null
    } finally {
        if ($job) { $job.Dispose() }
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

    $probeInfo = New-Object System.Diagnostics.ProcessStartInfo
    $probeInfo.FileName = Join-Path $env:SystemRoot 'System32\cmd.exe'
    $probeInfo.Arguments = '/d /c ping 127.0.0.1 -t'
    $probeInfo.UseShellExecute = $false
    $probeInfo.CreateNoWindow = $true
    $probeInfo.RedirectStandardOutput = $true
    $probeInfo.RedirectStandardError = $true
    $probe = New-Object System.Diagnostics.Process
    $probe.StartInfo = $probeInfo
    $probeJob = $null
    try {
        $probeJob = New-Object ClawdUpdateKillJob
        if (-not $probe.Start()) {
            $failures += 'process-tree test could not start parent process'
        } else {
            $probeJob.Assign($probe)
            Start-Sleep -Milliseconds 300
            $children = @(Get-CimInstance Win32_Process -Filter ("ParentProcessId = " + $probe.Id) -ErrorAction Stop)
            if ($children.Count -eq 0) {
                $failures += 'process-tree test did not observe a child process'
            }
            $childIds = @($children.ProcessId)
            $probeJob.Dispose()
            $probeJob = $null
            Start-Sleep -Milliseconds 300

            if (Get-Process -Id $probe.Id -ErrorAction SilentlyContinue) {
                $failures += "parent process survived job cleanup: $($probe.Id)"
            }
            foreach ($childId in $childIds) {
                if (Get-Process -Id $childId -ErrorAction SilentlyContinue) {
                    $failures += "child process survived job cleanup: $childId"
                }
            }
        }
    } catch {
        $failures += "process-tree test failed: $($_.Exception.Message)"
    } finally {
        if ($probeJob) { $probeJob.Dispose() }
        $probe.Dispose()
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
