param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$prefix = $root + '\'
$plan = [Collections.Generic.List[object]]::new()
function Add-Weight($file, $reason, $retained = '') {
    $resolved = (Resolve-Path -LiteralPath $file.FullName).Path
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Path outside workspace' }
    if ($resolved -notmatch '\\ckpt\\model_\d+\\[^\\]+\.safetensors$') { throw "Unexpected weight path: $resolved" }
    $plan.Add([pscustomobject]@{Path=$resolved;Bytes=$file.Length;Reason=$reason;Retained=$retained})
}
# Completed exploratory pilots and smoke tests; preserve every non-weight artifact.
Get-ChildItem -LiteralPath (Join-Path $root 'runs') -Directory | Where-Object { $_.Name -match '^(pilot_|smoke_)' } | ForEach-Object {
    if (Test-Path -LiteralPath (Join-Path $_.FullName 'summary.json')) {
        Get-ChildItem -LiteralPath $_.FullName -Recurse -File -Filter '*.safetensors' | ForEach-Object { Add-Weight $_ 'Completed exploratory pilot or smoke test' }
    }
}
# Only remove backup weights whose exact contents match the primary copy.
$backupRoot = Join-Path $root 'runs_backup_stageA'
Get-ChildItem -LiteralPath $backupRoot -Recurse -File -Filter '*.safetensors' | ForEach-Object {
    $relative = $_.FullName.Substring($backupRoot.Length+1)
    $primary = Join-Path (Join-Path $root 'runs') $relative
    if ((Test-Path -LiteralPath $primary) -and ((Get-Item -LiteralPath $primary).Length -eq $_.Length)) {
        Write-Host "Checking duplicate: $relative"
        if ((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash -eq (Get-FileHash -LiteralPath $primary -Algorithm SHA256).Hash) {
            Add-Weight $_ 'SHA256-verified duplicate backup' $primary
        }
    }
}
# Old English pilot model_4 has only shards 3 and 4; model_5 is retained intact.
$oldRun = Join-Path $root 'backups\runpod_2026-09-12\validation_batch16_pilot_8B_en_s1'
if ((Test-Path -LiteralPath (Join-Path $oldRun 'summary.json')) -and
    (@(Get-ChildItem -LiteralPath (Join-Path $oldRun 'ckpt\model_5') -Filter 'model-*-of-00004.safetensors').Count -eq 4)) {
    Get-ChildItem -LiteralPath (Join-Path $oldRun 'ckpt\model_4') -File -Filter '*.safetensors' | ForEach-Object { Add-Weight $_ 'Incomplete superseded pilot checkpoint; complete model_5 retained' }
}
$auditDir = Join-Path $root 'logs\checkpoint_cleanup'
New-Item -ItemType Directory -Force -Path $auditDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$manifest = Join-Path $auditDir ($stamp + '_manifest.json')
$plan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifest -Encoding UTF8
$total = ($plan | Measure-Object Bytes -Sum).Sum
Write-Host ("Planned: {0} files, {1:N2} GiB" -f $plan.Count,($total/1GB))
if (-not $Apply) { return }
# Snapshot non-weight artifacts for a post-delete integrity check.
$protected = @(Get-ChildItem -LiteralPath $root -Recurse -File -Force | Where-Object { $_.FullName -notin $plan.Path -and $_.Name -ne '_pruned.json' -and -not $_.FullName.StartsWith($auditDir) } | Select-Object FullName,Length)
foreach ($entry in $plan) {
    $resolved = (Resolve-Path -LiteralPath $entry.Path).Path
    if (-not $resolved.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'Deletion outside workspace' }
    if ((Get-Item -LiteralPath $resolved).Length -ne $entry.Bytes) { throw 'File changed during cleanup' }
    Remove-Item -LiteralPath $resolved -Force
    Add-Content -LiteralPath (Join-Path $auditDir ($stamp + '_deleted.txt')) -Value $resolved
}
foreach ($group in ($plan | Group-Object { Split-Path -Parent $_.Path })) {
    @{pruned=$true;removed=@($group.Group | ForEach-Object { Split-Path -Leaf $_.Path });reason='Authorized storage cleanup';manifest=$manifest} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $group.Name '_pruned.json') -Encoding UTF8
}
foreach ($item in $protected) {
    if (-not (Test-Path -LiteralPath $item.FullName)) { throw "Missing protected file: $($item.FullName)" }
    if ((Get-Item -LiteralPath $item.FullName).Length -ne $item.Length) { throw "Changed protected file: $($item.FullName)" }
}
$remaining = (Get-ChildItem -LiteralPath $root -Recurse -File -Force | Measure-Object Length -Sum).Sum
@{deleted_files=$plan.Count;freed_bytes=$total;remaining_bytes=$remaining;protected_files_checked=$protected.Count;manifest=$manifest} | ConvertTo-Json | Tee-Object -FilePath (Join-Path $auditDir ($stamp + '_result.json'))
