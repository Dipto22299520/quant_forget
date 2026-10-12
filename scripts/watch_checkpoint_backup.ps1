param([int]$TransferProcessId)
$ErrorActionPreference = 'Stop'
$backupRoot = Join-Path $PSScriptRoot '../backups/runpod_2026-09-12'
$statusPath = Join-Path $backupRoot 'backup_status.txt'
'Checkpoint transfer running. Keep the pod and this computer online.' | Set-Content $statusPath
Wait-Process -Id $TransferProcessId -ErrorAction SilentlyContinue
$manifest = Get-Content (Join-Path $backupRoot 'expected_files.json') -Raw | ConvertFrom-Json
$runRoot = Join-Path $backupRoot 'validation_batch16_pilot_8B_en_s1'
$problems = @()
foreach ($entry in $manifest.PSObject.Properties) {
    $target = Join-Path $runRoot $entry.Name
    if (!(Test-Path -LiteralPath $target -PathType Leaf)) {
        $problems += "Missing: $($entry.Name)"
    } elseif ((Get-Item -LiteralPath $target).Length -ne $entry.Value) {
        $problems += "Incomplete: $($entry.Name)"
    }
}
if ($problems.Count) {
    @('BACKUP INCOMPLETE. Preserve the pod.', $problems) | Set-Content $statusPath
} else {
    'BACKUP COMPLETE: every file matches the remote inventory size. Results archive SHA256 verified separately. Checkpoint content hashes have not been compared.' | Set-Content $statusPath
}
