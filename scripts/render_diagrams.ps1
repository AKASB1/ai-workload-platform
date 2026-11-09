# Render docs/diagrams/*.d2 to docs/figures/*.svg with D2 (https://d2lang.com).
# Needs d2.exe on PATH (or $env:D2 = path to d2.exe). CI does not run this script.
# The layout engine, sketch mode, and padding are set inside each .d2 file.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$d2 = if ($env:D2) { $env:D2 } else { 'd2' }
New-Item -ItemType Directory -Force docs/figures | Out-Null
Get-ChildItem docs/diagrams -Filter *.d2 | ForEach-Object {
    $out = Join-Path 'docs/figures' ($_.BaseName + '.svg')
    & $d2 $_.FullName $out
    if ($LASTEXITCODE -ne 0) { throw "d2 failed on $($_.Name)" }
    $text = [System.IO.File]::ReadAllText((Resolve-Path $out))
    $text = [regex]::Replace($text, ' data-d2-version="[^"]*"', '')
    [System.IO.File]::WriteAllText((Resolve-Path $out), $text, (New-Object System.Text.UTF8Encoding($false)))
    Write-Output "rendered $out"
}
