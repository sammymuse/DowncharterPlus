# build.ps1 — builds dist/Downcharter+/ (onedir) and zips it for distribution.
# Usage:  powershell -ExecutionPolicy Bypass -File build.ps1 [-Version v1.0.4]
# If -Version is omitted we fall back to the latest local git tag, or "dev".
param([string]$Version = "")
$ErrorActionPreference = "Stop"

if (-not $Version) {
    try { $Version = (git describe --tags --abbrev=0 2>$null).Trim() } catch {}
    if (-not $Version) { $Version = "dev" }
}
Write-Host "==> Version: $Version" -ForegroundColor Cyan

Write-Host "==> Installing build dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements-build.txt

Write-Host "==> Cleaning previous build..." -ForegroundColor Cyan
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist)  { Remove-Item -Recurse -Force dist }

Write-Host "==> PyInstaller (onedir)..." -ForegroundColor Cyan
python -m PyInstaller downcharter.spec --noconfirm

$built = "dist/Downcharter+"
if (-not (Test-Path $built)) { throw "Build failed: $built does not exist" }

# Stamp the version onto the distributed folder + zip (e.g. "Downcharter+ v1.0.4").
$name = "Downcharter+ $Version"
$out  = "dist/$name"
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
Rename-Item -Path $built -NewName $name

Write-Host "==> Zipping for distribution..." -ForegroundColor Cyan
$zip = "dist/$name.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path $out -DestinationPath $zip

Write-Host "OK -> $zip" -ForegroundColor Green
Write-Host "Executable -> $out/Downcharter+.exe" -ForegroundColor Green
