# build.ps1 — builds dist/Downcharter+/ (onedir) and zips it for distribution.
# Usage:  powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"

Write-Host "==> Installing build dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements-build.txt

Write-Host "==> Cleaning previous build..." -ForegroundColor Cyan
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist)  { Remove-Item -Recurse -Force dist }

Write-Host "==> PyInstaller (onedir)..." -ForegroundColor Cyan
python -m PyInstaller downcharter.spec --noconfirm

$out = "dist/Downcharter+"
if (-not (Test-Path $out)) { throw "Build failed: $out does not exist" }

Write-Host "==> Zipping for distribution..." -ForegroundColor Cyan
$zip = "dist/Downcharter+.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }
Compress-Archive -Path "$out/*" -DestinationPath $zip

Write-Host "OK -> $zip" -ForegroundColor Green
Write-Host "Executable -> $out/Downcharter+.exe" -ForegroundColor Green
