# Build Face Sequence Aligner into a standalone Windows executable.
# Run from the project root:  .\build.ps1
#
# Output: dist\FaceSequenceAligner\FaceSequenceAligner.exe
# Zip:    FaceSequenceAligner-windows.zip

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Prerequisites ─────────────────────────────────────────────────────────────
Write-Host "Checking prerequisites…" -ForegroundColor Cyan

$pyver = python --version 2>&1
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found on PATH."; exit 1 }
Write-Host "  $pyver"

python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing PyInstaller…"
    pip install pyinstaller --quiet
}

# ── Clean previous build ──────────────────────────────────────────────────────
Write-Host "Cleaning previous build…" -ForegroundColor Cyan
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

# ── Build ─────────────────────────────────────────────────────────────────────
Write-Host "Building with PyInstaller…" -ForegroundColor Cyan
pyinstaller face-sequence-aligner.spec
if ($LASTEXITCODE -ne 0) { Write-Error "PyInstaller failed."; exit 1 }

# ── Create zip ────────────────────────────────────────────────────────────────
$zip = "FaceSequenceAligner-windows.zip"
Write-Host "Creating $zip…" -ForegroundColor Cyan
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path "dist\FaceSequenceAligner" -DestinationPath $zip

$sizeMB = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host ""
Write-Host "Done!  $zip  ($sizeMB MB)" -ForegroundColor Green
Write-Host "Executable: dist\FaceSequenceAligner\FaceSequenceAligner.exe"
