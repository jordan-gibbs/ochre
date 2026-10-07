# Ochre installer for Windows: builds from this checkout and installs `ochre`.
# Usage: ./scripts/install.ps1
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

function Have($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

if (-not (Have cargo)) {
    Write-Host 'Installing Rust (rustup)...'
    if (Have winget) {
        winget install --id Rustlang.Rustup -e --silent --accept-package-agreements --accept-source-agreements
    } else {
        Invoke-WebRequest https://win.rustup.rs/x86_64 -OutFile "$env:TEMP\rustup-init.exe"
        & "$env:TEMP\rustup-init.exe" -y --default-toolchain stable
    }
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
}

# Tauri needs the MSVC build tools; WebView2 ships with Windows 11 (and current Windows 10).
if (-not (Get-Command link.exe -ErrorAction SilentlyContinue) -and -not (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe")) {
    Write-Warning 'Visual Studio C++ build tools not found. Install "Desktop development with C++" from https://visualstudio.microsoft.com/visual-cpp-build-tools/ and re-run.'
    exit 1
}

Write-Host 'Building Ochre (release). The first build takes a few minutes...'
cargo install --locked --path app/src-tauri
# Ochre used to be called openwhisprflow: drop that old binary (settings and models carry over on first launch).
if (Have openwhisprflow) { try { cargo uninstall openwhisprflow-app *> $null } catch {} }
Write-Host ''
Write-Host 'Installed. Start it with:  ochre'
Write-Host 'Hold Right Alt to dictate. Settings are in the tray icon.'
