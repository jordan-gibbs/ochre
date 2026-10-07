# Copies the DLLs ochre.exe imports but a clean Windows install may lack into
# app/src-tauri/redist/, where app/src-tauri/tauri.bundle-windows.json bundles them next to the exe:
#   - the Visual C++ runtime (ONNX Runtime is linked against the dynamic CRT); app-local
#     deployment of these files is allowed by the Visual C++ redistributable terms
#   - DirectML.dll, which ort's download-binaries copies into the target directory
# Run after `cargo build --release -p ochre-app` and before
#   tauri build --bundles nsis --config tauri.bundle-windows.json   (from app/src-tauri)
# The bundle config is separate because Tauri checks resource paths at compile time.
$ErrorActionPreference = 'Stop'
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$out = Join-Path $root 'app/src-tauri/redist'
New-Item -ItemType Directory -Force $out | Out-Null

foreach ($dll in 'msvcp140.dll', 'msvcp140_1.dll', 'vcruntime140.dll', 'vcruntime140_1.dll') {
    Copy-Item (Join-Path $env:SystemRoot "System32/$dll") $out -Force
}

$target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $root 'target' }
$dml = Get-ChildItem -Path (Join-Path $target 'release'), (Join-Path $target '*/release') -Filter DirectML.dll -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $dml) { throw "DirectML.dll not found under $target; build the release binary first" }
Copy-Item $dml.FullName $out -Force
Get-ChildItem $out | ForEach-Object { "staged $($_.Name) ($($_.Length) bytes)" }
