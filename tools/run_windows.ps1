param(
    [string]$Image = "openfusion:local",
    [string]$Context = "desktop-linux",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$OpenFusionArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

docker --context $Context run --rm --gpus all `
    -v "${repoRoot}:/workspace" `
    -w /workspace `
    $Image `
    python main.py @OpenFusionArgs
