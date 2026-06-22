param(
    [string]$Image = "openfusion:local",
    [string]$Context = "desktop-linux"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

docker --context $Context build `
    -t $Image `
    -f (Join-Path $repoRoot "docker/Dockerfile.local") `
    $repoRoot
