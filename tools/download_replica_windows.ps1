$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sampleDir = Join-Path $repoRoot "sample"
$replicaDir = Join-Path $sampleDir "replica"
$zipPath = Join-Path $sampleDir "Replica.zip"
$url = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"

New-Item -ItemType Directory -Force $sampleDir | Out-Null

if (-not (Test-Path $replicaDir)) {
    if (-not (Test-Path $zipPath)) {
        Write-Host "Downloading Replica.zip..."
        curl.exe -L --fail --retry 5 --retry-delay 10 -o $zipPath $url
    }

    Write-Host "Extracting Replica.zip..."
    tar.exe -xf $zipPath -C $sampleDir
}

$actualReplicaDir = Get-ChildItem -LiteralPath $sampleDir -Directory |
    Where-Object { $_.Name -ieq "replica" } |
    Select-Object -First 1

if ($actualReplicaDir -and $actualReplicaDir.Name -cne "replica") {
    $tmpReplicaDir = Join-Path $sampleDir "replica_tmp_casefix"
    if (Test-Path $tmpReplicaDir) {
        throw "Temporary path already exists: $tmpReplicaDir"
    }

    Move-Item -LiteralPath $actualReplicaDir.FullName -Destination $tmpReplicaDir
    Move-Item -LiteralPath $tmpReplicaDir -Destination $replicaDir
}

foreach ($scene in @("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2")) {
    $sceneDir = Join-Path $replicaDir $scene
    if (-not (Test-Path $sceneDir)) {
        continue
    }

    $rgbDir = Join-Path $sceneDir "rgb"
    $depthDir = Join-Path $sceneDir "depth"
    New-Item -ItemType Directory -Force $rgbDir | Out-Null
    New-Item -ItemType Directory -Force $depthDir | Out-Null

    $resultsDir = Join-Path $sceneDir "results"
    if (Test-Path $resultsDir) {
        Write-Host "Reorganizing $scene..."

        Get-ChildItem -LiteralPath $resultsDir -Filter "depth*.png" | ForEach-Object {
            $num = [int]([regex]::Match($_.BaseName, "\d+").Value)
            Move-Item -LiteralPath $_.FullName -Destination (Join-Path $depthDir "$num.png") -Force
        }

        Get-ChildItem -LiteralPath $resultsDir -Filter "frame*.jpg" | ForEach-Object {
            $num = [int]([regex]::Match($_.BaseName, "\d+").Value)
            Move-Item -LiteralPath $_.FullName -Destination (Join-Path $rgbDir "$num.jpg") -Force
        }

        Remove-Item -LiteralPath $resultsDir -Recurse -Force
    }
}

Write-Host "Replica dataset is ready at $replicaDir"
