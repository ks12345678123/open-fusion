$ErrorActionPreference = "Stop"

$base = Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..")).Path "sample/icl/living_room"
New-Item -ItemType Directory -Force $base | Out-Null

@(
    "481.20 0.0 319.5",
    "0.0 -480.0 239.5",
    "0.0 0.0 1.0"
) | Set-Content -Encoding ASCII -Path (Join-Path $base "intrinsics.txt")

$items = @(
    @{ Name = "kt0"; Archive = "living_room_traj0_frei_png.tar.gz"; Url = "http://www.doc.ic.ac.uk/~ahanda/living_room_traj0_frei_png.tar.gz"; Gt = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom0n.gt.sim" },
    @{ Name = "kt1"; Archive = "living_room_traj1_frei_png.tar.gz"; Url = "http://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz"; Gt = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom1n.gt.sim" },
    @{ Name = "kt2"; Archive = "living_room_traj2_frei_png.tar.gz"; Url = "http://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz"; Gt = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom2n.gt.sim" },
    @{ Name = "kt3"; Archive = "living_room_traj3_frei_png.tar.gz"; Url = "http://www.doc.ic.ac.uk/~ahanda/living_room_traj3_frei_png.tar.gz"; Gt = "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/livingRoom3n.gt.sim" }
)

foreach ($item in $items) {
    $dir = Join-Path $base $item.Name
    $rgbDir = Join-Path $dir "rgb"
    $depthDir = Join-Path $dir "depth"
    $gtPath = Join-Path $dir "livingRoom.gt.sim"

    if ((Test-Path $rgbDir) -and (Test-Path $depthDir) -and (Test-Path $gtPath)) {
        Write-Host "Skipping $($item.Name), already present."
        continue
    }

    New-Item -ItemType Directory -Force $dir | Out-Null
    $archive = Join-Path $dir $item.Archive

    Write-Host "Downloading $($item.Name) archive..."
    curl.exe -L --fail --retry 5 --retry-delay 5 -o $archive $item.Url

    Write-Host "Extracting $($item.Name)..."
    tar.exe -xzf $archive -C $dir
    Remove-Item -LiteralPath $archive -Force

    Write-Host "Downloading $($item.Name) trajectory..."
    curl.exe -L --fail --retry 5 --retry-delay 5 -o $gtPath $item.Gt
}

Write-Host "ICL dataset is ready at $base"
