param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('osm', 'osm-convert', 'label', 'building-train', 'road-train', 'infer', 'test')]
    [string]$Stage,
    [switch]$NoGui
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:USERPROFILE '.conda\envs\model\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "找不到 model 环境 Python: $python"
}
$environmentRoot = Split-Path -Parent $python
$gdalData = Join-Path $environmentRoot 'Library\share\gdal'
$projData = Join-Path $environmentRoot 'Library\share\proj'
if (Test-Path -LiteralPath $gdalData) { $env:GDAL_DATA = $gdalData }
if (Test-Path -LiteralPath $projData) { $env:PROJ_DATA = $projData }

$arguments = switch ($Stage) {
    'osm' { @('makeLabel/index.py', '--download-osm', '--config', 'makeLabel/config/osm_download.yaml') }
    'osm-convert' { @('makeLabel/index.py', '--convert-osm', '--config', 'makeLabel/config/osm_download.yaml') }
    'label' { @('makeLabel/index.py', '--config', 'makeLabel/config/default.yaml') }
    'building-train' { @('trainModel/index.py', '--config', 'trainModel/config/building.yaml') }
    'road-train' { @('trainModel/index.py', '--config', 'trainModel/config/road.yaml') }
    'infer' { @('getBuildAndRoad/index.py', '--config', 'getBuildAndRoad/config/infer.yaml') }
    'test' { @('-m', 'unittest', 'discover', '-s', 'tests', '-v') }
}
if ($NoGui -and $Stage -ne 'test') {
    $arguments += '--no-gui'
}

Push-Location $projectRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
