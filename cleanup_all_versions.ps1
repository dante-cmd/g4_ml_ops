$rootPath = "./data/ml_data"

# Check if root path exists
if (-not (Test-Path $rootPath)) {
    Write-Host "Path $rootPath does not exist." -ForegroundColor Red
    exit
}

Write-Host "Scanning for folders to remove in: $(Resolve-Path $rootPath)" -ForegroundColor Cyan

# Recursively find directories matching the pattern
Get-ChildItem -Path $rootPath -Directory -Recurse | Where-Object { 
    $_.Name -match '^v\d+$' -or $_.Name -eq 'champion' 
} | ForEach-Object {
    Write-Host "Removing: $($_.FullName)" -ForegroundColor Yellow
    Remove-Item -Path $_.FullName -Recurse -Force
}

Write-Host "Cleanup complete!" -ForegroundColor Green
