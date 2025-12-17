# 重命名 app.js 文件并清理旧文件
# 用法: .\rename_app_js.ps1 <源文件路径>

param(
    [Parameter(Mandatory=$true)]
    [string]$SourceFile
)

if (-not (Test-Path $SourceFile)) {
    Write-Error "文件不存在: $SourceFile"
    exit 1
}

# 读取文件内容
$content = Get-Content $SourceFile -Raw -Encoding UTF8

# 计算 MD5 hash
$md5 = [System.Security.Cryptography.MD5]::Create()
$hashBytes = $md5.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($content))
$hashString = ($hashBytes | ForEach-Object { $_.ToString("x2") }) -join ''
$hash = $hashString.Substring(0, 8)

# 生成新文件名
$sourceDir = Split-Path $SourceFile -Parent
$newFileName = "app.$hash.js"
$newFilePath = Join-Path $sourceDir $newFileName

# 复制文件到新名称
Copy-Item -Path $SourceFile -Destination $newFilePath -Force
Write-Output "文件已复制到: $newFilePath"

# 删除旧文件（如果存在且不是同一个文件）
if ((Split-Path $SourceFile -Leaf) -ne $newFileName) {
    Remove-Item -Path $SourceFile -Force -ErrorAction SilentlyContinue
    Write-Output "旧文件已删除: $SourceFile"
}

# 删除所有其他 app.*.js 文件（除了新文件）
Get-ChildItem -Path $sourceDir -Filter "app.*.js" | Where-Object {
    $_.Name -ne $newFileName
} | ForEach-Object {
    Remove-Item -Path $_.FullName -Force -ErrorAction SilentlyContinue
    Write-Output "已删除旧文件: $($_.Name)"
}

Write-Output "Hash: $hash"
Write-Output "新文件: $newFileName"

