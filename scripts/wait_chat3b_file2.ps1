param(
    [int]$WaitPid,
    [string]$Root = (Split-Path -Parent $PSScriptRoot)
)

Set-Location $Root
while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 10
}

$url = 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/model-00002-of-00002.safetensors'
$dst = Join-Path $Root 'models\Qwen2.5-3B-Instruct\model-00002-of-00002.safetensors'
$out = Join-Path $Root 'results\chat3b_file2.out.log'
$err = Join-Path $Root 'results\chat3b_file2.err.log'
$p = Start-Process -FilePath curl.exe -ArgumentList @(
    '-L', '-C', '-', '--retry', '10', '--retry-delay', '3',
    '--retry-all-errors', '-o', $dst, $url
) -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
$p.Id | Set-Content (Join-Path $Root 'results\chat3b_file2.pid')
