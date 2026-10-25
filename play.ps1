param(
    [ValidateRange(1,100)][int]$Games = 10,
    [switch]$NoGuess,
    [ValidateSet('beginner','intermediate','expert')][string]$Difficulty = 'beginner',
    [ValidateRange(0.1,10)][double]$Delay = 0.25
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$agentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $agentPython)) {
    py -3 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment' }
    & $agentPython -m pip install -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Could not install dependencies' }
}
$agentArguments = @('-m','minesweeper_agent','play','--games',"$Games",'--stop-on-win','--difficulty',$Difficulty,
                    '--delay',$Delay.ToString([Globalization.CultureInfo]::InvariantCulture))
if (-not $NoGuess) { $agentArguments += '--allow-guess' }
Write-Host 'Starting classic Minesweeper. Keep its window visible. Hold Esc or move the mouse to the top-left corner to stop.'
& $agentPython @agentArguments
exit $LASTEXITCODE
