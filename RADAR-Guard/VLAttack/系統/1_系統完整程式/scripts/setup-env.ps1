# One-time / new-shell setup: pyenv-win + Poetry PATH for this project.
$pyenvRoot = "$env:USERPROFILE\.pyenv\pyenv-win\pyenv-win"
$env:PYENV = "$env:USERPROFILE\.pyenv\pyenv-win\"
$env:PYENV_ROOT = $pyenvRoot
$env:PYENV_HOME = "$env:USERPROFILE\.pyenv\pyenv-win\"
$poetryScripts = "$env:USERPROFILE\AppData\Roaming\Python\Python310\Scripts"
$env:Path = "$poetryScripts;$pyenvRoot\bin;$pyenvRoot\shims;" + $env:Path

Write-Host "pyenv: $(pyenv --version)"
Write-Host "poetry: $(poetry --version)"
Set-Location (Join-Path $PSScriptRoot "..")
if (Test-Path ".python-version") {
    pyenv local (Get-Content ".python-version" -Raw).Trim()
    pyenv version
}
