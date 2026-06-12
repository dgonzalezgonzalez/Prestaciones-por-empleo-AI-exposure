$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Node = "C:\Users\ngonzalezp\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
$Rscript = "C:\Users\ngonzalezp\AppData\Local\Programs\R\R-4.5.2\bin\Rscript.exe"

& $Python (Join-Path $ProjectRoot "code\run_analysis.py")
& $Python (Join-Path $ProjectRoot "code\run_next_steps.py")
& $Python (Join-Path $ProjectRoot "code\run_research_moves.py")
& $Python (Join-Path $ProjectRoot "code\run_synthetic_methods.py")
& $Python (Join-Path $ProjectRoot "code\run_continuous_event_studies.py")
& $Python (Join-Path $ProjectRoot "code\run_binary_event_study_samples.py")
& $Python (Join-Path $ProjectRoot "code\run_exposure_variation.py")
& $Python (Join-Path $ProjectRoot "code\run_memo_v1_assets.py")
& $Python (Join-Path $ProjectRoot "code\run_memo_v2_assets.py")
& $Python (Join-Path $ProjectRoot "code\run_memo_v2_event_assets.py")
& $Rscript (Join-Path $ProjectRoot "code\run_contdid_analysis.R")
& $Python (Join-Path $ProjectRoot "code\replication\referee2_replicate_core.py")
& $Node (Join-Path $ProjectRoot "code\replication\referee2_replicate_node_counts.js")
