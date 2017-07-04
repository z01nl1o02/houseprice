@echo off

setlocal enabledelayedexpansion
set clf=ridge
set outroot=outI
for /L %%n in (3,1,8) do (
set r=0.%%n
python run_stageI.py . !r! -split !r!
mkdir !outroot!\!r!
copy !r!\!clf!.* !outroot!\!r!
)


python merge_resultI.py !outroot! inII
python run_stageII.py inII result


endlocal

@echo on
