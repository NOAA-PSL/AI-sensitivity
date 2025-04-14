myDir='./'
ic_dir='/home/sfrolov/ai-adjoint/work/2022/'
sfno_dir='/home/sfrolov/ai-adjoint/fourcastnetv2-small/'


cd $myDir

mkdir -p data/sfno/gradients
ln -s $ic_dir data/era5
ln -s $sfno_dir sfno_data







