python train_3d2d.py --dataset mmwhs --max_iteration 6000 --exp whs0 --consistency 0.1 --slice_strategy 12 --split 'train0' --quality_bar 0.98 --ht 0.9 --st 0.7
python test.py --gpu 0 --dataset mmwhs --model whs0 --min_iteration 100 --max_iteration 6000 --iteration_step 100 --split 'valid0'
