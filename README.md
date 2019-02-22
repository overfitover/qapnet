## mini_batch
python pre_processor/gen_mini_batches.py

## add path 
export PYTHONPATH=$PYTHONPATH:'/home/ovo/project/graduation_project/qapNet/wavedata'
export PYTHONPATH=$PYTHONPATH:'/home/ovo/project/graduation_project/qapNet'

## train
python experiments/run_training.py --pipeline_config=avod/configs/pyramid_cars_example.config  --device='0' --data_split='train'


## evaluator
python experiments/run_evaluation.py --pipeline_config=avod/configs/pyramid_cars_example.config --device='0' --data_split='val'

**results** scripts/offline_eval/results/

## inference
python experiments/run_inference.py --checkpoint_name='pyramid_cars_example' --data_split='val' --ckpt_indices=120 --device='0'

**results** avod/data/outputs/proposals_and_final_predictions

## viewing results
python visual/show_predictions_2d.py
