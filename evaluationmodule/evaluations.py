import itertools
from tqdm import tqdm
from Evaluation import main

eval_session_name='Evaluation240323'
scoring_method = 'db_eval_iou'
binary_method = 'binarise_argmax'
data_base_dir='/net/serpico-fs2/emeunier/'
pred_base_dir= f'{data_base_dir}/Models/SegGrOptFlow/vir-tempo/'

data_files = ['DAVIS_D16Split', 'SegTrackv2_EvalSplit', 'FBMSclean_FBMSSplitEval']
steps = ['val']
models = ['model230323']
suffixes = ['_seq_cs10binary_eval']

for combi in itertools.product(*[models, data_files, steps, suffixes]) :
    model_id, df, step, pred_suffix = combi
    data_file = f'{df}_{step}'
    print(f'Model : {model_id} data_file :{data_file} suffix : {pred_suffix}')
    main(scoring_method, binary_method, model_id, data_file, pred_suffix,
     eval_session_name, data_base_dir, pred_base_dir)
