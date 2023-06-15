from LinkPredictions import linkpredictions
from MaskSelection import maskselectionoptimax
from ProbaMultiMask import probatomultimask


base_dir = '/net/serpico-fs2/emeunier/'
models = [#'Models/SegGrOptFlow/vir-tempo/2ug45axe-patched/'
          #'Models/SegGrOptFlow/vir-tempo/2vmztvj6-patched/'
          'Models/SegGrOptFlow/vir-tempo/model230323/']
          #'Models/SegGrOptFlow/vir-tempo/entc5avp-patched/']
data_files = ['DAVIS17_D17Split', 'DAVIS_D16Split', 'FBMSclean_FBMSSplit', 'SegTrackv2_EvalSplit']
steps = ['val']
cuts_size = [10]

for model in models :
    for data_file in data_files :
        for step in steps :
            for cut_size in cuts_size :
                linkpredictions(model, data_file+f'_{step}', cut_size, base_dir=base_dir)
                if data_file == 'DAVIS17_D17Split' :
                    probatomultimask(model, data_file+f'_{step}', cut_size, base_dir=base_dir)
                else :
                    maskselectionoptimax(model, data_file+f'_{step}', cut_size, base_dir=base_dir)
