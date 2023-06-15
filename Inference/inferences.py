from FeaturesPredExtraction import extract_predfeatures


base_dir = '/net/serpico-fs2/emeunier/'
models = [#'Models/SegGrOptFlow/vir-tempo/2ug45axe-patched/'
          #'Models/SegGrOptFlow/vir-tempo/2vmztvj6-patched/'
          'Models/SegGrOptFlow/vir-tempo/model230323/']
          #'Models/SegGrOptFlow/vir-tempo/entc5avp-patched/']
data_files = ['DAVIS17_D17Split', 'DAVIS_D16Split', 'FBMSclean_FBMSSplit', 'SegTrackv2_EvalSplit']

# Predictions
for model in models :
    for data_file in data_files :
        extract_predfeatures(base_dir + model, data_file, extract_features=False)
