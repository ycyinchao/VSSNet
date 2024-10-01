import os
import cv2
from tqdm import tqdm
# pip install pysodmetrics
from metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

preds_data_root = "../res/VSSNet_384"
# preds_data_root = "/media/store/yc/projects/COD/PFNet-test/results/PFNet/CDS2K"

f = open(preds_data_root+"/log.txt","a+",encoding="UTF-8")
for _data_name in ['CHAMELEON','CAMO', 'COD10K','NC4K' ]:
# for _data_name in ['CDS2K' ]:

    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    mae = MAE()

    # masks_data_root = "../../Dataset/TestDataset/" +_data_name  +"/Positive"       # 1
    masks_data_root = "../../Dataset/TestDataset/" +_data_name

    mask_root = os.path.join(masks_data_root, "GT")
    pred_root = os.path.join(preds_data_root,_data_name)           # 2
    mask_name_list = sorted(os.listdir(mask_root))
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        mae.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = mae.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
    }

    print(results)
    f.write(_data_name+"\n")
    f.write(str(results) + "\n\n")
    f.flush()

f.close()
