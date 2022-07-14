import logging
import bottleneck as bn
import numpy as np


def load_rec(dataset):
    rec_dict = {}
    margin_score_dict = {}

    # 读取推荐结果
    with open("rec_list_%s.csv" % dataset, "r") as f:
        for line in f.readlines():
            l = line.split(",")
            if l[0] == 'user_id':
                continue

            u = int(l[0])
            margin_score = float(l[1])
            items = l[2].split("-")
            items = [int(i) for i in items]

            rec_dict[u] = items
            margin_score_dict[u] = margin_score
    
    return rec_dict, margin_score_dict


def set_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler('logging.txt', mode='w')
    fh.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)

    return logger


def get_topk_items(X_pred, k):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    return idx_topk