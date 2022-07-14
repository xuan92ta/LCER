import numpy as np
from scipy import sparse
from tqdm import tqdm
import tensorflow as tf

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from parser import parse_args
from data_loaders import data_loaders
from utils import load_rec, set_logger, get_topk_items
from multvae import MultiVAE  # model of Mult-VAE


class LCER(object):
    def __init__(self, n_items, vae, lr=1e-3, seed=2022):
        self.n_items = n_items

        self.seed = seed
        self.lr = lr

        self.vae = vae
        self.saver_vae = tf.train.Saver()
        
        with tf.name_scope('ce') as scope:
            # the first column indicates the probability of being counterfactual explanations
            self.delta = tf.Variable(tf.random_uniform(shape=[n_items, 2], seed = self.seed), name='delta')

        self.margin_score = tf.placeholder(dtype=tf.float32, shape=None)
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.n_items])
        self.index_i = tf.placeholder(dtype=tf.int32, shape=None)
        self.lam = tf.placeholder_with_default(1.0, shape=None)
        self.alpha = tf.placeholder_with_default(0.0, shape=None)
    
    def build_graph(self, hard=True):
        # sampling counterfactual set
        softmax_delta = self.gumbel_softmax(self.delta, self.seed)
        softmax_temp = softmax_delta
        if hard:
            hard_delta = tf.cast(tf.one_hot(tf.argmax(softmax_delta, -1), 2), softmax_delta.dtype, name='hard_delta')
            softmax_delta = tf.stop_gradient(hard_delta - softmax_temp) + softmax_delta

        input_perturbed = tf.multiply(softmax_delta[:, 1], self.input, name='input_perturbed')  # feedback after removing counterfactual set
        input_mask = tf.multiply(softmax_delta[:, 0], self.input)  # counterfactual set
        _, perturbed_score, _ = self.vae.forward_pass(input_perturbed)  # counterfactual preference
        
        loss = self.compute_loss(input_mask, perturbed_score[0, self.index_i])

        loss_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ce')
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss, var_list=loss_vars)

        return train_op, self.delta, input_mask, loss, perturbed_score[0, self.index_i], self.saver_vae
        # 第一个返回反事实解释集合
    
    def gumbel_softmax(self, logits, seed, temperature=1., eps=1e-10):
        uniform_samples = tf.random_uniform(shape=tf.shape(logits), seed=seed)
        gumbel_samples = -tf.log(eps - tf.log(uniform_samples + eps))

        gumbel_logits = logits + gumbel_samples
        softmax_logits = tf.nn.softmax(gumbel_logits / temperature)  # 默认维度为-1

        return softmax_logits
    
    def compute_loss(self, delta, perturbed_score):
        length = tf.reduce_sum(delta) * self.lam
        l_inf = tf.nn.relu(self.alpha + perturbed_score - self.margin_score)

        loss = l_inf + length
        return loss


if __name__ == '__main__':
    args = parse_args()

    GPU_ID = args.gpu_id
    DATASET = args.dataset
    PROCESSED_DIR = args.processed_dir
    CHECKPOINT_DIR = args.checkpoint_dir

    LAM = args.lam
    ALPHA = args.alpha
    K = args.k
    SEED = args.seed
    EPOCHS = args.epochs
    LR = args.lr

    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)

    # load dataset
    train_data, val_data_tr, val_data_te, test_data_tr, test_data_te, n_items = data_loaders(PROCESSED_DIR, DATASET)
    # load recommendation results
    rec_dict, margin_score_dict = load_rec(DATASET)

    logger = set_logger()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # begin training
    tf.reset_default_graph()

    vae = MultiVAE([200, 600, n_items], lam=0.0)
    arch_str = "I-%s-I" % ('-'.join([str(d) for d in vae.dims[1:-1]]))
    chkpt_dir = CHECKPOINT_DIR + DATASET + '/VAE_anneal{}K_cap{:1.1E}/{}'.format(
        200000/1000, 0.2, arch_str)

    ce = LCER(n_items, vae, lr=LR, seed=SEED)
    train_op, delta_org, delta, loss_all, predict_perturbed, saver_vae = ce.build_graph()

    # store information of counterfactual explanations
    exp_dict = {}  # counterfactual explanations
    exp_num = []  # the number of items in counterfactual explanations
    n_nonexp = 0  # the number of recommendations for which there is no counterfactual explanation

    
    with tf.Session(config=config) as sess:
        # initialize parameters
        init = tf.global_variables_initializer()
        sess.run(init)
        # load Mult-VAE
        saver_vae.restore(sess, '{}/model'.format(chkpt_dir))

        for u, items in tqdm(rec_dict.items()):
            margin_score = margin_score_dict[u]

            input = test_data_tr[u]
            if sparse.isspmatrix(input):
                input = input.toarray()
            input = input.astype('float32')

            if input.sum() == 0:
                continue
            
            for i in items:
                logger.info("----------------------------------------------------------------------------------------")
                logger.info("Finding counterfactual explanation for user %d and item %d." % (u, i))

                lowest_loss = 1e10
                best_predict, best_delta = None, None

                # initialize importance for current user and recommended item
                sess.run(delta_org.initializer)  # 初始化delta变量

                for e in range(EPOCHS):
                    
                    feed_dict = {ce.margin_score: margin_score, ce.input: input, ce.index_i: i, ce.lam: LAM, ce.alpha: ALPHA}
                    _, l_all, pre_perturbed, d = sess.run([train_op, loss_all, predict_perturbed, delta], feed_dict=feed_dict)

                    if l_all < lowest_loss:
                        lowest_loss = l_all
                        best_predict = pre_perturbed
                        best_delta = d

                
                if best_predict <= margin_score:
                    ce_items = np.argwhere(best_delta.squeeze() == 1)
                    exp_num.append(ce_items.shape[0])
                    exp_dict[(u, i)] = ce_items.squeeze()
                    logger.info("Congratulation! the size of counterfactual explanation set is %d" % (ce_items.shape[0]))

                else:
                    n_nonexp += 1
                    logger.info("Unfortunately, there is no counterfactual explanation for user %d and item %d" % (u, i))
    
    # evaluation
    pn_list, ps_list = [], []

    tf.reset_default_graph()
    vae = MultiVAE([200, 600, n_items], lam=0.0)
    saver, logits_var, _, _, _ = vae.build_graph()

    with tf.Session(config=config) as sess:
        saver.restore(sess, '{}/model'.format(chkpt_dir))

        pn_count = 0
        ps_count = 0

        for u_i, exp in exp_dict.items():
            user = u_i[0]
            target_item = u_i[1]

            # probability of necessity
            input = test_data_tr[user]
            if sparse.isspmatrix(input):
                input = input.toarray()
            input = input.astype('float32')
            input[0, exp] = 0

            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: input})
            pred_val[input.nonzero()] = -np.inf
            
            idx_topk = get_topk_items(pred_val, K)

            if target_item not in idx_topk.squeeze():
                pn_count += 1
                pn_list.append(1)
            else:
                pn_list.append(0)

            # probability of sufficiency
            input = np.zeros([1, n_items])
            input[0, exp] = 1

            pred_val = sess.run(logits_var, feed_dict={vae.input_ph: input})
            pred_val[input.nonzero()] = -np.inf
            
            idx_topk = get_topk_items(pred_val, K)

            if target_item in idx_topk.squeeze():
                ps_count += 1
                ps_list.append(1)
            else:
                ps_list.append(0)
        
        if len(exp_dict) != 0:
            pn = pn_count / len(exp_dict)
            ps = ps_count / len(exp_dict)
            f_ns = (2 * pn * ps) / (pn + ps)
        else:
            pn = 0
            ps = 0
            f_ns = 0

    logger.info("There are %d user-item pairs whose counterfactual explanations are not found." % n_nonexp)
    logger.info("Probability of Necessity/Sufficiency is %f/%f, and their harmonic mean is %f" % (pn, ps, f_ns))
    logger.info("The average size of counterfactual explanation set is %f." % np.mean(exp_num))
