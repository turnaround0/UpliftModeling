import numpy as np


# Criterion for splitting tree
def eucli_dist(small_df,
               pr_y1_ct1,
               pr_y1_ct0,
               pr_l,
               pr_r,
               pr_y1_l_ct1,
               pr_y1_l_ct0,
               pr_y1_r_ct1,
               pr_y1_r_ct0,
               pr_ct1,
               pr_ct0,
               pr_l_ct1,
               pr_l_ct0):
    # Euclidean gain
    ed_node = (pr_y1_ct1 - pr_y1_ct0) ** 2 + ((1 - pr_y1_ct1) - (1 - pr_y1_ct0)) ** 2
    ed_l = (pr_y1_l_ct1 - pr_y1_l_ct0) ** 2 + ((1 - pr_y1_l_ct1) - (1 - pr_y1_l_ct0)) ** 2
    ed_r = (pr_y1_r_ct1 - pr_y1_r_ct0) ** 2 + ((1 - pr_y1_r_ct1) - (1 - pr_y1_r_ct0)) ** 2
    ed_lr = pr_l * ed_l + pr_r * ed_r
    ed_gain = ed_lr - ed_node

    # Euclidean Normalization factor
    gini_ct = 2 * pr_ct1 * (1 - pr_ct1)
    ed_ct = (pr_l_ct1 - pr_l_ct0) ** 2 + ((1 - pr_l_ct1) - (1 - pr_l_ct0)) ** 2
    gini_ct1 = 2 * pr_l_ct1 * (1 - pr_l_ct1)
    gini_ct0 = 2 * pr_l_ct0 * (1 - pr_l_ct0)
    ed_norm = gini_ct * ed_ct + gini_ct1 * pr_ct1 + gini_ct0 * pr_ct0 + 0.5

    # Output
    info_gain_t = ed_gain / ed_norm

    return info_gain_t


def kl_divergence(small_df,
                  pr_y1_ct1,
                  pr_y1_ct0,
                  pr_l,
                  pr_r,
                  pr_y1_l_ct1,
                  pr_y1_l_ct0,
                  pr_y1_r_ct1,
                  pr_y1_r_ct0,
                  pr_ct1,
                  pr_ct0,
                  pr_l_ct1,
                  pr_l_ct0):
    # KL Gain
    kl_node = pr_y1_ct1 * np.log2(pr_y1_ct1 / pr_y1_ct0) + \
              (1 - pr_y1_ct1) * np.log2((1 - pr_y1_ct1) / (1 - pr_y1_ct0))
    kl_l = pr_y1_l_ct1 * np.log2(pr_y1_l_ct1 / pr_y1_l_ct0) + \
           (1 - pr_y1_l_ct1) * np.log2((1 - pr_y1_l_ct1) / (1 - pr_y1_l_ct0))
    kl_r = pr_y1_r_ct1 * np.log2(pr_y1_r_ct1 / pr_y1_r_ct0) + \
           (1 - pr_y1_r_ct1) * np.log2((1 - pr_y1_r_ct1) / (1 - pr_y1_r_ct0))
    kl_lr = pr_l * kl_l + pr_r * kl_r
    kl_gain = kl_lr - kl_node

    # KL Normalization factor
    ent_ct = -(pr_ct1 * np.log2(pr_ct1) + pr_ct0 * np.log2(pr_ct0))
    kl_ct = pr_l_ct1 * np.log2(pr_l_ct1 / pr_l_ct0) + \
            (1 - pr_l_ct1) * np.log2((1 - pr_l_ct1) / (1 - pr_l_ct0))
    ent_ct1 = -(pr_l_ct1 * np.log2(pr_l_ct1) + (1 - pr_l_ct1) * np.log2((1 - pr_l_ct1)))
    ent_ct0 = -(pr_l_ct0 * np.log2(pr_l_ct0) + (1 - pr_l_ct0) * np.log2((1 - pr_l_ct0)))

    norm = kl_ct * ent_ct + ent_ct1 * pr_ct1 + ent_ct0 * pr_ct0 + 0.5

    # Output
    info_gain_t = kl_gain / norm

    return info_gain_t


def chisq(small_df,
          pr_y1_ct1,
          pr_y1_ct0,
          pr_l,
          pr_r,
          pr_y1_l_ct1,
          pr_y1_l_ct0,
          pr_y1_r_ct1,
          pr_y1_r_ct0,
          pr_ct1,
          pr_ct0,
          pr_l_ct1,
          pr_l_ct0):
    # Chi-squared gain
    chisq_node = ((pr_y1_ct1 - pr_y1_ct0) ** 2) / pr_y1_ct0 + \
                 (((1 - pr_y1_ct1) - (1 - pr_y1_ct0)) ** 2) / (1 - pr_y1_ct0)
    chisq_l = ((pr_y1_l_ct1 - pr_y1_l_ct0) ** 2) / pr_y1_l_ct0 + \
              (((1 - pr_y1_l_ct1) - (1 - pr_y1_l_ct0)) ** 2) / (1 - pr_y1_l_ct0)
    chisq_r = ((pr_y1_r_ct1 - pr_y1_r_ct0) ** 2) / pr_y1_r_ct0 + \
              (((1 - pr_y1_r_ct1) - (1 - pr_y1_r_ct0)) ** 2) / (1 - pr_y1_r_ct0)
    chisq_lr = pr_l * chisq_l + pr_r * chisq_r
    chisq_gain = chisq_lr - chisq_node

    # Chi-squared Normalization factor
    gini_ct = 2 * pr_ct1 * (1 - pr_ct1)
    chisq_ct = ((pr_l_ct1 - pr_l_ct0) ** 2) / pr_l_ct0 + \
               (((1 - pr_l_ct1) - (1 - pr_l_ct0)) ** 2) / (1 - pr_l_ct0)
    gini_ct1 = 2 * pr_l_ct1 * (1 - pr_l_ct1)
    gini_ct0 = 2 * pr_l_ct0 * (1 - pr_l_ct0)
    chisq_norm = gini_ct * chisq_ct + gini_ct1 * pr_ct1 + gini_ct0 * pr_ct0 + 0.5

    # Output
    info_gain_t = chisq_gain / chisq_norm

    return info_gain_t


def interaction_split(small_df,
                      pr_y1_ct1,
                      pr_y1_ct0,
                      pr_l,
                      pr_r,
                      pr_y1_l_ct1,
                      pr_y1_l_ct0,
                      pr_y1_r_ct1,
                      pr_y1_r_ct0,
                      pr_ct1,
                      pr_ct0,
                      pr_l_ct1,
                      pr_l_ct0,
                      cs_ct1,
                      cs_ct0,
                      ncs_ct1,
                      ncs_ct0):
    # Compute elements for split formula
    C44 = 1 / cs_ct1 + 1 / cs_ct0 + 1 / ncs_ct1 + 1 / ncs_ct0

    UR = pr_y1_r_ct1 - pr_y1_r_ct0
    UL = pr_y1_l_ct1 - pr_y1_l_ct0

    SSE = cs_ct1 * pr_y1_l_ct1 * (1 - pr_y1_l_ct1) + \
          ncs_ct1 * pr_y1_r_ct1 * (1 - pr_y1_r_ct1) + \
          cs_ct0 * pr_y1_l_ct0 * (1 - pr_y1_l_ct0) + \
          ncs_ct0 * pr_y1_r_ct0 * (1 - pr_y1_r_ct0)

    n_node = len(small_df)

    # Output: Interaction split
    info_gain_t = ((n_node - 4) * (UR - UL) ** 2) / (C44 * SSE)

    return info_gain_t
