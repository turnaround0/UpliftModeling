import numpy as np


# Criterion for splitting tree
def eucli_dist(small_df,
               pr_y_t1,
               pr_y_t0,
               pr_l,
               pr_r,
               pr_y_t1_L,
               pr_y_t0_L,
               pr_y_t1_R,
               pr_y_t0_R,
               pr_t1,
               pr_t0,
               pr_t1_L,
               pr_t0_L):
    def ed(A, B):
        return ((A - B) ** 2).sum(axis=len(A.shape) - 1)

    # Euclidean gain
    ed_node = ed(pr_y_t1, pr_y_t0)
    ed_l = ed(pr_y_t1_L, pr_y_t0_L)
    ed_r = ed(pr_y_t1_R, pr_y_t0_R)
    ed_lr = pr_l * ed_l + pr_r * ed_r
    ed_gain = ed_lr - ed_node

    # Euclidean Normalization factor
    gini_t = 2 * pr_t1 * (1 - pr_t1)
    ed_t = (pr_t1_L - pr_t0_L) ** 2 + ((1 - pr_t1_L) - (1 - pr_t0_L)) ** 2
    gini_t1 = 2 * pr_t1_L * (1 - pr_t1_L)
    gini_t0 = 2 * pr_t0_L * (1 - pr_t0_L)
    ed_norm = gini_t * ed_t + gini_t1 * pr_t1 + gini_t0 * pr_t0 + 0.5

    # Output
    info_gain_t = ed_gain / ed_norm

    return info_gain_t


def kl_divergence(small_df,
                  pr_y_t1,
                  pr_y_t0,
                  pr_l,
                  pr_r,
                  pr_y_t1_L,
                  pr_y_t0_L,
                  pr_y_t1_R,
                  pr_y_t0_R,
                  pr_t1,
                  pr_t0,
                  pr_t1_L,
                  pr_t0_L):
    def kl(A, B):
        return (A * np.log2(A / B)).sum(axis=len(A.shape) - 1)

    # KL Gain
    kl_node = kl(pr_y_t1, pr_y_t0)
    kl_l = kl(pr_y_t1_L, pr_y_t0_L)
    kl_r = kl(pr_y_t1_R, pr_y_t0_R)
    kl_lr = pr_l * kl_l + pr_r * kl_r
    kl_gain = kl_lr - kl_node

    # KL Normalization factor
    ent_t = -(pr_t1 * np.log2(pr_t1) + pr_t0 * np.log2(pr_t0))
    kl_t = pr_t1 * np.log2(pr_t1 / pr_t0_L) + \
           (1 - pr_t1_L) * np.log2((1 - pr_t1_L) / (1 - pr_t0_L))
    ent_t1 = -(pr_t1_L * np.log2(pr_t1_L) + (1 - pr_t1_L) * np.log2((1 - pr_t1_L)))
    ent_t0 = -(pr_t0_L * np.log2(pr_t0_L) + (1 - pr_t0_L) * np.log2((1 - pr_t0_L)))

    norm = kl_t * ent_t + ent_t1 * pr_t1 + ent_t0 * pr_t0 + 0.5

    # Output
    info_gain_t = kl_gain / norm

    return info_gain_t


def chisq(small_df,
          pr_y_t1,
          pr_y_t0,
          pr_l,
          pr_r,
          pr_y_t1_L,
          pr_y_t0_L,
          pr_y_t1_R,
          pr_y_t0_R,
          pr_t1,
          pr_t0,
          pr_t1_L,
          pr_t0_L):
    def chi(A, B):
        return (((A - B) ** 2) / A).sum(axis=len(A.shape) - 1)

    # Chi-squared gain
    chisq_node = chi(pr_y_t1, pr_y_t0)
    chisq_l = chi(pr_y_t1_L, pr_y_t0_L)
    chisq_r = chi(pr_y_t1_R, pr_y_t0_R)
    chisq_lr = pr_l * chisq_l + pr_r * chisq_r
    chisq_gain = chisq_lr - chisq_node

    # Chi-squared Normalization factor
    gini_t = 2 * pr_t1 * (1 - pr_t1)
    chisq_t = ((pr_t1_L - pr_t0_L) ** 2) / pr_t0_L + \
              (((1 - pr_t1_L) - (1 - pr_t0_L)) ** 2) / (1 - pr_t0_L)
    gini_t1 = 2 * pr_t1_L * (1 - pr_t1_L)
    gini_t0 = 2 * pr_t0_L * (1 - pr_t0_L)
    chisq_norm = gini_t * chisq_t + gini_t1 * pr_t1 + gini_t0 * pr_t0 + 0.5

    # Output
    info_gain_t = chisq_gain / chisq_norm

    return info_gain_t
