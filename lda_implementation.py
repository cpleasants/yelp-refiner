"""Implement LDA in various ways"""

import lda
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve,\
                            average_precision_score, auc, accuracy_score,\
                            precision_score, recall_score
import itertools
import numpy as np
import pandas as pd


pos_neg_cols = ['pos_food', 'neg_food', 'pos_service', 'neg_service',
                'pos_staff', 'neg_staff', 'pos_wait', 'neg_wait',
                'pos_price', 'neg_price', 'pos_amb', u'neg_amb',
                'pos_clean', 'neg_clean', 'pos_serv', 'neg_serv']


def compareToManual(reviews, vectorized_revs, coded_reviews_df,
                    vectorized_coded_revs, n_topics=20, alpha=0.1, eta=0.01):
    """
    Fit an LDA on corpus of reviews, apply to corpus of labeled reviews,
    show a heatmap comparing the two, and print the best ROC AUC scores
    associated with each topic.
    """
    n = n_topics
    a = alpha
    b = eta
    print "Topics: {}, Alpha: {}, Beta: {}".format(n, a, b)

    ldalda = lda.LDA(n_topics=n, alpha=a, eta=b, refresh=1000, random_state=1)
    ldalda.fit(vectorized_revs)

    cat_array = ldalda.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(n):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]
    plt.figure(figsize=(20, 10))
    sns.heatmap(coded_reviews_df.corr().iloc[-n:, :-n], annot=True)
    plt.show()
    best_roc_scores = {}
    # using ROC-AUC score to compare the different models. ROC-AUC is
    # appropriate because it takes into consideration differenent thresholds,
    # and I may set different thresholds depending on some parameters of the
    # final model. This is also a more un-biased comparison than accuracy.
    for col in pos_neg_cols:
        try:
            best_roc = 0
            best_topic = None
            for topic in topic_array:
                roc = roc_auc_score(coded_reviews_df[col], coded_reviews_df[topic])
                if roc > best_roc:
                    best_roc = roc
                    best_topic = topic
            best_roc_scores[col] = [best_topic, best_roc]
        except Exception:
            pass

    best_2roc_scores = {}
    for col in pos_neg_cols:
        try:
            best_2roc = 0
            best_2topic = None
            for it in itertools.combinations(topic_array, 2):
                combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]]
                roc = roc_auc_score(coded_reviews_df[col], combo)
                if roc > best_2roc:
                    best_2roc = roc
                    best_2topic = it
            best_2roc_scores[col] = [best_2topic, best_2roc]
        except Exception:
            pass

    # best_3roc_scores = {}
    # for col in pos_neg_cols:
        # try:
        #     best_3roc = 0
        #     best_3topic = None
        #     for it in itertools.combinations(topic_array, 3):
        #         combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]] + \
        #                             coded_reviews_df[it[2]]
        #         roc = roc_auc_score(coded_reviews_df[col], combo)
        #         if roc > best_3roc:
        #             best_3roc = roc
        #             best_3topic = it
        #     best_3roc_scores[col] = [best_3topic, best_3roc]
        # except Exception:
        #     pass

    top_scores = []
    for col in pos_neg_cols:
        try:
            tops = [best_roc_scores[col], best_2roc_scores[col] \
                    # , best_3roc_scores[col]
                    ]
            arg = np.argmax([each[1] for each in tops])
            print col, tops[arg]
            top_scores += [(col, tops[arg])]
        except Exception:
            pass
    return top_scores


def compareToManual2(reviews, vectorized_revs, coded_reviews_df,
                     vectorized_coded_revs, n_topics=20, alpha=0.1, eta=0.01):
    """
    Fit an LDA on corpus of reviews, apply to corpus of labeled reviews,
    show a heatmap comparing the two, and print the best average precision
    scores associated with each topic.
    """
    n = n_topics
    a = alpha
    b = eta
    print "Topics: {}, Alpha: {}, Beta: {}".format(n, a, b)

    ldalda = lda.LDA(n_topics=n, alpha=a, eta=b, refresh=1000, random_state=1)
    ldalda.fit(vectorized_revs)

    cat_array = ldalda.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(n):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]
    plt.figure(figsize=(20, 10))
    sns.heatmap(coded_reviews_df.corr().iloc[-n:, :-n], annot=True)
    plt.show()
    best_prec_scores = {}
    for col in pos_neg_cols:
        try:
            best_prec = 0
            best_topic = None
            for topic in topic_array:
                prec = average_precision(coded_reviews_df[col], coded_reviews_df[topic])
                if prec > best_prec:
                    best_prec = prec
                    best_topic = topic
            best_prec_scores[col] = [best_topic, best_prec]
        except Exception:
            pass

    best_2prec_scores = {}
    for col in pos_neg_cols:
        try:
            best_2prec = 0
            best_2topic = None
            for it in itertools.combinations(topic_array, 2):
                combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]]
                prec = average_precision(coded_reviews_df[col], combo)
                if prec > best_2prec:
                    best_2prec = prec
                    best_2topic = it
            best_2prec_scores[col] = [best_2topic, best_2prec]
        except Exception:
            pass

    # best_3prec_scores = {}
    # for col in pos_neg_cols:
        # try:
        #     best_3prec = 0
        #     best_3topic = None
        #     for it in itertools.combinations(topic_array, 3):
        #         combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]] + \
        #                             coded_reviews_df[it[2]]
        #         prec = prec_auc_score(coded_reviews_df[col], combo)
        #         if prec > best_3prec:
        #             best_3prec = prec
        #             best_3topic = it
        #     best_3prec_scores[col] = [best_3topic, best_3prec]
        # except Exception:
        #     pass

    top_scores = []
    for col in pos_neg_cols:
        try:
            tops = [best_prec_scores[col], best_2prec_scores[col] \
                    # , best_3prec_scores[col]
                    ]
            arg = np.argmax([each[1] for each in tops])
            print col, tops[arg]
            top_scores += [(col, tops[arg])]
        except Exception:
            pass
    return top_scores


def compareToManualNoPrint(reviews, vectorized_revs, coded_reviews_df,
                    vectorized_coded_revs, n_topics=20, alpha=0.1, eta=0.01):
    """
    Fit an LDA on corpus of reviews, apply to corpus of labeled reviews,
    show a heatmap comparing the two, and print the best ROC AUC scores
    associated with each topic.
    """
    n = n_topics
    a = alpha
    b = eta

    ldalda = lda.LDA(n_topics=n, alpha=a, eta=b, refresh=1000, random_state=1)
    ldalda.fit(vectorized_revs)

    cat_array = ldalda.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(n):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]

    best_roc_scores = {}
    # using ROC-AUC score to compare the different models. ROC-AUC is
    # appropriate because it takes into consideration differenent thresholds,
    # and I may set different thresholds depending on some parameters of the
    # final model. This is also a more un-biased comparison than accuracy.
    for col in pos_neg_cols:
        try:
            best_roc = 0
            best_topic = None
            for topic in topic_array:
                roc = roc_auc_score(coded_reviews_df[col], coded_reviews_df[topic])
                if roc > best_roc:
                    best_roc = roc
                    best_topic = topic
            best_roc_scores[col] = [best_topic, best_roc]
        except Exception:
            pass

    best_2roc_scores = {}
    for col in pos_neg_cols:
        try:
            best_2roc = 0
            best_2topic = None
            for it in itertools.combinations(topic_array, 2):
                combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]]
                roc = roc_auc_score(coded_reviews_df[col], combo)
                if roc > best_2roc:
                    best_2roc = roc
                    best_2topic = it
            best_2roc_scores[col] = [best_2topic, best_2roc]
        except Exception:
            pass

    # best_3roc_scores = {}
    # for col in pos_neg_cols:
    #     try:
    #         best_3roc = 0
    #         best_3topic = None
    #         for it in itertools.combinations(topic_array, 3):
    #             combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]] + \
    #                                 coded_reviews_df[it[2]]
    #             roc = roc_auc_score(coded_reviews_df[col], combo)
    #             if roc > best_3roc:
    #                 best_3roc = roc
    #                 best_3topic = it
    #         best_3roc_scores[col] = [best_3topic, best_3roc]
    #     except Exception:
    #         pass

    top_scores = []
    for col in pos_neg_cols:
        try:
            tops = [best_roc_scores[col], best_2roc_scores[col] \
            # , best_3roc_scores[col]
            ]
            arg = np.argmax([each[1] for each in tops])
            print col, tops[arg]
            top_scores += [(col, tops[arg])]
        except Exception:
            pass
    print "Done with n={}, b={}, c={}".format(n, a, b)
    return top_scores


def compareToManual2(reviews, vectorized_revs, coded_reviews_df,
                     vectorized_coded_revs, n_topics=20, alpha=0.1, eta=0.01):
    """
    Fit an LDA on corpus of reviews, apply to corpus of labeled reviews,
    show a heatmap comparing the two, and print the best average precision
    scores associated with each topic.
    """
    n = n_topics
    a = alpha
    b = eta

    ldalda = lda.LDA(n_topics=n, alpha=a, eta=b, refresh=1000, random_state=1)
    ldalda.fit(vectorized_revs)

    cat_array = ldalda.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(n):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]
    best_prec_scores = {}
    for col in pos_neg_cols:
        try:
            best_prec = 0
            best_topic = None
            for topic in topic_array:
                prec = average_precision(coded_reviews_df[col], coded_reviews_df[topic])
                if prec > best_prec:
                    best_prec = prec
                    best_topic = topic
            best_prec_scores[col] = [best_topic, best_prec]
        except Exception:
            pass

    best_2prec_scores = {}
    for col in pos_neg_cols:
        try:
            best_2prec = 0
            best_2topic = None
            for it in itertools.combinations(topic_array, 2):
                combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]]
                prec = average_precision(coded_reviews_df[col], combo)
                if prec > best_2prec:
                    best_2prec = prec
                    best_2topic = it
            best_2prec_scores[col] = [best_2topic, best_2prec]
        except Exception:
            pass

    # best_3prec_scores = {}
    # for col in pos_neg_cols:
        # try:
        #     best_3prec = 0
        #     best_3topic = None
        #     for it in itertools.combinations(topic_array, 3):
        #         combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]] + \
        #                             coded_reviews_df[it[2]]
        #         prec = prec_auc_score(coded_reviews_df[col], combo)
        #         if prec > best_3prec:
        #             best_3prec = prec
        #             best_3topic = it
        #     best_3prec_scores[col] = [best_3topic, best_3prec]
        # except Exception:
        #     pass

    top_scores = []
    for col in pos_neg_cols:
        try:
            tops = [best_prec_scores[col], best_2prec_scores[col] \
                    # , best_3prec_scores[col]
                    ]
            arg = np.argmax([each[1] for each in tops])
            print col, tops[arg]
            top_scores += [(col, tops[arg])]
        except Exception:
            pass
    print "Done with n:{}, a:{}, b:{}".format(n, a, b)
    return top_scores



def bestTopics(lda_imlementation, vectorized_coded_revs, coded_reviews_df):
    """
    Return a dictionary with the best topic(s) for each manually-coded category
    in a given LDA implementation.
    """
    cat_array = lda_imlementation.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(lda_imlementation.n_topics):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]
    best_topics = {}
    for col in pos_neg_cols:
        try:
            best_roc = 0
            best_topic = None
            for it in itertools.combinations(topic_array, 1):
                combo = coded_reviews_df[it[0]]
                roc = roc_auc_score(coded_reviews_df[col], combo)
                if roc > best_roc:
                    best_roc = roc
                    best_topic = it
            for it in itertools.combinations(topic_array, 2):
                combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]]
                roc = roc_auc_score(coded_reviews_df[col], combo)
                if roc > best_roc:
                    best_roc = roc
                    best_topic = it
            # for it in itertools.combinations(topic_array, 3):
            #     combo = coded_reviews_df[it[0]] + coded_reviews_df[it[1]] + \
            #                         coded_reviews_df[it[2]]
            #     roc = roc_auc_score(coded_reviews_df[col], combo)
            #     if roc > best_roc:
            #         best_roc = roc
            #         best_topic = it
            best_topics[col] = best_topic
        except Exception:
            pass
    return best_topics


def getCurves(lda_imlementation, vectorized_coded_revs, column, best_tops,
              coded_reviews_df):
    """
    Print the ROC and Precision-Recall curves for a given manually-coded topic
    """
    cat_array = lda_imlementation.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(lda_imlementation.n_topics):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]
    # Get ROC curve
    y_score = sum([coded_reviews_df[top] for top in best_tops])
    fpr, tpr, thresholds = roc_curve(coded_reviews_df[column], y_score)

    plt.plot(fpr, tpr, label='AUC: {:.2}'.format(auc(fpr, tpr)))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve for {}'.format(column))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc='lower right')
    plt.show()

    # Compute Precision-Recall and plot curve
    precision, recall, thresh = precision_recall_curve(
                                    coded_reviews_df[column], y_score)
    average_precision = average_precision_score(coded_reviews_df[column],
                                                y_score)

    # Plot Precision-Recall curve
    plt.plot(recall, precision, label='AUC={:.2f}'.format(average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve for {}'.format(column))
    plt.legend(loc="lower left")
    plt.show()

    auc_thresh = zip(fpr, tpr, thresholds)
    auc_df = pd.DataFrame(auc_thresh, columns=['fpr', 'tpr', 'threshold'])

    prc_thresh = zip(precision, recall, thresh)
    prc_df = pd.DataFrame(prc_thresh, columns=['prec', 'recall', 'threshold'])

    return auc_df, prc_df


def bestThreshold(lda_imlementation, vectorized_coded_revs, column, best_tops,
                  coded_reviews_df):
    """
    Get the threshold associated with the lowest distance from (0,1) on the ROC
    curve on the test set of data
    """
    cat_array = lda_imlementation.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(lda_imlementation.n_topics):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]

    # Compute ROC curve point
    y_score = sum([coded_reviews_df[top] for top in best_tops])
    fpr, tpr, thresholds = roc_curve(coded_reviews_df[column], y_score)


    auc_thresh = zip(fpr, tpr, thresholds)
    auc_df = pd.DataFrame(auc_thresh, columns=['fpr', 'tpr', 'threshold'])
    auc_df['dist'] = ((0-auc_df.fpr)**2 + (1-auc_df.tpr)**2)**.5
    return auc_df.threshold.iloc[auc_df.dist.argmin()]


def bestThreshold2(lda_imlementation, vectorized_coded_revs, column, best_tops,
                   coded_reviews_df):
    """
    Get the threshold associated with the lowest distance from (1,1) on the PR
    curve on the test set of data
    """
    cat_array = lda_imlementation.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(lda_imlementation.n_topics):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]

    # Compute ROC curve point
    y_score = sum([coded_reviews_df[top] for top in best_tops])
    precision, recall, thresh = precision_recall_curve(
                                    coded_reviews_df[column], y_score)


    prc_thresh = zip(precision, recall, thresh)
    prc_df = pd.DataFrame(prc_thresh, columns=['prec', 'recall', 'threshold'])
    prc_df['dist'] = ((1-prc_df.prec)**2 + (1-prc_df.recall)**2)**.5
    return prc_df.threshold.iloc[prc_df.dist.argmin()]


def bestThreshold3(lda_imlementation, vectorized_coded_revs, column, best_tops,
                   coded_reviews_df):
    """
    Get the threshold associated with the highest precision score with a recall
    over 0.5 on the test set.
    """
    cat_array = lda_imlementation.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(lda_imlementation.n_topics):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]

    # Compute ROC curve point
    y_score = sum([coded_reviews_df[top] for top in best_tops])
    precision, recall, thresh = precision_recall_curve(
                                    coded_reviews_df[column], y_score)


    prc_thresh = zip(precision, recall, thresh)
    prc_df = pd.DataFrame(prc_thresh, columns=['prec', 'recall', 'threshold'])
    prc_df2 = prc_df[prc_df.recall > .5]
    return prc_df2.threshold.iloc[prc_df2.prec.argmax()]


def restaurantSummary(lda_imlementation, vectorized_coded_revs, column,
                      best_tops, coded_reviews_df, threshold):
    """
    Return the accuracy, precision, and recall scores for the LDA best topics
    at a specified threshold.
    """
    cat_array = lda_imlementation.transform(vectorized_coded_revs)

    topic_array = []
    for i in range(lda_imlementation.n_topics):
        coded_reviews_df['topic_'+str(i)] = cat_array[:, i]
        topic_array += ['topic_'+str(i)]

    # Summarize manual codes.
    total = float(coded_reviews_df[column].sum())/coded_reviews_df.shape[0]

    print "ACTUAL: {}: {}".format(column, total)

    # Summarize predictions
    coded_reviews_df['combo'] = sum([coded_reviews_df[top] for top in best_tops])

    def fun(x):
        if x > threshold:
            return 1.
        else:
            return 0.

    coded_reviews_df['combo_thresh'] = coded_reviews_df['combo'].apply(fun)
    PREDtotal = coded_reviews_df['combo_thresh'].sum()/coded_reviews_df.shape[0]

    print "PREDICTED: {}: {}".format(column, PREDtotal)

    # Calculate accuracy, precision, and recall
    act = coded_reviews_df[column]
    pred = coded_reviews_df['combo_thresh']

    acc = accuracy_score(act, pred)
    prec = precision_score(act, pred)
    rec = recall_score(act, pred)

    print 'Accuracy: {}, Precision: {}, Recall: {}'.format(acc, prec, rec)

    return total, PREDtotal, acc, prec, rec
