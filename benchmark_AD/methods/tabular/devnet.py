import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
import sys
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from benchmark_AD.methods.tabular.prepare_traces_tabular import prepare_traces_tabular
from benchmark_AD.utils import log_results, predict_scores

MAX_INT = np.iinfo(np.int32).max
data_format = 0

def dev_network_linear(input_shape):
    '''
    network architecture with no hidden layer, equivalent to linear mapping from
    raw inputs to anomaly scores
    '''    
    x_input = Input(shape=input_shape)
    intermediate = Dense(1, activation='linear',  name = 'score')(x_input)
    return Model(x_input, intermediate)


def dev_network_s(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)    
    return Model(x_input, intermediate)


def dev_network_d(input_shape):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
    return Model(x_input, intermediate)


def deviation_loss(y_true, y_pred):
    '''
    z-score-based deviation loss
    '''    
    confidence_margin = 5.     
    ## size=5000 is the setting of l in algorithm 1 in the paper
    ref = tf.constant(
    np.random.normal(loc=0., scale=1.0, size=5000),
    dtype=tf.float32
)
    dev = (y_pred - tf.reduce_mean(ref)) / tf.math.reduce_std(ref)
    inlier_loss = tf.abs(dev) 
    outlier_loss = tf.abs(tf.maximum(confidence_margin - dev, 0.))
    return tf.reduce_mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


def deviation_network(input_shape, network_depth):
    '''
    construct the deviation network-based detection model
    '''
    if network_depth == 4:
        model = dev_network_d(input_shape)
    elif network_depth == 2:
        model = dev_network_s(input_shape)
    elif network_depth == 1:
        model = dev_network_linear(input_shape)
    else:
        sys.exit("The network depth is not set properly")
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=deviation_loss, optimizer=rms)
    return model


def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    '''      
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.asarray(training_labels, dtype=np.float32).reshape(-1, 1)


def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for libsvm stored sparse data.
    Alternates between positive and negative pairs.
    '''      
    ref = np.empty((batch_size))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = x_train[ref, :].toarray()
    return ref, np.asarray(training_labels, dtype=np.float32).reshape(-1, 1)


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:                
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        else:
            ref, training_labels = input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield(ref, training_labels)
        if (counter > nb_batch):
            counter = 0


def load_model_weight_predict(model, input_shape, network_depth, x_test):
    '''
    load the saved weights to make predictions
    '''
    scoring_network = Model(inputs=model.input, outputs=model.output)    
    
    if data_format == 0:
        scores = scoring_network.predict(x_test)
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores


def devnet(args, X_train, y_train, X_test, y_test, train_ids, test_ids):
    tf.keras.utils.set_random_seed(args.seed)
    epoch_num = getattr(args, "epochs", None) or 50

    # create labels: clean inlier set and a few labeled anomalies (inliers 0, outliers 1)
    outlier_indices = np.where(y_train == 1)[0]
    normal_indices = np.where(y_train == 0)[0]
    
    n_outliers_to_keep = int(len(outlier_indices) * args.perc_outliers_train)
    np.random.seed(args.seed)
    keep_outlier_indices = np.random.choice(
        outlier_indices, 
        size=n_outliers_to_keep, 
        replace=False
    )

    train_indices = np.concatenate([normal_indices, keep_outlier_indices])

    print(f"number of training outliers included: {len(keep_outlier_indices)} out of {len(outlier_indices)} total outliers in the training set")

    # Create new training set
    X_train_semi = X_train[train_indices]
    y_train_semi = y_train[train_indices]
    train_ids_semi = train_ids[train_indices]

    train_outlier_indices = np.where(y_train_semi == 1)[0]
    train_inlier_indices = np.where(y_train_semi == 0)[0]
    
    # modeling 
    epochs = getattr(args, "epochs", None) or 50
    batch_size = 512 
    nb_batch = 20
    rng = np.random.RandomState(args.seed)
    input_shape = (X_train_semi.shape[1],)
    network_depth = 2
    if not len(train_outlier_indices) or not len(train_inlier_indices):
        raise ValueError("DevNet requires normal samples and at least one retained anomaly; increase --perc_outliers_train")
    known_outliers = len(train_outlier_indices)
    cont_rate = 0.02
    
    model = deviation_network(input_shape, network_depth)

    model_name = str(args.run_dir / "devnet.keras")

    checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                           save_best_only = True, save_weights_only = False)

    model.fit(batch_generator_sup(X_train_semi, train_outlier_indices, train_inlier_indices, batch_size, nb_batch, rng),
                                          steps_per_epoch = nb_batch,
                                          epochs = epochs,
                                          callbacks=[checkpointer])
    
    devnet_scores = load_model_weight_predict(model, input_shape, network_depth, X_test)
    
    devnet_scores = devnet_scores.reshape(-1)
    # model = DevNet(epochs=epoch_num, random_seed=args.seed, device=args.device, known_outliers=len(keep_outlier_indices))

    # model.fit(X_train_semi, y=y_train_semi)

    # devnet_scores = model.decision_function(X_test)

    y_test_pred = predict_scores(args, devnet_scores, y_test)

    f1 = f1_score(y_test, y_test_pred)
    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, devnet_scores)
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)

    print(f"DeepSVDD F1 Score: {f1:.4f}")
    print(f"DeepSVDD Accuracy: {acc:.4f}")
    print(f"DeepSVDD AUC: {auc:.4f}")
    print(f"DeepSVDD Balanced Accuracy: {bal_acc:.4f}")

    log_results(args, epoch_num, epoch_num+1, f1, acc, auc, bal_acc, model=None, gt_labels=y_test, pred_labels=y_test_pred.reshape(-1), pred_scores=devnet_scores, train_ids=train_ids_semi, test_ids=test_ids)  # epoch_num for epoch


def run_devnet(args, train_df, test_df):
    print("Running DevNet")
    train_type = "semi_supervised"

    # read in data and convert raw traces to tabular format
    X_train, y_train, X_test, y_test, train_ids, test_ids = prepare_traces_tabular(args, train_type, train_df, test_df)
   
    devnet(args, X_train, y_train, X_test, y_test, train_ids, test_ids)