# from LMPforPSPP.getdata import *
from lmcodes.getdata import *

from keras.losses import CategoricalCrossentropy, LossFunctionWrapper
from keras.layers import *
from keras.regularizers import *
from keras.models import *
from keras.metrics import *
import tensorflow as tf
from keras import backend as k
from keras.utils import losses_utils
from keras.callbacks import ModelCheckpoint

# Custom defined accuracy
def accuracy(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    y_ = tf.argmax(y_pred, axis =- 1)
    mask = tf.greater(y, 0)
    return K.cast(K.equal(tf.boolean_mask(y, mask), tf.boolean_mask(y_, mask)), K.floatx())

def custom_loss(y_true, y_pred):
    y = tf.argmax(y_true, axis =- 1)
    mask = tf.greater(y, 0)
    return CategoricalCrossentropy(
             (tf.boolean_mask(y_true, mask)), 
             (tf.boolean_mask(y_pred, mask)))

def custom_categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    mask = tf.greater(y, 0)
    return K.sum(tf.divide(
                 K.categorical_crossentropy(
                    y_true, y_pred, from_logits=from_logits), 
                 mask.shape[0]))

class CustomCategoricalCrossentropy(LossFunctionWrapper):
    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE,
                 name='categorical_crossentropy'):
        super(CustomCategoricalCrossentropy, self).__init__(
            custom_categorical_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing)


""" Build model """
def conv_block(x, activation=True, batch_norm=True, drop_out=True, res=True):
    cnn = Conv1D(64, 11, padding="same")(x)
    if activation: cnn = TimeDistributed(Activation("relu"))(cnn)
    if batch_norm: cnn = TimeDistributed(BatchNormalization())(cnn)
    if drop_out:   cnn = TimeDistributed(Dropout(0.5))(cnn)
    if res:        cnn = Concatenate(axis=-1)([x, cnn])
    
    return cnn

def super_conv_block(x):
    c3 = Conv1D(32, 1, padding="same")(x)
    c3 = TimeDistributed(Activation("relu"))(c3)
    c3 = TimeDistributed(BatchNormalization())(c3)
    
    c7 = Conv1D(64, 3, padding="same")(x)
    c7 = TimeDistributed(Activation("relu"))(c7)
    c7 = TimeDistributed(BatchNormalization())(c7)
    
    c11 = Conv1D(128, 5, padding="same")(x)
    c11 = TimeDistributed(Activation("relu"))(c11)
    c11 = TimeDistributed(BatchNormalization())(c11)
    
    x = Concatenate(axis=-1)([x, c3, c7, c11])
    x = TimeDistributed(Dropout(0.5))(x)
    return x

def CNN_BIGRU(maxlen_seq, n_words, n_tags):
    # Inp is one-hot encoded version of inp_alt
    inp          = Input(shape=(maxlen_seq, n_words))
    inp_alt      = Input(shape=(maxlen_seq,))
    inp_profiles = Input(shape=(maxlen_seq, n_words))

    # Concatenate embedded and unembedded input
    x_emb = Embedding(input_dim=n_words, output_dim=64, 
                      input_length=maxlen_seq)(inp_alt)
    x = Concatenate(axis=-1)([inp, x_emb, inp_profiles])

    x = super_conv_block(x)
    x = conv_block(x)
    x = super_conv_block(x)
    x = conv_block(x)
    x = super_conv_block(x)
    x = conv_block(x)

    x = Bidirectional(GRU(units = 256, return_sequences = True, recurrent_regularizer=l2(0.2)))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256, activation = "relu"))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    
    y = TimeDistributed(Dense(n_tags, activation = "softmax"))(x)
    # https://stackoverflow.com/questions/48018457/removing-layers-from-a-pretrained-keras-model-gives-the-same-output-as-original
    model = Model([inp, inp_alt, inp_profiles], y)
    # for m in model.layers:
    #     print(m.name, m.output_shape)

    return model

def decode_results(y_, reverse_decoder_index):
    print("prediction: " + str(onehot_to_seq(y_, reverse_decoder_index).upper()))
    return str(onehot_to_seq(y_, reverse_decoder_index).upper())

def run_test(_model, data1, data2, data3, csv_name, npy_name, tokenizer_decoder):
    reverse_decoder_index = {value:key for key,value in tokenizer_decoder.word_index.items()}
    reverse_encoder_index = {value:key for key,value in tokenizer_encoder.word_index.items()}
    
    # Get predictions using our model
    y_test_pred = _model.predict([data1, data2, data3])

    decoded_y_pred = []
    for i in range(len(test_input_data)):
        res = decode_results(y_test_pred[i], reverse_decoder_index)
        decoded_y_pred.append(res)

    # Set Columns
    out_df = pd.DataFrame()
    out_df["id"] = test_df.id.values
    out_df["expected"] = decoded_y_pred

    # Save results
    with open(csv_name, "w") as f:
        out_df.to_csv(f, index=False)

    np.save(npy_name, y_test_pred)

""" Run below for a single run """
def model_train(x_train, y_train, maxlen_seq, n_words, n_tags, 
    x_val=None, y_val=None, n_epoch = 100, batch_size=128,
    mpath="savedModels/", lmmodel=None):
    """
    Main Training function with the following properties:
        Optimizer - Nadam
        Loss function - Categorical Crossentropy
        Batch Size - 128 (any more will exceed Collab GPU RAM)
        Epochs - 50
    """
    if lmmodel is None:
        model = CNN_BIGRU(maxlen_seq, n_words, n_tags)
    else:
        new_out = TimeDistributed(
            Dense(n_tags, activation = "softmax"))(
            lmmodel.layers[-1].output)
        model = Model(lmmodel.inputs, new_out)

    model.compile(
        optimizer="Nadam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy", accuracy])

    filepath=mpath+"weights-{epoch:03d}-{val_accuracy:.4f}.hdf5"
    checkpointer = ModelCheckpoint(
        filepath, 
        monitor='val_accuracy', 
        verbose=1, 
        save_best_only=True, 
        mode='max')
    earlstopper = EarlyStopping(monitor='val_accuracy',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)
    
    if x_val is not None and y_val is not None:
        history = model.fit( x_train, y_train,
            batch_size = batch_size, epochs = n_epoch,
            validation_data = (x_val, y_val),
            callbacks=[checkpointer, earlstopper])
    else:
        history = model.fit( x_train, y_train,
            batch_size=batch_size, epochs=n_epoch,
            callbacks=[checkpointer, earlstopper])

    return history, model