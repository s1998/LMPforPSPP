def decode_results(y_, reverse_decoder_index):
    print("prediction: " + str(onehot_to_seq(y_, reverse_decoder_index).upper()))
    return str(onehot_to_seq(y_, reverse_decoder_index).upper())

def run_test(_model, data1, data2, data3, csv_name, npy_name):
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
def train(X_train, y_train, X_val=None, y_val=None, n_epoch = 100):
    """
    Main Training function with the following properties:
        Optimizer - Nadam
        Loss function - Categorical Crossentropy
        Batch Size - 128 (any more will exceed Collab GPU RAM)
        Epochs - 50
    """
    model = CNN_BIGRU()
    model.compile(
        optimizer="Nadam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy", accuracy])
    
    if X_val is not None and y_val is not None:
        history = model.fit( X_train, y_train,
            batch_size = 128, epochs = n_epoch,
            validation_data = (X_val, y_val))
    else:
        history = model.fit( X_train, y_train,
            batch_size = 128, epochs = n_epoch)

    return history, model

from keras.layers import *
from keras.regularizers import *
from keras.models import *
import tensorflow as tf

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

def CNN_BIGRU():
    # Inp is one-hot encoded version of inp_alt
    inp          = Input(shape=(maxlen_seq, n_words))
    inp_alt      = Input(shape=(maxlen_seq,))
    inp_profiles = Input(shape=(maxlen_seq, 22))

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

    x = Bidirectional(CuDNNGRU(units = 256, return_sequences = True, recurrent_regularizer=l2(0.2)))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(256, activation = "relu"))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    
    y = TimeDistributed(Dense(n_tags, activation = "softmax"))(x)
    
    model = Model([inp, inp_alt, inp_profiles], y)
    
    return model
