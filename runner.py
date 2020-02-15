!wget https://www.princeton.edu/~jzthree/datasets/ICML2014/cullpdb+profile_6133_filtered.npy.gz
!wget https://www.princeton.edu/~jzthree/datasets/ICML2014/cb513+profile_split1.npy.gz

!gzip -d cullpdb+profile_6133_filtered.npy.gz
!gzip -d cb513+profile_split1.npy.gz

from createmodel import *

X_train = [train_input_data, train_input_data_alt, train_profiles_np]
y_train = train_target_data

history, model = train(X_train, y_train, n_epoch = 10)

# Save the model as a JSON format
model.save_weights("cb513_weights_1.h5")
with open("model_1.json", "w") as json_file:
    json_file.write(model.to_json())

# Save training history for parsing
with open("history_1.pkl", "wb") as hist_file:
    pickle.dump(history.history, hist_file)
