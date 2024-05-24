import torch
from torch.utils.data import DataLoader, TensorDataset
import mne
from mne.preprocessing import ICA
import numpy as np
import re
import hickle
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler

def data_generator(dataset, batch_size, ICA):

    if dataset != "":

        if dataset == "TUH_RAW":
            ###specify  file_name_x   train_path_x   train_y_path   dev_path  dev_path_x dev_path_y
            file_name_x = ""
            train_y_path =""
            dev_path =""
            dev_path_x =""
            dev_path_y =""

            #For training part.
        if ICA:
            train_X_train = ICA_Data(file_name_x)
        else:
            train_X_train = np.load(dev_path_x)
            train_X_train = train_X_train.astype(np.float16)

        train_X_train = train_X_train[0:45000]  #quick test
        train_y_train = np.load(train_y_path)[0:45000] #quick test

        train_y_train = train_y_train.astype(np.int64)

        print (train_X_train.shape,train_y_train.shape)

        print("Number of 1s:", np.count_nonzero(train_y_train == 1))
        print("Number of 0s:", np.count_nonzero(train_y_train == 0))

        if ICA:
            test_X_train = ICA_Data(dev_path)
            test_X_train = test_X_train.astype(np.float16)
        else:
            test_X_train = np.load(dev_path_x)
            test_X_train = test_X_train.astype(np.float16)

        def Noise_Min_Max(X_test):

            new_or = X_test.reshape(-1, X_test.shape[-1])
            scaler = MinMaxScaler()
            new_or = scaler.fit_transform(new_or)

            sfreq = 250  # Sample frequency in Hz, adjust as needed
            chs = ['names']

            info1 = mne.create_info(chs, sfreq, ch_types='eeg')
            raw1 = mne.io.RawArray(new_or.transpose(1, 0), info1)
            raw1.notch_filter(60, fir_design='firwin')

            filtered_data = raw1.get_data().transpose(1, 0)
            filtered_data = filtered_data.reshape(X_test.shape[0], 3000, X_test.shape[2])

            return filtered_data

        # train_X_train = Noise_Min_Max(train_X_train) Maybe it is because i already have an average on the signal (?)
        # train_X_train = train_X_train.astype(np.float16)
        # test_X_train = Noise_Min_Max(test_X_train[0:10000])
        # test_X_train = test_X_train.astype(np.float16)

        test_X_train = test_X_train[0:20000]
        test_y_train = np.load(dev_path_y)[0:20000]
        test_y_train = test_y_train.astype(np.int64)

        train_dataset = TensorDataset(torch.FloatTensor(train_X_train), torch.tensor(train_y_train))
        test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)   #I m gonna use 120.
        test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False)   # i m gonna use 120

        n_classes = 2
        seq_length = 12 * 250
        input_channels = 19

    return train_loader, test_loader, seq_length, input_channels, n_classes


def create_mne_raw(data, sfreq, chs=None):

    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1',
           'O2']

    if chs is None:
        chs_ = ['ch{}'.format(i) for i in range(data.shape[0])]
    else:
        # assert data.shape[0] == len(chs)
        chs_ = ch_names

    ch_types = ['eeg' for _ in range(len(chs_))]

    info = mne.create_info(ch_names=chs_, sfreq=sfreq, ch_types=ch_types, verbose=False)
    print (info)
    raw = mne.io.RawArray(data * 1e-7, info)
    print ("here1")

    return raw

def ica_arti_remove(data, sfreq, chs=None):

    raw = create_mne_raw(data, sfreq, chs)
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=0.1, h_freq=None, verbose=False)

    ica = ICA(n_components=19, random_state=13)
    try:
        ica.fit(filt_raw, verbose=False)
    except:
        return None

    print ("here2")

    ica.exclude = []

    eog_indices1, eog_scores1 = ica.find_bads_eog(filt_raw, threshold=2, ch_name='Fp1', verbose=False)
    print('eog_indices1', eog_indices1)
    eog_indices2, eog_scores2 = ica.find_bads_eog(filt_raw, threshold=2, ch_name='Fp2', verbose=False)
    print('eog_indices2', eog_indices2)

    if len(eog_indices1) > 0:
        ica.exclude.append(eog_indices1[0])
    if len(eog_indices2) > 0:
        ica.exclude.append(eog_indices2[0])

    print('ica.exclude', ica.exclude)

    if len(ica.exclude) > 0:
        reconst_raw = filt_raw.copy()
        reconst_raw.load_data()
        ica.apply(reconst_raw)
        print('Reconstructing data from ICA components...')
        return reconst_raw.get_data() * 1e6

    return data

def Segmentation (data_processed):

    segment_duration_samples = 12 * 250
    num_segments = int (data_processed.shape[1]//segment_duration_samples)
    segmented_data_shape = (19, 3000, num_segments)
    segmented_data = np.zeros(segmented_data_shape)

    for i in range(num_segments):
        start_idx = i * segment_duration_samples
        end_idx = (i + 1) * segment_duration_samples
        segment = data_processed[:, start_idx:end_idx]
        segmented_data[:,:,i] = segment

    segmented_data = segmented_data.transpose(2, 1, 0)

    return segmented_data

def initialize_savings (data, output_fol):

    start_idx1= 1000
    print (data.shape[0])
    n = int(data.shape[0]/start_idx1)
    for i in range(n):
        start_idx = i * start_idx1 #0,
        end_idx = (i + 1) * start_idx1 #1000
        data1 = data[start_idx:end_idx]
        data2 = data1.transpose(2, 0, 1).reshape(19, -1) #19, 3000 x length.
        data_processed = ica_arti_remove(data2, 250, chs=19)
        segmented_data = Segmentation(data_processed)
        segmented_data = segmented_data.astype(np.float16)
        np.save (output_fol + str(i) +"_subset.npy", segmented_data)


###for creating files

# #train
# data = np.load("train_x.npy")
# initialize_savings(data, output_fol = "")

#dev
# data = np.load("dev_x.npy")
# initialize_savings(data, output_fol = "")

# data = np.load("testx.npy")
# initialize_savings(data, output_fol = "/dev/")

def extract_number(filename):
    """Extracts the number from the filename."""
    match = re.search(r'(\d+)_subset', filename)
    return int(match.group(1)) if match else None

def ICA_Data(directory):

    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    files.sort(key=extract_number)

    # List to store the loaded arrays
    arrays = []

    # Loop through each sorted file and load the array
    for filename in files:
        file_path = os.path.join(directory, filename)
        print (filename)
        data = np.load(file_path)
        arrays.append(data)

    # Stack the arrays along the first dimension
    final_array = np.vstack(arrays)

    # Print the final shape
    print("Final shape:", final_array.shape)
    return final_array

def CHB_MIT_Hickle(pat_number, batch_size):

    X_f = []
    Y_f = []
    for data_name in ['interictal', 'ictal']:
        # Load data
        data = hickle.load(f"yourdata.hickle")

        X = []
        Y = []

        for i in range(len(data[0])):
            x = data[0][i]
            X.append(x)
            y = data[1][i]
            Y.append(y)

        X_test = np.concatenate(X, axis=0)
        Y_test = np.concatenate(Y, axis=0)

        # Masking to find indices where Y_label is not equal to 2
        indices_to_keep = Y_test != 2

        # Filtering X_train and Y_label based on the indices
        X_test_filtered = X_test[indices_to_keep]
        Y_test_filtered = Y_test[indices_to_keep]

        print(X_test_filtered.shape)
        print(Y_test_filtered.shape)

        X_f.append(X_test_filtered)
        Y_f.append (Y_test_filtered)

    X_test = np.concatenate(X_f, axis=0)
    Y_test = np.concatenate(Y_f, axis=0)

    ones_indices = np.where(Y_test == 1)[0]  # indices.

    final_data = X_test[ones_indices]
    final_data_Y = Y_test[ones_indices]

    duplicated_ones_indices = np.repeat(final_data_Y, 30)
    duplicated_train_X_train = np.repeat(final_data, 30, axis=0)

    X_test = np.concatenate((X_test, duplicated_train_X_train), axis=0)
    Y_test = np.concatenate((Y_test, duplicated_ones_indices), axis=0)

    test_y_train = Y_test.astype(np.int64)

    print(X_test.shape)
    print(test_y_train.shape)

    # Define the desired order of electrodes
    electrode_order = [1, 13, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 5, 6, 7, 8, 9, 10, 11, 12, 2]

    electrode_order_minus_one = [elec - 1 for elec in electrode_order]
    print(electrode_order_minus_one)

    # Reorder the electrodes axis based on the defined order
    reordered_data = X_test[:, :, electrode_order_minus_one]

    new_o = reordered_data[:,:,0:21]
    new_or = np.concatenate((new_o[:, :, :2], new_o[:, :, 4:]), axis=2)

    channel_names = [
        'names']

    new_or = new_or.reshape(-1,new_or.shape[-1])

    print (X_test.shape)
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.tensor(test_y_train))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Number of 1s:", np.count_nonzero(train_y_train == 1))
    print("Number of 0s:", np.count_nonzero(train_y_train == 0))

    train_y_train = train_y_train.astype(np.int64)
    train_dataset = TensorDataset(torch.FloatTensor(train_X_train), torch.tensor(train_y_train))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    n_classes = 2
    seq_length = 12 * 250
    input_channels = 19

    return train_loader, test_loader, seq_length, input_channels, n_classes

def CHB_MIT_Test_by_pat_name(batch_size, pat_name):

    X_f = []
    Y_f = []


    for data_name in ['interictal', 'ictal']:
        # Load data
        data = hickle.load(f"/mnt/data12_16T/CHBMIT_SZDET_ACCURATE/{data_name}_{pat_name}.hickle")

        X = []
        Y = []

        for i in range(len(data[0])):
            x = data[0][i]
            X.append(x)
            y = data[1][i]
            Y.append(y)

        X_test = np.concatenate(X, axis=0)
        Y_test = np.concatenate(Y, axis=0)

        def Power_Noise_Max_Min_scaling (X_test):

            new_or = X_test.reshape(-1, X_test.shape[-1])
            sfreq = 256  # Sample frequency in Hz, adjust as needed
            chs = ['names']
            info1 = mne.create_info(chs, sfreq, ch_types='eeg')
            raw1 = mne.io.RawArray(new_or.transpose(1, 0), info1)
            raw1 = raw1.resample(sfreq=250)
            raw1.notch_filter(60, fir_design='firwin')

            # After filtering, reshape back to the original shape
            filtered_data = raw1.get_data().transpose(1, 0)
            filtered_data = filtered_data.reshape(X_test.shape[0],3000,X_test.shape[2])
            return filtered_data

            new_o = X_test
            new_or = np.concatenate((new_o[:, :, :2], new_o[:, :, 5:]), axis=2)
            print ("new_or.shape", new_or.shape)
            return new_or

        X_test = Power_Noise_Max_Min_scaling(X_test)

        indices_to_keep = (Y_test == 1) | (Y_test == 0) | (Y_test == 2)

        # Filtering X_train and Y_label based on the indices
        X_train_filtered = X_test[indices_to_keep]  #only the ones that are either 1 or 0
        Y_train_filtered = Y_test[indices_to_keep]

        X_f.append(X_train_filtered)
        Y_f.append(Y_train_filtered)

    X_test = np.concatenate(X_f, axis=0)
    Y_test = np.concatenate(Y_f, axis=0)

    X_test = X_test.astype(np.float16)
    Y_test = Y_test.astype(np.int64)
    Y_test[Y_test==2] = 1

    print ("X_shape", X_test.shape)
    print ("1: ",np.count_nonzero(Y_test==1))
    print ("0: ",np.count_nonzero(Y_test==0))

    #Zero_train_dataset_Not_Neccesary_for_Testing
    X_train = np.zeros((1000, 3000, 19))
    Y_train = np.zeros(1000, )
    Y_train[0:500] = 1
    Y_train = Y_train.astype(np.int64)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.tensor(Y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.tensor(Y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_classes = 2
    seq_length = 12 * 250
    input_channels = 19

    return train_loader, test_loader, seq_length, input_channels, n_classes


def Reading_FB_training(batch_size):

    pat_name = [
        # '1','3','4','5',
        # '6','14',
        # '15','16','17', '18', '19',
        '20',
        # '21',
    ]

    X_f = []
    Y_f = []

    X_t_f = []
    Y_t_f = []

    for pat_n in pat_name:
        print (pat_n)
        for data_name in ['interictal', 'ictal']:

            # Load data
            data = hickle.load(f"/mnt/data12_16T/FreigburgPre/{data_name}_{pat_n}.hickle")

            X = []
            Y = []

            for i in range(len(data[0])):
                x = data[0][i]
                X.append(x)
                # print('X', len(X), X[0].shape)
                y = data[1][i]
                Y.append(y)
                # print ("Y", len(y), y[0].shape)

            X_test = np.concatenate(X, axis=0)
            Y_test = np.concatenate(Y, axis=0)

            def Power_Noise_Max_Min_scaling (X_test):
                #MAX-MIN-scaling
                new_or = X_test.reshape(-1, X_test.shape[-1])
                scaler = MinMaxScaler()
                new_or = scaler.fit_transform(new_or)
                #Power Removing
                sfreq = 256  # Sample frequency in Hz, adjust as needed
                channel_names = ["0", "1", "2", "3", "4", "5"]
                info1 = mne.create_info(channel_names, sfreq, ch_types='ecog')
                raw1 = mne.io.RawArray(new_or.transpose(1, 0), info1)
                raw1.notch_filter(50, fir_design='firwin')

                # After filtering, reshape back to the original shape
                filtered_data = raw1.get_data().transpose(1, 0)
                filtered_data = filtered_data.reshape(X_test.shape)

                return filtered_data

            X_test = Power_Noise_Max_Min_scaling(X_test)

            indices_to_keep = (Y_test != 1) | (Y_test == 0)
            indices_to_keep2 = (Y_test == 1)

            # Filtering X_train and Y_label based on the indices
            X_train_filtered = X_test[indices_to_keep]
            Y_train_filtered = Y_test[indices_to_keep]

            X_train_filtered2 = X_test[indices_to_keep2] #only seizures
            Y_train_filtered2 = Y_test[indices_to_keep2] #only seizures

            X_f.append(X_train_filtered)
            Y_f.append(Y_train_filtered)

            X_t_f.append(X_train_filtered2)
            Y_t_f.append(Y_train_filtered2)

    X_train = np.concatenate(X_f, axis=0)
    Y_train = np.concatenate(Y_f, axis=0)
    Y_train[Y_train == 2] = 1

    X_train = X_train.astype(np.float16)
    Y_train = Y_train.astype(np.int64)

    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=99)

    print (X_train.shape)
    print (Y_train.shape)

    print ("Number of occurrences of label '1':", np.count_nonzero(Y_train==1))
    print ("Number of occurrences of label '0':", np.count_nonzero(Y_train==0))
    print ("Number of occurrences of label '1':", np.count_nonzero(Y_test==1))
    print ("Number of occurrences of label '0':", np.count_nonzero(Y_test==0))

    #TESTING PART.ONLY 1 seizures.
    X_test_1 = np.concatenate(X_t_f, axis=0)
    y_test_1 = np.concatenate(Y_t_f, axis=0)
    y_test_1 = y_test_1.astype(np.int64)

    print ("Number of occurrences of label '1':", np.count_nonzero(y_test_1==1))
    print ("Number of occurrences of label '0':", np.count_nonzero(y_test_1==0))

    #keeping 0 only.
    indices_to_keep_3 = (Y_test == 0)  #only keep the 0 from Y_test.
    Y_train_filtered =  X_test[indices_to_keep_3]  #filtered
    Y_test_filt =  Y_test[indices_to_keep_3]  #filtered. Good no more here

    print("Number of occurrences of label '1':", np.count_nonzero(Y_test_filt == 1))
    print("Number of occurrences of label '0':", np.count_nonzero(Y_test_filt == 0))

    X_test = np.vstack((Y_train_filtered,X_test_1))
    y_test = np.concatenate((Y_test_filt, y_test_1),axis=0)

    print ("Number of occurrences of label '1':", np.count_nonzero(y_test==1))
    print ("Number of occurrences of label '0':", np.count_nonzero(y_test==0))

    X_test, c, y_test, c = train_test_split(X_test, y_test, test_size=0.05, random_state=42)

    print("Number of occurrences of label '1' final shape:", np.count_nonzero(y_test == 1))
    print("Number of occurrences of label '0' final shape:", np.count_nonzero(y_test == 0))

    train_dataset = TensorDataset(torch.FloatTensor(X_train[0:50000]), torch.tensor(Y_train[0:50000]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_classes = 2
    seq_length = 12 * 256
    input_channels = 6

    return train_loader, test_loader, seq_length, input_channels, n_classes

def Epilepsiae_iEEG (batch_size):

    pat_name = [
        # '1',
        # '2',
        # '3',
        # '4',
        # '5',
        # '6',
        # '7',
        '8',
        # '9',
        # '10',
        # '11',
        # '12',
        # '13',
        # '14',
        # '15',
    ]

    X_f = []
    Y_f = []

    for pat_n in pat_name:
        print(pat_n)
        for data_name in ['bckg', 'seiz']:

            # Load data
            data = hickle.load(f"/mnt/data12_16T/EpilepsiaeSurf_iEEG_Time_domain/{data_name}_{pat_n}.hickle")

            X = []
            Y = []

            for i in range(len(data[0])):
                x = data[0][i]
                X.append(x)
                y = data[1][i]
                Y.append(y)

            X_test = np.concatenate(X, axis=0)
            Y_test = np.concatenate(Y, axis=0)

            def Power_Noise_Max_Min_scaling(X_test):
                new_or = X_test.reshape(-1, X_test.shape[-1])
                # scaler = MinMaxScaler()
                # new_or = scaler.fit_transform(new_or)   #Not required.
                sfreq = 256
                channel_names = [f"ecog{i + 1}" for i in range(X_test.shape[2])]
                info1 = mne.create_info(channel_names, sfreq, ch_types='ecog')
                raw1 = mne.io.RawArray(new_or.transpose(1, 0), info1)
                raw1.notch_filter(50, fir_design='firwin')

                filtered_data = raw1.get_data().transpose(1, 0)
                filtered_data = filtered_data.reshape(X_test.shape)

                return filtered_data

            X_test = Power_Noise_Max_Min_scaling(X_test)

            X_f.append(X_test)
            Y_f.append(Y_test)

    X_train = np.concatenate(X_f, axis=0)
    Y_train = np.concatenate(Y_f, axis=0)

    X_train = X_train.astype(np.float16)
    Y_train = Y_train.astype(np.int64)

    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=99)

    print(X_train.shape)
    print(Y_train.shape)

    print("Number of occurrences of label '1':", np.count_nonzero(Y_train == 1))
    print("Number of occurrences of label '0':", np.count_nonzero(Y_train == 0))
    print("Number of occurrences of label '1':", np.count_nonzero(Y_test == 1))
    print("Number of occurrences of label '0':", np.count_nonzero(Y_test == 0))

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.tensor(Y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.tensor(Y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_classes = 2
    seq_length = 12 * 256
    input_channels = X_train.shape[2]

    return train_loader, test_loader, seq_length, input_channels, n_classes, pat_name


