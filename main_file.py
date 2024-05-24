import matplotlib.pyplot as plt
import os
from dataset_processing import data_generator, CHB_MIT_Hickle, Reading_FB_training, CHB_MIT_Test, Epilepsiae_iEEG, RPAH_dataset, CHB_MIT_Test_by_pat_name
import sys
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
from sklearn import metrics
import argparse
from torch.optim.lr_scheduler import StepLR,MultiStepLR
from dLIF.spike_dense import *
from dLIF.spike_neuron import *
from dLIF.spike_rnn import *
import sys
import time
from spikingjelly import visualizing


thr_func = ActFun_adp.apply

is_bias=True

parser = argparse.ArgumentParser(description='Sequential Decision Making..')
parser.add_argument('--dataset', type=str,
                    default='TUH_raw',
                    help='path to load the model')

# parser.add_argument('--load', type=str,
#                     default="",
#                     help='path to load the model')

parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')
parser.add_argument('--branches', type=int, default= '4',
                    help="number of branches")
parser.add_argument('--batch', type=int, default= '128',
                    help="number of branches")
parser.add_argument('--n', type=int, default= '100',
                    help="number of neurons")
parser.add_argument('--ICA', type=bool, default= False,
                    help="Process by ICA")
args = parser.parse_args()

ICA = args.ICA
torch.manual_seed(42)
batch_size = args.batch

# TUH TRAINING or Testing

train_loader, test_loader, seq_length, input_channels, n_classes = data_generator(args.dataset, batch_size, ICA)
output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(2) + 'L-' + str(
    args.branches) + 'BR-' + str(args.n) + 'n-' + str(ICA) + 'ICA-' + str(args.dataset) + "/"

# #For training/testing
# train_loader, test_loader, seq_length, input_channels, n_classes = Reading_FB_training(batch_size = batch_size)
# output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(2) + 'L-' + str(
#     args.branches) + 'BR-' + str(args.n) + 'n-' + str(ICA) + 'ICA-' + str(args.dataset) + "/"

# for Testing using TUH weights
#

# train_loader, test_loader, seq_length, input_channels, n_classes = CHB_MIT_Test_by_pat_name(batch_size = batch_size, pat_name='1')
# output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(2) + 'L-' + str(
#     args.branches) + 'BR-' + str(args.n) + 'n-' + str(ICA) + 'ICA-' + str(args.dataset) + "/"

#For training and validating per individual in EPILEPSIAE

# train_loader, test_loader, seq_length, input_channels, n_classes, pat_num = Epilepsiae_iEEG(batch_size = batch_size)
# output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(2) + 'L-' + str(
#     args.branches) + 'BR-' + str(args.n) + 'n-' + str(ICA) + 'ICA-' + str(pat_num) + "pat_num" + str(args.dataset) + "/"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_dim = 2

os.makedirs(output_file_fol, exist_ok=True)

#DH-SRNN model
class RNN_test(nn.Module):
    def __init__(self,args):
        super(RNN_test, self).__init__()

        n = args.n

        #DH-SRNN layer
        self.rnn_1 = spike_rnn_test_denri_wotanh_R(input_channels,n,tau_ninitializer='uniform',low_n = 0,high_n = 4,vth= 1,dt = 1,branch=args.branches,device=device,bias=is_bias)
        self.rnn_2 = spike_rnn_test_denri_wotanh_R(n,256, tau_ninitializer = 'uniform',low_n = 0,high_n = 4,vth= 1,branch=args.branches,dt = 1,device=device,bias=is_bias)
        self.dense_2 = readout_integrator_test(256,output_dim,dt = 1,device=device,bias=is_bias)

        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
      
        if is_bias:
            torch.nn.init.constant_(self.dense_2.dense.bias,0)

    def forward(self,input):
        input.to(device)
        b,seq_length,input_dim = input.shape

        self.dense_2.set_neuron_state(b)
        self.rnn_1.set_neuron_state(b)
        self.rnn_2.set_neuron_state(b)

        output = torch.zeros(b, output_dim).to(device)
        Acc_mem = []
        Acc_spikes = []
        Acc_mem_2 = []
        Acc_spikes_2 = []

        for i in range(seq_length):

            input_x = input[:,i,:].reshape(b,input_dim)
            mem_layer1,spike_layer1 = self.rnn_1.forward(input_x)
            mem_layer2,spike_layer2 = self.rnn_2.forward(spike_layer1)
            mem_layer3 = self.dense_2.forward(spike_layer2)


            if i>0:
                output += mem_layer3

        output = output/seq_length

        return output

model = RNN_test(args)
criterion = nn.CrossEntropyLoss()

print("device:",device)
model.to(device)

def test(model):

    test_acc = 0.
    sum_sample = 0.

    model.eval()
    model.rnn_1.apply_mask()
    model.rnn_2.apply_mask()


    with torch.no_grad():

        predictS= []
        true_labels = []
        predicted1 = []
        model.rnn_1.apply_mask()
        model.rnn_2.apply_mask()

        for i, (images, labels) in enumerate(test_loader):
            print (i*batch_size)
            images = images.to(device)
            labels = labels.view((-1)).long().to(device)
            labels1 = labels
            predictions = model(images)

            _, predicted = torch.max(predictions.data, 1)
            output = torch.softmax(predictions, dim=1)[:, 1]

            labels = labels.cpu()
            predicted = predicted.cpu().t()
            model.rnn_1.apply_mask()
            model.rnn_2.apply_mask()
            test_acc += (predicted == labels).sum()
            sum_sample+=predicted.numel()

            predicted1.append(predicted)
            predictS.append(output.squeeze())
            true_labels.append(labels1.squeeze())

        valid_acc = test_acc.data.cpu().numpy() / sum_sample

        output_1 = torch.cat(predictS, axis=0)
        target_1 = torch.cat(true_labels, axis=0)
        predicted_1 = torch.cat(predicted1, axis=0)

        val_auroc = calculate_auroc(target_1.cpu(), output_1.cpu())
        print ("val_auroc: ", val_auroc)

        plot_AUROC(target_1.cpu(), output_1.cpu(), val_auroc)
        plot_AUPRC (target_1.cpu(), output_1.cpu())

        val_recall = calculate_recall(target_1.cpu(), predicted_1.cpu())
        print ("val_recall", val_recall)
        val_precision = calculate_precision(target_1.cpu(), predicted_1.cpu())
        print ("val_precision", val_precision)

    return valid_acc, val_auroc, val_recall, val_precision

def test_CHB_MIT_by_pat(model):

    pat_names = [
        # '1',
        # '2',
        # '3',
        # '5','6','7',  '8','9'
        "10",
        "11",
        "14",
        "15",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
    ]

    for patname in pat_names:
        print (patname)
        train_loader, test_loader, seq_length, input_channels, n_classes =\
            CHB_MIT_Test_by_pat_name(batch_size=batch_size,pat_name=patname)
        test_acc = 0.
        sum_sample = 0.

        model.eval()
        model.rnn_1.apply_mask()
        model.rnn_2.apply_mask()

        output_file_fol = "./Results/" + str(batch_size) + 'B-' + str(2) + 'L-' + str(
            args.branches) + 'BR-' + str(args.n) + 'n-' + "-" + str(ICA) + 'ICA-' + str(args.dataset) + "/"
        os.makedirs(output_file_fol, exist_ok=True)

        with torch.no_grad():

            predictS= []
            true_labels = []
            predicted1 = []
            model.rnn_1.apply_mask()
            model.rnn_2.apply_mask()

            for i, (images, labels) in enumerate(test_loader):
                print (i*batch_size)
                images = images.to(device)
                labels = labels.view((-1)).long().to(device)
                labels1 = labels
                predictions = model(images)

                _, predicted = torch.max(predictions.data, 1)
                output = torch.softmax(predictions, dim=1)[:, 1]

                labels = labels.cpu()
                predicted = predicted.cpu().t()
                model.rnn_1.apply_mask()
                model.rnn_2.apply_mask()
                test_acc += (predicted == labels).sum()
                sum_sample+=predicted.numel()

                predicted1.append(predicted)
                predictS.append(output.squeeze())
                true_labels.append(labels1.squeeze())

            valid_acc = test_acc.data.cpu().numpy() / sum_sample

            output_1 = torch.cat(predictS, axis=0)
            target_1 = torch.cat(true_labels, axis=0)

            val_auroc = calculate_auroc(target_1.cpu(), output_1.cpu())
            print ("val_auroc: ", val_auroc)

            plot_AUROC(target_1.cpu(), output_1.cpu(), val_auroc)

            save_values (epoch = int(patname), train_loss_sum = 0, train_acc = test_acc, valid_acc=valid_acc, val_auroc=val_auroc, val_recall=val_recall, val_precision=val_precision, output_file_fol=output_file_fol)

    return valid_acc, val_auroc, val_recall, val_precision



def save_values (epoch, train_loss_sum, train_acc, valid_acc,val_auroc,val_recall, val_precision, output_file_fol):

    output_file = output_file_fol + str(batch_size) + 'B-' + str(2) + 'L-' + str(
        args.branches) + 'BR-' + str(args.n)+'n-' + str(ICA) + 'ICA-' + str(args.dataset) + '.txt'

    with open(output_file, 'a') as file:
        # Create the formatted string
        output_str = 'epoch: {:7d}, Train Loss: {:.4f}, Train Acc: {:.4f},Valid Acc: {:.4f},Val_AUROC: {:.4f}, Val_Recal: {:.4f}, Val_Precision: {:.4f}'.\
            format(epoch,train_loss_sum/len(train_loader), train_acc,valid_acc,val_auroc,val_recall, val_precision)
        # Write the string to the file
        file.write(output_str + "\n")

    plot_results(output_file,output_file_fol)

def calculate_auroc(labels, predicted_probs):
    return roc_auc_score(labels, predicted_probs)

def calculate_recall(labels, predicted_probs):
    return recall_score(labels, predicted_probs)

def calculate_precision(labels, predicted_probs):
    return precision_score(labels, predicted_probs)

def plot_results(output_file,output_file_fol):
    epochs = []
    valid_acc = []
    val_auroc = []
    val_recall = []
    val_precision = []
    train_loss = []

    # Read the log file and extract metric values

    with open(output_file, 'r') as log_file:
        for line in log_file:
            if line.startswith('epoch'):
                parts = line.strip().split(',')
                epoch = int(parts[0].split(':')[1].strip())
                loss = float(parts[1].split(':')[1].strip())
                acc = float(parts[3].split(':')[1].strip())
                auroc = float(parts[4].split(':')[1].strip())
                recall = float(parts[5].split(':')[1].strip())
                precision = float(parts[6].split(':')[1].strip())

                epochs.append(epoch)
                valid_acc.append(acc)
                val_auroc.append(auroc)
                val_recall.append(recall)
                val_precision.append(precision)
                train_loss.append(loss)

    # Create plots
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, valid_acc, label='Valid Acc')
    plt.plot(epochs, val_auroc, label='Val AUROC')
    plt.plot(epochs, val_recall, label='Val Recall')
    plt.plot(epochs, val_precision, label='Val Precision')
    plt.plot(epochs, train_loss, label="train_loss")

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Validation Metrics Over Epochs')
    plt.legend()

    outputfold = os.path.join(output_file_fol + str(batch_size) + 'B-' + str(2) + 'L-' + str(
        args.branches) + 'BR-' + str(args.n)+'n-' + str(ICA) + 'ICA-' + str(args.dataset) + '.png')

    plt.savefig(outputfold)

def plot_AUROC (target_1,output_1,auroc):

    fpr, tpr, thresholds = metrics.roc_curve(target_1, output_1)
    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    # Save the plot to a file
    output_folder = output_file_fol + '/AUROC/'
    os.makedirs(output_folder, exist_ok=True)
    outputfold = os.path.join(output_folder + str(batch_size) + 'B-' + str(2) + 'L-' + str(
        args.branches) + 'BR-' + str(args.n)+'n-'+ str(ICA) + 'ICA' + str(args.dataset) + '.pdf')

    plt.savefig(outputfold)

def plot_AUPRC (target_1,output_1):

    precision, recall, thresholds = metrics.precision_recall_curve(target_1, output_1)
    auprc = metrics.average_precision_score(target_1, output_1)
    print("AUPRC: ", auprc)

    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label='AUPRC curve (area = %0.2f)' % auprc)
    plt.plot([0, 1], [0, 0], color='navy', lw=2, linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.10])
    plt.ylim([-0.10, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    # Save the plot to a file
    output_folder = output_file_fol + '/AUPRC/'

    os.makedirs(output_folder, exist_ok=True)
    outputfold = os.path.join(output_folder + str(batch_size) + 'B-' + str(2) + 'L-' + str(
        args.branches) + 'BR-' + str(args.n)+'n-'+ str(ICA) + 'ICA' + str(args.dataset) + '.pdf')

    plt.savefig(outputfold)

def print_message(i, epoch, batch_size, train_loss, start_time):

    if int(i) > 0:
        elapsed_time = time.time() - start_time
        data_processed = i * batch_size
        total_data = len(train_loader.dataset)
        progress_percentage = 100. * data_processed / total_data
        time_per_iteration = elapsed_time / i if i > 0 else 0
        remaining_iterations = len(train_loader) - i
        remaining_time_seconds = remaining_iterations * time_per_iteration

        # Convert remaining time to hours, minutes, and seconds
        remaining_hours, remaining_seconds = divmod(remaining_time_seconds, 3600)
        remaining_minutes, remaining_seconds = divmod(remaining_seconds, 60)

        message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.5f}\tRemaining: {:1}:{:02}:{:02}s'.format(
            epoch, data_processed, total_data, progress_percentage, train_loss, int(remaining_hours),
            int(remaining_minutes), int(remaining_seconds))
        print(message, end='\r', flush=True)
        sys.stdout.flush()

def train(epochs,criterion,optimizer,scheduler,output_file_fol):

    acc_list = []
    best_acc = 0
    best_rec = 0


    for epoch in range(epochs):
        start_time = time.time()
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0 
        model.train()
        model.rnn_1.apply_mask()
        model.rnn_2.apply_mask()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.view((-1)).long().to(device)
            optimizer.zero_grad()
            predictions = model(images)

            _, predicted = torch.max(predictions.data, 1)

            train_loss = criterion(predictions,labels)
            train_loss.backward()
            train_loss_sum += train_loss.item()
            optimizer.step()

            labels = labels.cpu()
            predicted = predicted.cpu().t()

            #apply the connection pattern
            model.rnn_1.apply_mask()
            model.rnn_2.apply_mask()

            train_acc += (predicted == labels).sum()
            sum_sample+=predicted.numel()

            print_message (i,epoch,batch_size,train_loss,start_time)

        if scheduler:
            scheduler.step()

        train_acc = train_acc.data.cpu().numpy()/sum_sample
        train_loss_sum+= train_loss
        acc_list.append(train_acc)
        # print('lr: ',optimizer.param_groups[0]["lr"])

        valid_acc, val_auroc, val_recall, val_precision = test(model)

        if val_auroc>best_acc and train_acc>0.200: #change this later.
            best_acc = val_auroc
            print ("saving new model validation at:", best_acc)

            torch.save(model,output_file_fol +"AUC-"+str(best_acc)[:7]+'-'+str(batch_size)+'B-'+str(2)+'L-'+str(args.branches)+'BR-'+str(args.n)+'n-'+ str(ICA) + 'ICA-' +str(args.dataset)+'.pth')

        if val_recall>best_rec and train_acc>0.200: #change this later.
            best_rec = val_recall
            print ("saving new model validation at:", best_rec)
            torch.save(model,output_file_fol +"REC-"+str(best_rec)[:7]+'-'+str(batch_size)+'B-'+str(2)+'L-'+str(args.branches)+'BR-'+str(args.n)+'n-'+ str(ICA) + 'ICA-' +str(args.dataset)+'.pth')

        print(
            'epoch: {:7d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Acc: {:.4f}, Val_AUROC: {:.4f}, Val_Recall: {:.4f}, Val_Precision: {:.4f}'.format(
                epoch,
                train_loss_sum / len(train_loader),
                train_acc,
                valid_acc,
                val_auroc,
                val_recall,
                val_precision), flush=True)

        save_values (epoch, train_loss_sum, train_acc, valid_acc, val_auroc,val_recall, val_precision, output_file_fol)

    return acc_list

learning_rate = 1e-2 #1.2e-2

base_params = [
                    model.dense_2.dense.weight,
                    model.dense_2.dense.bias,

                    model.rnn_2.dense.weight,
                    model.rnn_2.dense.bias,

                    model.rnn_1.dense.weight,
                    model.rnn_1.dense.bias,
                ]


optimizer = torch.optim.Adam([{'params': base_params, 'lr': learning_rate},
                              {'params': model.dense_2.tau_m, 'lr': learning_rate*2},
                              {'params': model.rnn_1.tau_m, 'lr': learning_rate*2},  
                              {'params': model.rnn_1.tau_n, 'lr': learning_rate*2}, 
                              {'params': model.rnn_2.tau_m, 'lr': learning_rate},
                              {'params': model.rnn_2.tau_n, 'lr': learning_rate},
                              ], lr=learning_rate)

scheduler = StepLR(optimizer, step_size=100, gamma=.1) # 20

epochs = 200

print (model.parameters)
total_params = sum(p.numel() for p in model.parameters())
print("Total number of parameters:", total_params)

bytes_per_parameter = 4  # for 32-bit floats
total_bytes = total_params * bytes_per_parameter
total_megabytes = total_bytes / (1024 ** 2)
print("Model size: {:.2f} MB".format(total_megabytes))

if len(args.load) > 0:
    model_ckp = torch.load(args.load)

    if args.dataset== "RPAH":
        valid_acc, val_auroc, val_recall, val_precision = test_RPAH(model_ckp)
    elif args.dataset == "CHB_MIT_by_pat":
        valid_acc, val_auroc, val_recall, val_precision = test_CHB_MIT_by_pat(model_ckp)
    else:
        valid_acc, val_auroc, val_recall, val_precision = test(model_ckp)

    sys.exit()
else:
    acc_list = train(epochs,criterion,optimizer,scheduler,output_file_fol)
    test_acc = test()
    print(test_acc)


