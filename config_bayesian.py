############### Configuration file for Bayesian ###############
n_epochs = 10000
lr_start = 0.0001
num_workers = 16
valid_size = 0.1
batch_size = 1024
train_ens = 1
valid_ens = 1

record_mean_var = True
recording_freq_per_epoch = 8
record_layers = ['fc5']

# Cross-module global variables
mean_var_dir = None
record_now = False
curr_epoch_no = None
curr_batch_no = None
