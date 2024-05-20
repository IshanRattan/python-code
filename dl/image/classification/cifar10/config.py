
########################
class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_path = './dataset/'
batch_size = 64
epochs = 10
device = 'mps'

########################
learning_rate = .01
momentum = .9
scheduler_step_size = 1000
scheduler_gamma = 0.1