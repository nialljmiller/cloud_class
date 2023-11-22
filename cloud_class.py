import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as torch
import numpy as np
from torch.utils import data
from pathlib import Path
from astropy.io import fits
import os
from torch.optim import Adam
#import torchvision.transforms as T

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib import gridspec
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cycle(dl):
    while True:
        for data in dl:
            yield data


#  _   _                      _   _   _      _                      _    
# | \ | |                    | | | \ | |    | |                    | |   
# |  \| | ___ _   _ _ __ __ _| | |  \| | ___| |___      _____  _ __| | __
# | . ` |/ _ \ | | | '__/ _` | | | . ` |/ _ \ __\ \ /\ / / _ \| '__| |/ /
# | |\  |  __/ |_| | | | (_| | | | |\  |  __/ |_ \ V  V / (_) | |  |   < 
# |_| \_|\___|\__,_|_|  \__,_|_| |_| \_|\___|\__| \_/\_/ \___/|_|  |_|\_\
 

class Net(nn.Module):   
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 2, padding = 1),				#convolve the image 
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 2, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 2, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 2, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 2, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 2, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(952576,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,1),
            #nn.ReLU()
        )
    
    def forward(self, xb):
        return self.network(xb)

#  _____        _          _                     _           
# |  __ \      | |        | |                   | |          
# | |  | | __ _| |_ __ _  | |     ___   __ _  __| | ___ _ __ 
# | |  | |/ _` | __/ _` | | |    / _ \ / _` |/ _` |/ _ \ '__|
# | |__| | (_| | || (_| | | |___| (_) | (_| | (_| |  __/ |   
# |_____/ \__,_|\__\__,_| |______\___/ \__,_|\__,_|\___|_|   

class Clouds(data.Dataset):
    def __init__(self, clear_paths, notclear_paths, fits_paths, minmaxnorms=(0, 1)):
        super().__init__()        #ignore this, its magic python stuff
        self.clear_paths = list(Path(f'{clear_paths}').glob(f'**/*.JPG')) #find all the fits files in a path
        self.notclear_paths = list(Path(f'{notclear_paths}').glob(f'**/*.JPG')) #find all the fits files in a path
        self.fits_paths =  fits_paths #list(Path(f'{fits_paths}').glob(f'**/*.FIT')) #find all the fits files in a path
        self.min_ = minmaxnorms[0]
        self.max_ = minmaxnorms[1]
        self.toggle = 0
    def __len__(self):
        return len(self.fits_paths)

    def __getitem__(self, dummy_index):
        try_flag = False
        while try_flag == False:
            self.toggle = np.random.randint(2) # generate a random 0 or 1
            if self.toggle == 1:
                self.paths = self.notclear_paths
            else:
                self.paths = self.clear_paths

            index = np.random.randint(len(self.paths))    #create 
            path  = self.paths[index]
            path = str(path).split('/')[-1]
            path = path.split('.')[0]
            path = self.fits_paths + str(path) + '.FIT'
            try:
                with fits.open(path) as hdul:
                    x = hdul[0].data         
                try_flag = True
            except:
                try_flag = False

        y = self.toggle#float(path.split('_')[0])
        ri0 = 0#np.random.randint(480)
        ri1 = 0#np.random.randint(480)
        x = x[np.newaxis, ri0:ri0+480, ri1:ri1+480].astype(np.float32)


        #if np.random.rand() > 0.5:        #get more data for free! clouds are not going to be directinally dependent
        #    x = np.flip(x, axis=0)        #so were going to flip the images randomly
        #    #print('a')
        #if np.random.rand() > 0.5:        #we can do a lot more pertubations such as rotation 
        #    x = np.flip(x, axis=1)
        #    #print('a')

        def norm(ar, min_, max_):
            ar = np.clip(ar, min_, max_)
            return 2*((ar - min_)/(max_ - min_)) - 1

        #x = norm(x, self.min_, self.max_)    #need to find range before we do this. shoudl be 0-256 or something
        mean = 9388.427     #calculated for 342 samples 
        std = 5615.842
        #transform = T.Compose([T.ToTensor(), T.Normalize(mean =mean, std = std)])
        #y = transform(np.array(y))#.clone().detach()
        y = y - mean
        y = y / std
        y = torch.tensor([y]).clone().detach()

        #y = F.normalize(y, mean = mean, std = std)
        x = torch.tensor(x.astype(np.float)).clone().detach()
        #print(x,y)
        return x, y   #pytorch does not care what happens above this line as long as this line returns y and x as tensors





#  _______        _       
# |__   __|      (_)      
#    | |_ __ __ _ _ _ __  
#    | | '__/ _` | | '_ \ 
#    | | | | (_| | | | | |
#    |_|_|  \__,_|_|_| |_|

class Trainer(object):
    def __init__(
        self,
        batch_size = 128,    #how many images do we show it at a time. Ideally this number is as big as possible but we cant fit infinte images on GPU ram so trial and error increase this number until it crashes
        lr = 2e-5,          #what rate do we learn. how much do we update the network per image
        num_steps = 100000, #how many steps to we take before we finish training. You cant predict this so just set it to be a big number.
        num_workers = 128,  #leave this, its just how many parallel workers do we have crunching the numbers
        save_every = 100,   #how often do we back up the network
        sample_every = 10,  #how often do we update our loss plot and see how its behaving, dont do this frequently as its computationally expensive
        logdir = '/beegfs/car/njm/cloud_class/logs',  #where do we put our plots and stuff
        clear_paths = '/beegfs/car/njm/cloud_class/data/clear_images_sorted/',    #where is the location of the jpegs with no clouds
        notclear_paths = '/beegfs/car/njm/cloud_class/data/cloudy_images_sorted/', #'                             ' with clouds
        fits_paths = '/beegfs/car/njm/cloud_class/data/fits/' #where are the fits files that correspond to that location
    ):

        super().__init__()

        #everything below is saving all of these values to the class. this means we dont need to pass those values around to access them we can just do "self.value" to get it. 

        self.clear_paths = clear_paths
        self.notclear_paths = notclear_paths
        self.fits_paths = fits_paths

        self.batch_size = batch_size
        self.lr = lr
        self.num_steps = num_steps
        self.num_workers = num_workers
        self.save_every = save_every
        self.sample_every = sample_every
        self.logdir = logdir

        self.net = Net().to(device=DEVICE, dtype=torch.float) #make the network and make it acessable in this training class

        #criterion = nn.MSELoss()   #There are multiple different types of 'loss' here. theyre very simple to understand, a quick 5 minute google and you will get it
        #criterion = nn.L1Loss()
        self.criterion = nn.HuberLoss()    #how do we teach it, training tries to make this number small. some measure of "prediction - real = loss" 
        self.opt = Adam(self.net.parameters(), lr=self.lr)    #some fancy stuff to make it train optimally. this controlls step size "how much do we change the network for each calculation of loss"

        self.ds = Clouds(self.clear_paths, self.notclear_paths, self.fits_paths)    #set up the data loader. this is a function that gives us data when called, there is a lot to play with here, feel free
        self.dl = cycle(data.DataLoader(self.ds, batch_size = 256, shuffle=True, num_workers=512))    #turns that function into a magic data loader. dont worry too much about it but batch size is important

    def train(self):
        step = 0
        while step < self.num_steps:    #see here we defined self.steps above so we can call it as just self.steps anywhere in this class
            data = next(self.dl)        #grab some data from the data loader
            inputs, labels = data #sperate into thing and name for thing
            inputs = inputs.to(device=DEVICE, dtype=torch.float)    #push the data to the gpu "DEVICE" is defined at the top of the code
            labels = labels.to(device=DEVICE, dtype=torch.float)    #
            
            outputs = self.net(inputs)    #give the network the data and ask what it thinks the name of the data is
            loss = self.criterion(outputs, labels)    #inform the network on how wrong it was
            loss.backward()    #let the network update itself based on how wrong it was
            self.opt.step()    #end of cycle, next!
            self.save_params(loss, step) #write step and loss to a file. this doesnt need to be done every step, we could implement somethng similar to the "sample_every" check
            if step % self.sample_every == 0 and step > 1:   #if the current step is divisible by the how often we sample, sample
                self.loss_plot()    #make a plot of the loss 
            step = step + 1 #and keep looping

    def save_params(self, loss, step):
        with open(str(self.logdir +'/'+ 'loss.txt'), 'a') as df:
            df.write(f'{step},{loss.item()}\n')
        print(loss.item(),'--',step)

    def loss_plot(self): 
        #===============
        #Loss plot
        #===============
        plt_epoch, plt_loss = np.genfromtxt(str(self.logdir +'/'+ 'loss.txt'), delimiter = ',').T #load the stuff from the file. 
        sort_idx = np.argsort(plt_epoch)
        median_window_size = 100
        #med_loss = self.RunningMedian(plt_loss[sort_idx],median_window_size)       #it would be nice to have a function which creates a runnning median of the loss
        #med_epoch = self.RunningMedian(plt_epoch[sort_idx],median_window_size)     #i tried to do it with numpy but it was crap
        #print(med_loss)
        
        plt.clf()
        fig, ax = plt.subplots()
        ax.axhline(y = 0, color = 'r', linestyle = '-')
        ax.plot(plt_epoch[sort_idx], plt_loss[sort_idx], 'k', alpha = 0.9, ms = 2)

        #ax.plot(med_epoch, med_loss, 'b', alpha = 1, ms = 2)
        ax.set(xlabel = 'Epoch', ylabel='Loss')
        fig.tight_layout()
        plt.savefig(str(self.logdir) + '/loss.jpg', bbox_inches = 'tight')
        plt.close()
        plt.clf()


        #===============
        #Log Loss plot
        #===============
        plt.clf()
        fig, ax = plt.subplots()
        ax.axhline(y = 0, color = 'r', linestyle = '-')
        ax.plot(plt_epoch[sort_idx], plt_loss[sort_idx], 'k', alpha = 0.9, ms = 2)
        #ax.plot(med_epoch, med_loss, 'b', alpha = 1, ms = 2)
        ax.set(xlabel = 'Epoch', ylabel='Loss', xscale = 'log', yscale = 'log')
        fig.tight_layout()
        plt.savefig(str(self.logdir) + '/log_loss.jpg', bbox_inches = 'tight')
        plt.close()
        plt.clf()


trainer = Trainer()
trainer.train()



