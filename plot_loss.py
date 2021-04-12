import pickle
import numpy
import matplotlib.pyplot as plt


all_reconloss = []
all_ncut = []

with open('reconstruction_loss.pkl','rb') as f:
    while True:
        try:
            all_reconloss.append(pickle.load(f))
        except EOFError:
            break

with open('n_cut_loss.pkl','rb') as fb:
    while True:
        try:
            all_ncut.append(pickle.load(fb))
        except EOFError:
            break


recon = numpy.array(all_reconloss) # changing to array
recon = recon.reshape((-1,1)) #making a single column
div = int(recon.shape[0]/100) # getting the last iteration divisible by 100
recon = recon[:div*100]
recon = recon.reshape((-1,100)).mean(axis=1) # takin mean of 100 iterations



n_cut = numpy.array(all_ncut)
n_cut = n_cut.reshape((-1,1))
div = int(n_cut.shape[0]/100)
n_cut = n_cut[:div*100]
n_cut = n_cut .reshape((-1,100)).mean(axis=1)

plt.figure(1)
#plt.subplot(211)
plt.title("Reconstruction Loss")
plt.ylabel("Loss")
plt.xlabel("Per 100 Iteration (Batch Size 10)")
plt.plot(recon)

plt.figure(2)
#plt.subplot(212)

plt.title("N-Cut Loss")
plt.ylabel("Loss")
plt.xlabel("Per 100 Iteration (Batch Size 10)")
plt.plot(n_cut)
plt.show()
