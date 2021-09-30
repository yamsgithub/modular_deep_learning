from sklearn import datasets
from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import math

import torch

from helper import datagen
from helper.visualise_results import plot_data

def generate_data(dataset, d_size, batchsize=128, num_classes=2):
    X = y = None
    if 'checker_board_rotated' in dataset:
        X, y, num_classes = checker_board_rotated(dataset, d_size)
    elif 'checker_board' in dataset:
        X, y, num_classes = checker_board(dataset, d_size)
    elif 'con_circles' in dataset:
        X, y, num_classes = concentric_circles(dataset, d_size)
    elif 'non_linear' in dataset:
        X, y, num_classes = non_linear(dataset, d_size)
    elif 'multi_class' in dataset:
        X, y, num_classes = multi_class(dataset, d_size, num_classes)                    
    elif 'data1' in dataset:
        X, y, num_classes = data1(dataset, d_size)        
    elif 'data5' in dataset:
        X, y, num_classes = data5(dataset, d_size)

    #plot_data(X, y, num_classes, 'figures/all/'+dataset+'_'+str(num_classes)+'.png')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create trainloader

    trainset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), 
                                              torch.tensor(y_train, dtype=torch.long))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=1, pin_memory=True)
    testset = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                             torch.tensor(y_test, dtype=torch.long))
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                             shuffle=True, num_workers=1, pin_memory=True)


    return X, y, trainset, trainloader, testset, testloader, num_classes

def checker_board_rotated(dataset, size, num_classes=2):
    x_a = [-1.0,-1.0,0.0,0.0]
    x_b = [0.0,0.0,1.0,1.0]
    y_a = [-1.0,0.0,-1.0,0.0]
    y_b = [0.0,1.0,0.0,1.0]
    X = 2 * np.random.random((size,2)) - 1
    y = (np.sum( np.ceil( 2 * X).astype(int), axis=1 ) % 2).astype(float)

    deg = [math.radians(30),math.radians(45),math.radians(60),math.radians(120)]
    X_new = None
    y_new = None
    for i in range(0,4):
        rm = np.asarray([[math.cos(deg[i]),-1*math.sin(deg[i])],
                         [math.sin(deg[i]),math.cos(deg[i])]])
        # select the corresponding rectangle from the checkerboard
        index = (X[:,0]>=x_a[i])&(X[:,0]<=x_b[i])&(X[:,1]>=y_a[i])&(X[:,1]<=y_b[i])
        X_sub = X[index]
        
        # translate bottom left corner of the corresponding rectangle to origin for rotation
        X_sub[:,0] = X_sub[:,0]-x_a[i]
        X_sub[:,1] = X_sub[:,1]-y_a[i]
        
        # rotate
        X_tmp = np.transpose(np.dot(rm, np.transpose(X_sub)))
        
        # translate back bottom left corner of the corresponding rectangle
        X_tmp[:,0] = X_tmp[:,0]+x_a[i]
        X_tmp[:,1] = X_tmp[:,1]+y_a[i]
        
        # normalize to corresponding x and y limits
        r_min = X_tmp[:,0].min()
        r_max = X_tmp[:,0].max()
        X_tmp[:,0] = ((x_b[i]-x_a[i])*(X_tmp[:,0]-r_min)/(r_max-r_min))+x_a[i]
        r_min = X_tmp[:,1].min()
        r_max = X_tmp[:,1].max()
        X_tmp[:,1] = ((y_b[i]-y_a[i])*(X_tmp[:,1]-r_min)/(r_max-r_min))+y_a[i]
        
        if not X_new is None:
            X_new = np.vstack((X_new,X_tmp))
        else:
            X_new = X_tmp
        if not y_new is None:
            y_new = np.concatenate((y_new, y[index]))
        else:
            y_new = y[index]
    
    X = X_new
    y = y_new

    return X, y, num_classes

def checker_board(dataset, size, num_classes=2):
    clf = int(dataset.split('-')[-1])
    
    X = 2 * np.random.random((size,2)) - 1
    def classifier0(X):
        return (np.sum( X * X, axis=1) < 0.66 ).astype(float)
    def classifier1(X): # a 3x2 checkerboard pattern
        return (( np.ceil((3/2)*(X[:,0]+1)).astype(int) + np.ceil( X[:,1]+1).astype(int) ) %2).astype(float)
    def classifier2(X): # a 4x4 checkerboard pattern -- you can use the same method to make up your own checkerboard patterns
        return (np.sum( np.ceil( 2 * X).astype(int), axis=1 ) % 2).astype(float)
    classifiers = [classifier0, classifier1, classifier2]

    y = classifiers[clf]( X )

    return X, y, num_classes

def concentric_circles(dataset, size, num_classes=2):
    np.random.seed(0)
    
    n_samples = size
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.11)
    
    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    X, y = noisy_circles

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    return X, y, num_classes

def non_linear(dataset, size, num_classes=2):
    # Generate non-linear dataset
    split = int(size/3)
    transl = - 3
    X = np.random.normal(size=(size, 2))
    X[0:split, ] = X[0:split,:] + transl
    X[split+1:split*2, ] = X[split+1:split*2,:] - transl
    y = np.zeros((size))
    y[0:split*2] = 1

    return X, y, num_classes

# def multi_class(dataset, size, num_classes=3):
#     from sklearn.datasets import make_classification
#     # Easy decision boundary
#     X,y = make_classification(n_samples=size, n_features=2, n_informative=2, 
#                               n_redundant=0, n_repeated=0, n_classes=num_classes, 
#                               n_clusters_per_class=1,class_sep=2,flip_y=0,
#                               random_state=17)

#     return X, y, num_classes

#from matplotlib import pyplot as plt
def multi_class(dataset, size, num_classes=3):
    print('NUM CLASSES', num_classes)
    #prop_cycle = plt.rcParams['axes.prop_cycle']
    #colors = prop_cycle.by_key()['color']
    omc=datagen.gm_kmc(ininum=1000, nn=0.06, nump=int(size/num_classes),clu=num_classes,noise=0,seed=4058803790, labeler=lambda l:l)#,col=lambda k:colors[k])
    return omc.get_instances(), omc.get_labels(), num_classes

def data1(dataset, size, num_classes=2):
    plt.figure(figsize=(10,5))
    im = plt.imread('figures/data/'+dataset+'.png')
    
    X1 = np.where(im[:,:,0] > 0)[0]
    Y1 = np.where(im[:,:,0] > 0)[1]
    labels1 = np.ones(X1.shape).astype(int)
    X2 = np.where(im[:,:,2] > 0)[0]
    Y2 = np.where(im[:,:,2] > 0)[1]
    labels2 = np.zeros(X2.shape).astype(int)
    X = np.asarray([list(X1)+list(X2), list(Y1)+list(Y2)]).transpose().astype('float32')
    X = preprocessing.scale(X)
    y = list(labels1) + list(labels2)
    
    original_X = X
    original_y = y
    
    X, y = resample(X, y, n_samples=size, replace=False, stratify=y, random_state=0)

    return X, y, num_classes

def data5(dataset, size):
    num_classes = 3
    im = plt.imread('figures/data/'+dataset+'.png')
    X = []
    Y = []
    labels = []
    for i in range(0, num_classes):
        X_tmp = np.where(im[:,:,i] > 0)[0]
        Y_tmp = np.where(im[:,:,i] > 0)[1]
        print(X_tmp[0:5])
        X += list(X_tmp)
        Y += list(Y_tmp)
        labels += list(np.full(X_tmp.shape, i).astype(int))
    X = np.asarray([X, Y]).transpose().astype('float32')
    X = preprocessing.scale(X)
    y = labels
    
    X, y = resample(X, y, n_samples=size, replace=False, stratify=y, random_state=0)
    
    return X, y, num_classes


def create_meshgrid(X):
    #create meshgrid
    resolution = 100 # 100x100 background pixels
    a2d_min, a2d_max = np.min(X[:,0]), np.max(X[:,0])
    b2d_min, b2d_max = np.min(X[:,1]), np.max(X[:,1])
    a, b = np.meshgrid(np.linspace(a2d_min, a2d_max, resolution), 
                       np.linspace(b2d_min, b2d_max, resolution))
    generated_data = torch.tensor(np.c_[a.ravel(), b.ravel()], dtype=torch.float32)

    return generated_data
