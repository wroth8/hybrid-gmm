'''
@author Wolfgang Roth
'''

import numpy as np

from os import remove
from urllib import urlretrieve
from zipfile import ZipFile

def downloadMnistBasic(filename):
    '''
    Downloads the MNIST Basic data set and stores it to the given file in npz
    format
    '''
    url_data = 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip'
    
    tmp_data = 'tmp_mnist_basic.zip'
    tmp_data_train = 'mnist_train.amat'
    tmp_data_test = 'mnist_test.amat'
    
    print 'Downloading MNIST Basic...'
    urlretrieve(url_data, tmp_data)
    print 'Downloading finished'
    
    print 'Uncompressing zip files...'
    with ZipFile(tmp_data, 'r') as zipfile:
        zipfile.extractall('.')

    print 'Loading uncompressed data...'
    x_tr_np = np.loadtxt(tmp_data_train, dtype=np.float32)
    x_te_np = np.loadtxt(tmp_data_test, dtype=np.float32)
    t_tr_np = x_tr_np[:, -1].astype(np.uint8)
    t_te_np = x_te_np[:, -1].astype(np.uint8)
    x_tr_np = x_tr_np[:, :-1]
    x_te_np = x_te_np[:, :-1]
    x_va_np = x_tr_np[10000:]
    t_va_np = t_tr_np[10000:]
    x_tr_np = x_tr_np[:10000]
    t_tr_np = t_tr_np[:10000]
    
    print 'Removing temporary files...'
    remove(tmp_data)
    remove(tmp_data_train)
    remove(tmp_data_test)
    
    print 'Storing MNIST Basic data to ''%s''' % (filename)
    np.savez_compressed(filename,
                        x_tr_np=x_tr_np, t_tr_np=t_tr_np,
                        x_va_np=x_va_np, t_va_np=t_va_np,
                        x_te_np=x_te_np, t_te_np=t_te_np)
    
    print 'MNIST Basic is now ready'