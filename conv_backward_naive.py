from builtins import range
import numpy as np
import copy
import math

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    dx, dw, db = None, None, None

    x = cache[0]
    w = cache[1]
    b = cache[2]
    conv_param = cache[3]
    pad = conv_param['pad']
    stride = conv_param['stride']
    (N, C, H, W) = x.shape
    HH = dout.shape[2]
    WW = dout.shape[3]
    
    dx = np.zeros(x.shape)
    db = np.zeros(b.shape)
    dw = np.zeros((N,w.shape[0],w.shape[1],w.shape[2],w.shape[3]))
    for i in range(N):
        for f in range(w.shape[0]):
            db[f]=np.sum(dout[:,f,:,:])
            for k in range(w.shape[2]):
                for j in range(w.shape[3]):
                    for c in range(w.shape[1]):
                        sample = x[i,:,:,:]
                        zero_pad_sample = np.pad(sample, pad_width=((0,0),(pad, pad), (pad, pad)), mode = 'constant')
                        temp = zero_pad_sample[c, k:k+stride*HH:stride, j:j+stride*WW:stride]
                        dw[i,f,c,k,j] = np.sum(dout[i,f,:,:] * temp)
    pass
    dw = np.sum(dw,axis=0)
    
    for k in range(H):
        for j in range(W):
            zero_pad_dout = np.pad(dout, pad_width=((0,0),(0,0),(w.shape[2]-1,w.shape[2]-1),(w.shape[3]-1,w.shape[3]-1)),mode = 'constant')
            for c in range(w.shape[1]):
                for f in range(w.shape[0]):
                    temp_filter = np.flipud(np.fliplr(w[f,c,:,:]))
                    for i in range(N):
                        temp_window = zero_pad_dout[i,f,math.floor((k+pad)/stride):math.floor((k+pad)/stride)+w.shape[2],math.floor((j+pad)/stride):math.floor((j+pad)/stride)+w.shape[3]]
                        #temp_window[stride-1,stride-1]=0
                        dx[i,c,k,j] += np.sum(temp_window*temp_filter)

    return dx, dw, db
