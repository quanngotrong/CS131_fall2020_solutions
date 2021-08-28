"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for m in range(Hi):
        for n in range(Wi):
            sum = 0
            for i in range(Hk):
                for j in range(Wk):
                    x = m + Hk//2 -i
                    y = n + Wk//2 - j
                    if x > Hi-1 or x< 0 or y >Wi-1 or y < 0:
                        sum+= 0
                    else:
                        sum+= kernel[i][j] * image[x][y]
            out[m][n] += sum
            
                        
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    n_h = H + pad_height * 2
    n_w = W + pad_width * 2
    out = np.zeros((n_h, n_w))
    
    out[pad_height: -pad_height,pad_width: -pad_width] = image
#     out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
#     out[pad_height:pad_height , pad_width:pad_width + W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
#     tem = zero_pad(image, Hk//2,Wk//2)
#     kernel = np.flip(np.flip(kernel,0),1)
#     for i in range(Hk//2, Hi+Hk//2):
#         for j in range(Wk//2, Wi + Wk//2):
#             out[i-Hk//2,j-Wk//2]= np.sum(
#                 np.multiply(tem[i - Hk//2: i+Hk -Hk//2, j-Wk//2: j+Wk-Wk//2], kernel)
#             )
  
    image = zero_pad(image, Hk//2, Wk//2)
    kernel = np.flip(kernel, 0)
    kernel = np.flip(kernel, 1)
    for m in range(Hi):
        for n in range(Wi):
            out[m, n] =  np.sum(image[m: m+Hk, n: n+Wk] * kernel)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    flipped_g = np.flip(np.flip(g,0),1)
    out = conv_fast(f,flipped_g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    
    out = cross_correlation(f ,g - np.mean(g))
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = np.shape(f)
    Hg, Wg= np.shape(g)
    out = np.zeros((Hf, Wf))
    
    Hpad = Hg//2
    Wpad = Wg//2
    
    std_g = np.std(g)
    mean_g = np.mean(g)
    nor_g = (g - mean_g) / std_g
    
    pad_f = zero_pad(f, Hpad, Wpad)
    
    for i in range(Hpad , Hpad + Hf):
        for j in range(Wpad, Wpad + Wf):
            patch_f = pad_f[i - Hpad: i+ Hg - Hpad,j-Wpad:j+Wg -Wpad ]
            mean_patch_f = np.mean(patch_f)
            std_patch_f=np.std(patch_f)
            
            nor_patch_f = (patch_f - mean_patch_f)/std_patch_f
            
            out[i - Hpad, j-Wpad] = np.sum((nor_g*nor_patch_f))
    
    
    ### END YOUR CODE

    return out
