import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from numpy.lib.arraypad import _as_pairs

_integer_types = (np.byte, np.ubyte,  # 8 bits
                      np.short, np.ushort,  # 16 bits
                      np.intc, np.uintc,  # 16 or 32 or 64 bits
                      np.int_, np.uint,  # 32 or 64 bits
                      np.longlong, np.ulonglong)  # 64 bits

_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max)
                   for t in _integer_types}

dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}

dtype_range.update(_integer_ranges)

def _assert_compatible(im1, im2):
    """Raise an error if the shape and dtype do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return

def _as_floats(im1, im2):
    """Promote im1, im2 to nearest appropriate floating point precision."""
    float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
    im1 = np.asarray(im1, dtype=float_type)
    im2 = np.asarray(im2, dtype=float_type)
    return im1, im2

def crop(ar, crop_width, copy=False, order='K'):
    """Crop array `ar` by `crop_width` along each dimension.
    Parameters
    ----------
    ar : array-like of rank N
        Input array.
    crop_width : {sequence, int}
        Number of values to remove from the edges of each axis.
        ``((before_1, after_1),`` ... ``(before_N, after_N))`` specifies
        unique crop widths at the start and end of each axis.
        ``((before, after),)`` specifies a fixed start and end crop
        for every axis.
        ``(n,)`` or ``n`` for integer ``n`` is a shortcut for
        before = after = ``n`` for all axes.
    copy : bool, optional
        If `True`, ensure the returned array is a contiguous copy. Normally,
        a crop operation will return a discontiguous view of the underlying
        input array.
    order : {'C', 'F', 'A', 'K'}, optional
        If ``copy==True``, control the memory layout of the copy. See
        ``np.copy``.
    Returns
    -------
    cropped : array
        The cropped array. If ``copy=False`` (default), this is a sliced
        view of the input array.
    """
    ar = np.array(ar, copy=False)
    crops = _as_pairs(crop_width, ar.ndim, as_index=True)
    slices = tuple(slice(a, ar.shape[i] - b)
                   for i, (a, b) in enumerate(crops))
    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped

def MSE(im1, im2):
    """Compute the mean-squared error between two images.
    Parameters
    ----------
    im1, im2 : ndarray
        Image.  Any dimensionality.
    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.
    """
    _assert_compatible(im1, im2)
    im1, im2 = _as_floats(im1, im2)
    return np.mean(np.square(im1 - im2), dtype=np.float64)


def PSNR(ref_img, pred_img):
    """
    Compute average PSNR score for a batch of pairs of images

    :param ref_img: ndarray of shape (batch_size, width, height, channels) with target noise-free image
    :param pred_img: ndarray of shape (batch_size, width, height, channels) with predicted denoised image
    :return: float, averaged over batch PSNR score
    """

    _assert_compatible(ref_img, pred_img)

    dmin, dmax = dtype_range[ref_img.dtype.type]
    true_min, true_max = np.min(ref_img), np.max(ref_img)
    if true_max > dmax or true_min < dmin:
        raise ValueError(
            "im_true has intensity values outside the range expected for "
            "its data type.  Please manually specify the data_range")
    if true_min >= 0:
        # most common case (255 for uint8, 1 for float)
        data_range = dmax
    else:
        data_range = dmax - dmin

    ref_img, pred_img = _as_floats(ref_img, pred_img)

    err = MSE(ref_img, pred_img)
    return 10 * np.log10((data_range ** 2) / err)


def SSIM(X, Y, win_size=None, gradient=False,
                 data_range=None, multichannel=False, gaussian_weights=False,
                 full=False, **kwargs):
    """Compute the mean structural similarity index between two images.
    Parameters
    ----------
    X, Y : ndarray
        Image. Any dimensionality.
    win_size : int or None
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient with respect to Y.
    data_range : float, optional
        The data range of the input image (distance between minimum and
        maximum possible values). By default, this is estimated from the image
        data-type.
    multichannel : bool, optional
        If True, treat the last dimension of the array as channels. Similarity
        calculations are done independently for each channel then averaged.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, also return the full structural similarity image.
    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        Algorithm parameter, K1 (small constant, see [1]_).
    K2 : float
        Algorithm parameter, K2 (small constant, see [1]_).
    sigma : float
        Standard deviation for the Gaussian when `gaussian_weights` is True.
    Returns
    -------
    mssim : float
        The mean structural similarity over the image.
    grad : ndarray
        The gradient of the structural similarity index between X and Y [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.
    Notes
    -----
    To match the implementation of Wang et. al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, and `use_sample_covariance` to False.
    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`
    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       :arXiv:`0901.0065`
       :DOI:`10.1007/s10043-009-0119-z`
    """
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if multichannel:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    multichannel=False,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = X.shape[-1]
        mssim = np.empty(nch)
        if gradient:
            G = np.empty(X.shape)
        if full:
            S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = SSIM(X[..., ch], Y[..., ch], **args)
            if gradient and full:
                mssim[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7   # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        assert X.dtype == Y.dtype, \
            "Inputs have mismatched dtype. Setting data_range based on X.dtype."
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin

    ndim = X.ndim

    if gaussian_weights:
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma, 'truncate': truncate}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * X
        grad += filter_func(-S / B2, **filter_args) * Y
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim