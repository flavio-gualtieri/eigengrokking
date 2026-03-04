import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from skdim import TwoNN
except Exception:
    try:
        from skdim.id import TwoNN
    except Exception:
        TwoNN = None



def PredDimFromRatio( locscale_ratio, slope, intercept ):
  pred_val = np.exp( ( locscale_ratio - intercept)/slope )

  return pred_val


def transformed_nn_ratios( points, all_nn_pairs, max_neighbors = 20 ):
    nn = NearestNeighbors(n_neighbors=max_neighbors + 1, algorithm='auto')
    nn.fit(points)
    
    dists, _ = nn.kneighbors(points)

    sorted_dists = dists[:, 1:]
    
    ret_results = []
    
    for p in all_nn_pairs:
      ratios = sorted_dists[ :, p[1] ] / sorted_dists[ :, p[0] ]
      ratios = ratios[ ratios > 1.0 ]
      ratios = ratios[ np.isfinite( ratios ) ]
      transformed = np.log(np.log(ratios))
              
      cur_res = np.mean( transformed )
      ret_results.append( -cur_res )
      
    return ret_results


def l2n2_dim_est(points):
    use_slope =  0.904825
    use_intercept = 0.65670
    all_ratios = transformed_nn_ratios( points, [[0,1]] )
      
    # get the predictions from all the curves
    pred_id = PredDimFromRatio( all_ratios[0], use_slope, use_intercept )
    return pred_id


def compute_layer_outputs(model, dataset, device, batch_size=50):
#def get_layer_outputs_and_labels(model, test_loader, device):
    """
    Returns:
        outputs_per_layer: list of length L where each element is a tensor
                           containing the output of layer i for all test samples.
        all_labels: tensor containing the labels of all test samples (in order).
    
    """
    model.eval()
    layers = list(model)
    L = len(layers)

    collected = [[] for _ in range(L)]
    labels_list = []

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            # Handle (x, y) or x-only datasets
            if isinstance(batch, (list, tuple)):
                x, y = batch
                labels_list.append(y.detach().cpu())
            else:
                x = batch
                y = None  # if no labels available
            
            x = x.to(device)

            out = x
            for i, layer in enumerate(layers):
                out = layer(out)
                collected[i].append(out.detach().cpu())

    # Concatenate all stored outputs and labels
    outputs_per_layer = [torch.cat(collected[i], dim=0) for i in range(L)]
    all_labels = torch.cat(labels_list, dim=0) if labels_list else None

    return outputs_per_layer, all_labels


def compute_intrinsic_dims(layer_outputs, sample_limit=10000):
    """Compute intrinsic dimensionality for each layer output using skdim's TwoNN.

    Returns a NumPy array of shape (num_layers,) with the estimated intrinsic dimension
    for each layer. If `TwoNN` is not available, raises ImportError. Layers that fail
    estimation will be assigned np.nan.
    """
    if TwoNN is None:
        raise ImportError("skdim TwoNN not found. Install 'skdim' to compute intrinsic dims.")

    dims = []
    for arr in layer_outputs:
        # Convert torch.Tensor inputs to NumPy arrays (TwoNN expects NumPy input)
        if isinstance(arr, torch.Tensor):
            np_arr = arr.cpu().numpy()
        else:
            np_arr = np.asarray(arr)

        # np_arr is (num_samples, features)
        n_samples = np_arr.shape[0]
        if n_samples < 3:
            dims.append(np.nan)
            continue

        # subsample if dataset is large to keep TwoNN fast
        if n_samples > sample_limit:
            idx = np.random.choice(n_samples, sample_limit, replace=False)
            X = np_arr[idx].astype(np.float64)
        else:
            X = np_arr.astype(np.float64)

        try:
            estimator = TwoNN()
            estimator.fit(X)
            # estimator.dimension_ is typically a scalar
            d = float(getattr(estimator, 'dimension_', np.nan))
        except Exception:
            d = np.nan
        dims.append(d)
    return np.array(dims)


def compute_l2n2_intrinsic_dims(layer_outputs, sample_limit=10000):
    """Compute intrinsic dimensionality for each layer output using l2n2_dim_est.

    Returns a NumPy array of shape (num_layers,) with the estimated intrinsic dimension
    for each layer. Layers that fail estimation will be assigned np.nan.
    """
    dims = []
    for arr in layer_outputs:
        # Convert torch.Tensor inputs to NumPy arrays
        if isinstance(arr, torch.Tensor):
            np_arr = arr.cpu().numpy()
        else:
            np_arr = np.asarray(arr)

        # np_arr is (num_samples, features)
        n_samples = np_arr.shape[0]
        if n_samples < 3:
            dims.append(np.nan)
            continue

        # subsample if dataset is large
        if n_samples > sample_limit:
            idx = np.random.choice(n_samples, sample_limit, replace=False)
            X = np_arr[idx].astype(np.float64)
        else:
            X = np_arr.astype(np.float64)

        try:
            d = l2n2_dim_est(X)
        except Exception:
            d = np.nan
        dims.append(d)
    return np.array(dims)