# Exact function excerpts from DeepNSD; see NOTICE.md and manifest.json.
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.random_projection import SparseRandomProjection


def get_feature_map_filepaths(feature_map_names, output_dir):
    return {feature_map_name: os.path.join(output_dir, feature_map_name + '.npy')
                                for feature_map_name in feature_map_names}


def recursive_delete_if_empty(path):
    if not os.path.isdir(path):
        return False
    
    recurse_list = [recursive_delete_if_empty(os.path.join(path, filename))
                    for filename in os.listdir(path)]
    
    if all(recurse_list):
        os.rmdir(path)
        return True
    if not all(recurse_list):
        return False


def delete_saved_output(output_filepaths, output_dir = None, remove_empty_output_dir = False):
    for file_path in output_filepaths:
        os.remove(output_filepaths[file_path])
    if output_dir is not None and remove_empty_output_dir:
        output_dir = output_dir.split('/')[0]
        recursive_delete_if_empty(output_dir)


def compute_jl_lemma(n_samples, eps=0.1):
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    return johnson_lindenstrauss_min_dim(n_samples, eps=eps)


def get_feature_map_srps(feature_maps, n_projections=None, upsampling=True, eps=0.1, seed=0,
                        save_outputs=False, output_dir='temp_data/srp', 
                        delete_originals=False, delete_saved_outputs=True):
    
    if n_projections is None:
        if isinstance(feature_maps, np.ndarray):
            n_samples = feature_maps.shape[0]
        if isinstance(feature_maps, dict):
            n_samples = next(iter(feature_maps.values())).shape[0]
        n_projections = compute_jl_lemma(n_samples, eps=eps)
        
    srp = SparseRandomProjection(n_projections, random_state=seed)
    
    def get_srps(feature_map):
        if feature_map.shape[1] <= n_projections and not upsampling:
            srp_feature_map = feature_map
        if feature_map.shape[1] >= n_projections or upsampling:
            srp_feature_map = srp.fit_transform(feature_map)
            
        return srp_feature_map
        
    if isinstance(feature_maps, np.ndarray) and save_outputs:
        raise ValueError('Please provide a dictionary of the form {feature_map_name: feature_map}' + 
                                 'in order to save_outputs.')
        
    if isinstance(feature_maps, np.ndarray) and not save_outputs:
        return srp.fit_transform(feature_maps)
         
    if isinstance(feature_maps, dict) and not save_outputs:
        srp_feature_maps = {}
        for feature_map_name in tqdm(list(feature_maps), desc = 'SRP Extraction (Layer)'):
            srp_feature_maps[feature_map_name] = get_srps(feature_maps[feature_map_name])
            
        if delete_originals:
            feature_maps.pop(feature_map_name)
            
        return srp_feature_maps
    
    if isinstance(feature_maps, dict) and save_outputs:
        output_dir = os.path.join(output_dir, '_'.join(['projections', str(n_projections), 'seed', str(seed)]))
        output_filepaths = get_feature_map_filepaths(feature_maps, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        srp_feature_maps = {}
        for feature_map_name in tqdm(list(feature_maps), desc = 'SRP Extraction (Layer)'):
            output_filepath = output_filepaths[feature_map_name]
            if not os.path.exists(output_filepath):
                srp_feature_maps[feature_map_name] = get_srps(feature_maps[feature_map_name])
                np.save(output_filepath, srp_feature_maps[feature_map_name])
            if os.path.exists(output_filepath):
                srp_feature_maps[feature_map_name] = np.load(output_filepath, allow_pickle=True)
                
            if delete_originals:
                feature_maps.pop(feature_map_name)
                
        if delete_saved_outputs:
            delete_saved_output(output_filepaths, output_dir, remove_empty_output_dir = True)
                
        return srp_feature_maps      
