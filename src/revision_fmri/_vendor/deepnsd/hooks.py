# Exact function excerpts from DeepNSD; see NOTICE.md and manifest.json.
import logging
import numpy as np
import torch
import torch.nn as nn
from functools import lru_cache
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from tqdm.auto import tqdm


@lru_cache(128)
def warn_once(msg: str):
    logging.warning(msg)


def prep_model_for_extraction(model, inputs=None):
    if model.training:
        model = model.eval()
    if not next(model.parameters()).is_cuda:
        if torch.cuda.is_available():
            model = model.cuda()

    if inputs == None:
        return(model)
    
    if inputs != None:
        if next(model.parameters()).is_cuda:
            inputs = ({k:v.cuda() for k,v in inputs.items()} 
                      if isinstance(inputs, Mapping) else inputs.cuda())
            
        return(model, inputs)


def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)


def get_module_name(module, module_list):
    class_name = str(module.__class__).split(".")[-1].split("'")[0]
    class_count = str(sum(class_name in module for module in module_list) + 1)
    
    return '-'.join([class_name, class_count])


def remove_duplicate_feature_maps(feature_maps, method = 'hashkey', return_matches = False, use_tqdm = False):
    matches, layer_names = [], list(feature_maps.keys())
        
    if method == 'iterative':
        
        target_iterator = tqdm(range(len(layer_names)), leave = False) if use_tqdm else range(len(layer_names))
        
        for i in target_iterator:
            for j in range(i+1,len(layer_names)):
                layer1 = feature_maps[layer_names[i]].flatten()
                layer2 = feature_maps[layer_names[j]].flatten()
                if layer1.shape == layer2.shape and torch.all(torch.eq(layer1,layer2)):
                    if layer_names[j] not in matches:
                        matches.append(layer_names[j])

        deduplicated_feature_maps = {key:value for (key,value) in feature_maps.items()
                                         if key not in matches}
        
    if method == 'hashkey':
        
        target_iterator = tqdm(layer_names, leave = False) if use_tqdm else layer_names
        layer_lengths = [len(tensor.flatten()) for tensor in feature_maps.values()]
        random_tensor = torch.rand(np.array(layer_lengths).max())
        
        tensor_dict = defaultdict(lambda:[])
        for layer_name in target_iterator:
            target_tensor = feature_maps[layer_name].flatten()
            tensor_dot = torch.dot(target_tensor, random_tensor[:len(target_tensor)])
            tensor_hash = np.array(tensor_dot).tobytes()
            tensor_dict[tensor_hash].append(layer_name)
            
        matches = [match for match in list(tensor_dict.values()) if len(match) > 1]
        layers_to_keep = [tensor_dict[tensor_hash][0] for tensor_hash in tensor_dict]
        
        deduplicated_feature_maps = {key:value for (key,value) in feature_maps.items()
                                         if key in layers_to_keep}
    
    if return_matches:
        return(deduplicated_feature_maps, matches)
    
    if not return_matches:
        return(deduplicated_feature_maps)


def check_for_input_axis(feature_map, input_size):
    axis_match = [dim for dim in feature_map.shape if dim == input_size]
    return True if len(axis_match) == 1 else False


def reset_input_axis(feature_map, input_size):
    input_axis = feature_map.shape.index(input_size)
    return torch.swapaxes(feature_map, 0, input_axis)


def get_feature_maps(model, inputs, layers_to_retain = None, remove_duplicates = True, 
                     enforce_input_shape = True, verbose = False):
    
    model, inputs = prep_model_for_extraction(model, inputs)
    
    if layers_to_retain:
        if not isinstance(layers_to_retain, list):
            layers_to_retain = [layers_to_retain]
    
    def fix_outputs_shape(inputs, outputs, module_name):
        input_shape = inputs[next(iter(inputs))].shape[0] if isinstance(inputs, Mapping) else inputs.shape[0]
    
        if len(outputs.shape) == 0:
            warn_once('Output in {} is empty. Skipping...'.format(module_name))
            return None
        if enforce_input_shape:
            if outputs.shape[0] == input_shape:
                return outputs
            if outputs.shape[0] != input_shape:
                if check_for_input_axis(outputs, input_shape):
                    return reset_input_axis(outputs, input_shape)
                if not check_for_input_axis(outputs, input_shape):
                    warn_once('Ambiguous input axis in {}. Skipping...'.format(module_name))
                    return None
        if not enforce_input_shape:
            return outputs
    
    def register_hook(module):
        def hook(module, input, output):
            def process_output(output, module_name):
                if layers_to_retain is None or module_name in layers_to_retain:
                    if isinstance(output, torch.Tensor):
                        outputs = output.cpu().detach().type(torch.FloatTensor)
                        outputs = fix_outputs_shape(inputs, outputs, module_name)
                        feature_maps[module_name] = outputs
                if layers_to_retain is not None and module_name not in layers_to_retain:
                    feature_maps[module_name] = None
                            
            module_name = get_module_name(module, feature_maps)
            
            if not any([isinstance(output, type_) for type_ in (tuple,list)]):
                process_output(output, module_name)
            
            if any([isinstance(output, type_) for type_ in (tuple,list)]):
                for output_i, output_ in enumerate(output):
                    module_name_ = '-'.join([module_name, str(output_i+1)])
                    process_output(output_, module_name_)
                    
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))
    
    feature_maps = OrderedDict()
    hooks = []
    
    model.apply(convert_relu)
    model.apply(register_hook)
    with torch.no_grad():
        model(**inputs) if isinstance(inputs, Mapping) else model(inputs)

    for hook in hooks:
        hook.remove()
        
    feature_maps = {map:features for (map,features) in feature_maps.items()
                        if features is not None}
        
    if remove_duplicates == True:
        feature_maps = remove_duplicate_feature_maps(feature_maps)
        
    return(feature_maps)
