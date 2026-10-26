# Exact function excerpts from DeepNSD; see NOTICE.md and manifest.json.
import torchvision.transforms as transforms


def get_torchvision_model(model_path, pretrained = True):
    from torchvision import models
    return eval(f'{model_path}(pretrained = {pretrained})')


def get_torchvision_transforms(train_type, input_type = 'PIL'):
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}
    
    base_transforms = [transforms.Resize((224,224)), transforms.ToTensor()]
    
    if train_type == 'random': specific_transforms = base_transforms
    
    if train_type == 'classification' or train_type == 'imagenet':
        specific_transforms = base_transforms + [transforms.Normalize(**imagenet_stats)]
    
    if input_type == 'PIL':
        recommended_transforms = specific_transforms
    if input_type == 'numpy':
        recommended_transforms = [transforms.ToPILImage()] + specific_transforms
        
    return transforms.Compose(recommended_transforms)
