import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
import torch.cuda as cuda
from torchvision import transforms
from PIL import Image
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def preprocess_image(image_path, transform, transform_normalize):
    img = Image.open(image_path)
    transformed_img = transform(img)
    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)   
    return input, transformed_img


def visualize_attributions(attributions, transformed_img, cmap, method='heat_map', **kwargs):
    _ = viz.visualize_image_attr(np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 method=method,
                                 cmap=cmap,
                                 **kwargs)


def visualise_model(model, image_path):
    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)

    # Captum Model Part
    ig = IntegratedGradients(model)

    transform = transforms.Compose([
    transforms.Resize(100),
    transforms.CenterCrop(100),
    transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]     
    )

    input, transformed_img = preprocess_image(image_path, transform, transform_normalize)

    output = model(input.to(device))
    output = F.softmax(output, dim=1)
    pred_label_idx = torch.topk(output, 1)
    _, pred_label_idx = torch.topk(output, 1)

    integrated_gradients = IntegratedGradients(model)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)

    attributions_ig = integrated_gradients.attribute(input.to(device), 
                                                    target=pred_label_idx, 
                                                    n_steps=200)
    
    visualize_attributions(attributions_ig, transformed_img, default_cmap,
                        method='heat_map', show_colorbar=True,
                        sign='positive', outlier_perc=1)
    
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(input.to(device), 
                                                nt_samples=10, 
                                                nt_type='smoothgrad_sq', 
                                                target=pred_label_idx)
    visualize_attributions(attributions_ig_nt, transformed_img, default_cmap,
                           method='heat_map', show_colorbar=True)
                                

    # Defining baseline distribution of images
    torch.manual_seed(0)
    np.random.seed(0)
    gradient_shap = GradientShap(model)
    rand_img_dist = torch.cat([input.to(device) * 0, input.to(device)* 1])
    attributions_gs = gradient_shap.attribute(input.to(device),
                                            n_samples=50,
                                            stdevs=0.0001,
                                            baselines=rand_img_dist,
                                            target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "absolute_value"],
                                        cmap=default_cmap,
                                        show_colorbar=True)


    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(input.to(device),
                                        strides = (3, 8, 8),
                                        target=pred_label_idx,
                                        sliding_window_shapes=(3,15, 15),
                                        baselines=0)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                        ["original_image", "heat_map"],
                                        ["all", "positive"],
                                        show_colorbar=True,
                                        outlier_perc=2)

    plt.show()                           