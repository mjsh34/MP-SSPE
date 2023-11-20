import torch
from torch import nn
import numpy as np
from scipy import linalg
from collections import defaultdict


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


# eval_scripts/final_evaluation.py
def evaluate_fid(ground_truth_motion_loader, gru_classifier_for_fid, motion_loaders, device, file):
    ret = defaultdict(dict)
    print('========== Evaluating FID ==========')
    ground_truth_activations, ground_truth_labels = \
        calculate_activations_labels(ground_truth_motion_loader, gru_classifier_for_fid, device)
    if ground_truth_activations is not None:
        ground_truth_statistics = calculate_activation_statistics(ground_truth_activations)
    else:
        ground_truth_statistics = None

    for motion_loader_name, motion_loader in motion_loaders.items():
        gru_classifier_for_fid.eval()
        activations, labels = calculate_activations_labels(motion_loader, gru_classifier_for_fid, device)
        if activations is not None and ground_truth_statistics is not None:
            statistics = calculate_activation_statistics(activations)
            fid = calculate_fid(ground_truth_statistics, statistics)
            #diversity, multimodality = \
            #    calculate_diversity_multimodality(activations, labels, len(dataset_opt.label_dec))
        else:
            fid = float('nan')

        ret['fid'][motion_loader_name] = fid
    return ret


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1],
                                      statistics_2[0], statistics_2[1])


def calculate_activations_labels(motion_loader, classifier, device):
    print('Calculating Activations...')
    activations = []
    labels = []

    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            batch_motion, batch_label = batch
            batch_motion = torch.clone(batch_motion).float().detach_().to(device)
            #breakpoint()

            assert batch_motion is not None
            activations.append(classifier(batch_motion, None))
            #labels.append(batch_label)

        if len(activations) > 0:
            activations = torch.cat(activations, dim=0)
        else:
            activations = None
    return activations, labels


def calculate_activation_statistics(activations):
    activations = activations.cpu().numpy()
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


class MotionDiscriminator(nn.Module):
    def __init__(self, device, input_size, hidden_size, hidden_layer, output_size=1, use_noise=None):
        super(MotionDiscriminator, self).__init__()

        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise

        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

        self.initial_hs_random = True

    def forward(self, motion_sequence, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            motion_sequence = motion_sequence.permute(1, 0, 2)
            if self.initial_hs_random:
                hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin2, _

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size,
                device=self.device, requires_grad=False)


# eval_scripts/load_classifier.py
classifier_model_files = {
    'ntu_rgbd_vibe': 'data/action2motion/model_file/action_recognition_model_vibe_v2.tar',
    'humanact12': 'data/action2motion/model_file/action_recognition_model_humanact12.tar',
    'mocap': 'data/action2motion/model_file/action_recognition_model_mocap_new.tar'
}


class MotionDiscriminatorForFID(MotionDiscriminator):
    def forward(self, motion_sequence, hidden_unit=None):
        # dim (motion_length, num_samples, hidden_size)
        if hidden_unit is None:
            motion_sequence = motion_sequence.permute(1, 0, 2)
            if self.initial_hs_random:
                hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        # dim (num_samples, 30)
        lin1 = self.linear1(gru_o[-1, :, :])
        lin1 = torch.tanh(lin1)
        return lin1


def load_classifier_for_fid(device, **opt):
    model = torch.load(classifier_model_files[opt['dataset_type']], map_location=device)
    classifier = MotionDiscriminatorForFID(
            device, opt['input_size_raw'], 128, 2, 12).to(device)
    if 'initial_hs_random' in opt:
        classifier.initial_hs_random = opt['initial_hs_random']
    classifier.load_state_dict(model['model'])
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    return classifier


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    nj = 24
    fid_cls = load_classifier_for_fid(device, input_size_raw=3*nj,
            dataset_type='humanact12')
    breakpoint()
