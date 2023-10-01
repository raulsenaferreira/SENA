import os

import h5py
import warnings
from tqdm import tqdm

import numpy as np
from scipy.special import softmax
import torch
from torchvision.models.feature_extraction import create_feature_extractor
import torchattacks

import models

from Params.params_networks import *
from Params.params_dataset import *

warnings.filterwarnings("ignore")


class FeatureExtractor:
    def __init__(self, network, id_dataset, layers_ids, device_name=None):
        self.network = network
        self.id_dataset = id_dataset
        self.layers_id = layers_ids
        self.layers_id.sort()
        self.n_classes_id = n_classes_dataset[id_dataset]

        self._check_accepted_network()
        self.layers = layers[network]

        self._check_accepted_id_dataset()
        self._check_accepted_layers_id()

        if device_name is None:
            device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device_name = device_name
        self.device = torch.device(self.device_name)

        self.linear_weights, self.linear_bias = None, None

        self.model_dataset_name = self.network + "_" + self.id_dataset + ".pth"
        self._load_model()
        self.model.eval()
        self.model.to(self.device)

    def get_features(self, dataset, save=True):
        if not os.path.exists(save_features_path):
            os.makedirs(save_features_path)

        predictions = None
        logits = None
        softmax_values = None
        labels = None
        images = None

        layers_names = [list(self.layers.items())[i][0] for i in self.layers_id]
        to_extract, file_names_to_extract = [], []
        features = [[] for _ in range(len(self.layers_id))]
        for i, l in enumerate(layers_names):
            perturbations = ""
            try:
                if dataset.additional_transform is not None and dataset.adversarial_attack is not None:
                    perturbations += "_" + dataset.additional_transform + "_" + dataset.adversarial_attack
                elif dataset.additional_transform is not None:
                    perturbations += "_" + dataset.additional_transform
                elif dataset.adversarial_attack is not None:
                    perturbations += "_" + dataset.adversarial_attack
            except:
                print('dataset object has no transform param')

            file_name = save_features_path + "%s_%s%s__%s_%s_%s.h5" % (dataset.name, dataset.split,
                                                                       perturbations,
                                                                       dataset.network, self.id_dataset, l)
            if os.path.exists(file_name):
                images, features[i], logits, softmax_values, predictions, labels = self._load_features(file_name)
            else:
                to_extract.append(i)
                file_names_to_extract.append(file_name)

        torch.cuda.empty_cache()
        if len(to_extract) != 0:
            if dataset.adversarial_attack is None:
                images, features_extracted, logits, softmax_values, predictions, labels = self._extract_features(dataset,
                                                                                                         to_extract,
                                                                                                         layers_names)
            else:
                images, features_extracted, logits, \
                    softmax_values, predictions, labels = self._extract_features_adv(dataset, to_extract, layers_names)
            for i in range(len(layers_names)):
                if i in to_extract:
                    features[i] = features_extracted[0]
                    if save:
                        self._save_features(images, features[i], logits, softmax_values,
                                            predictions, labels, file_names_to_extract[0])
                    features_extracted = list(features_extracted)  # ensuring that pop works by forcing it to be a list
                    features_extracted.pop(0)
                    file_names_to_extract.pop(0)

        return images, features, logits, softmax_values, predictions, labels

    def _extract_features(self, dataset, to_extract, layers_names):
        print('Extracting layers: %s' % str([layers_names[i] for i in to_extract])[1:-1])
        features = [[] for _ in range(len(to_extract))]
        predicted_classes = []
        all_images = []
        all_labels = []
        all_logits = []
        all_softmax = []

        layers_selected = dict([list(self.layers.items())[self.layers_id[i]] for i in to_extract])
        layers_refs = [list(self.layers.items())[self.layers_id[i]][1] for i in to_extract]

        feature_extractor = create_feature_extractor(self.model, return_nodes=layers_selected)

        with torch.no_grad():
            for data in tqdm(dataset.dataloader):
                images, labels = data[0].to(self.device), data[1].to(self.device)

                outputs = feature_extractor(images)
                try:
                    for i, l in enumerate(layers_refs):
                        # print("outputs from try", np.shape(outputs[l]), outputs[l])
                        features[i].append(torch.mean(outputs[l], (2, 3)))
                except:
                    for i, l in enumerate(layers_refs):
                        # print("outputs from except", np.shape(outputs[l]), outputs[l])
                        features[i].append(outputs[l])

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                predicted_classes.append(predicted)
                all_images.append(images)
                all_labels.append(labels)
                all_logits.append(outputs.data)
                all_softmax.append(softmax(outputs.data.cpu().detach().numpy(), axis=1))

        try:
            for i in range(len(features)):
                features[i] = torch.cat(features[i], dim=0)
                features[i] = features[i].cpu().detach().numpy()
        except:
            # print(np.shape(features))
            features = torch.tensor(features)
            features = features.cpu().detach().numpy()


        predicted_classes = torch.cat(predicted_classes)
        predicted_classes = predicted_classes.cpu().detach().numpy()
        all_images = torch.cat(all_images)
        all_images = all_images.cpu().detach().numpy()
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cpu().detach().numpy()
        all_logits = torch.cat(all_logits)
        all_logits = all_logits.cpu().detach().numpy()
        all_softmax = np.concatenate(all_softmax)

        return all_images, features, all_logits, all_softmax, predicted_classes, all_labels

    def _extract_features_adv(self, dataset, to_extract, layers_names):
        print('Extracting layers: %s' % str([layers_names[i] for i in to_extract])[1:-1])

        attacker = AdversarialAttack(dataset.adversarial_attack, self.model)

        features = [[] for _ in range(len(to_extract))]
        predicted_classes = []
        all_images = []
        all_labels = []
        all_logits = []
        all_softmax = []

        layers_selected = dict([list(self.layers.items())[self.layers_id[i]] for i in to_extract])
        layers_refs = [list(self.layers.items())[self.layers_id[i]][1] for i in to_extract]

        feature_extractor = create_feature_extractor(self.model, return_nodes=layers_selected)

        for data in tqdm(dataset.dataloader):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            images = attacker.run(images, labels).to(self.device)
            outputs = feature_extractor(images)

            try:
                for i, l in enumerate(layers_refs):
                    # print("outputs from try", np.shape(outputs[l]), outputs[l])
                    features[i].append(torch.mean(outputs[l], (2, 3)))
            except:
                for i, l in enumerate(layers_refs):
                    # print("outputs from except", np.shape(outputs[l]), outputs[l])
                    features[i].append(outputs[l])

            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_classes.append(predicted)
            all_images.append(images)
            all_labels.append(labels)
            all_logits.append(outputs.data)
            all_softmax.append(softmax(outputs.data.cpu().detach().numpy(), axis=1))

        # for i in range(len(features)):
        #    features[i] = np.concatenate(features[i], axis=0)
        try:
            for i in range(len(features)):
                features[i] = np.concatenate(features[i], axis=0)
        except:
            with torch.no_grad():
                features = np.array([t.numpy() for t in features[0]]) #torch.tensor(features[0])

        predicted_classes = torch.cat(predicted_classes)
        predicted_classes = predicted_classes.cpu().detach().numpy()
        all_images = torch.cat(all_images)
        all_images = all_images.cpu().detach().numpy()
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cpu().detach().numpy()
        all_logits = torch.cat(all_logits)
        all_logits = all_logits.cpu().detach().numpy()
        all_softmax = np.concatenate(all_softmax)

        return all_images, features, all_logits, all_softmax, predicted_classes, all_labels

    @staticmethod
    def _save_features(images, features, logits, softmax_values, predictions, labels, file_name):
        hf = h5py.File(file_name, 'w')
        hf.create_dataset('images', data=images)
        hf.create_dataset('features', data=features)
        hf.create_dataset('logits', data=logits)
        hf.create_dataset('softmax', data=softmax_values)
        hf.create_dataset('predictions', data=predictions)
        hf.create_dataset('labels', data=labels)
        hf.close()

    @staticmethod
    def _load_features(file_name):
        hf = h5py.File(file_name, 'r')
        features = np.array(hf.get("features"))
        logits = np.array(hf.get("logits"))
        softmax_values = np.array(hf.get("softmax"))
        predictions = np.array(hf.get("predictions"))
        labels = np.array(hf.get("labels"))
        images = np.array(hf.get("images"))
        hf.close()

        return images, features, logits, softmax_values, predictions, labels

    def _check_accepted_network(self):
        accepted_networks = list(mean_transform.keys())
        if self.network not in accepted_networks:
            raise ValueError("Accepted network architectures are: %s" % str(accepted_networks)[1:-1])

    def _check_accepted_id_dataset(self):
        accepted_dataset = list(n_classes_dataset.keys())
        if self.id_dataset not in accepted_dataset:
            raise ValueError("Accepted ID datasets are: %s" % str(accepted_dataset)[1:-1])

    def _check_accepted_layers_id(self):
        n_layers = len(list(self.layers.items()))
        if self.layers_id[0] < 0:
            raise ValueError("All layers IDs must be >= 0")
        elif self.layers_id[-1] > n_layers:
            raise ValueError("All layers IDs must be <= %i" % n_layers)

    def _load_model(self):
        if self.network == "resnet":
            self._load_resnet()
        elif self.network == "densenet":
            self._load_densenet()
        elif self.network == "cnn":
            self._load_cnn()
        else:
            pass

    def _load_cnn(self):
        self.model = models.CNN()
        self.model.load_state_dict(torch.load(models_path + self.model_dataset_name,
                                              map_location=self.device_name))
        self.linear_weights = self.model.fc2.weight.cpu().detach().numpy()
        self.linear_bias = self.model.fc2.bias.cpu().detach().numpy()

    def _load_resnet(self):
        self.model = models.ResNet34(num_c=self.n_classes_id)
        self.model.load_state_dict(torch.load(models_path + self.model_dataset_name,
                                              map_location=self.device_name))
        self.linear_weights = self.model.linear.weight.cpu().detach().numpy()
        self.linear_bias = self.model.linear.bias.cpu().detach().numpy()

    def _load_densenet(self):
        from torchvision import models as torch_model
        '''
        # if self.id_dataset == "svhn":
        self.model = models.DenseNet3(100, self.n_classes_id)
        self.model.load_state_dict(torch.load(models_path + self.model_dataset_name,
                                              map_location=self.device_name))
        # elif "cifar" in self.id_dataset:
        #     self.model = torch.load(models_path + self.model_dataset_name,
        #                             map_location=self.device_name)

        self.linear_weights = self.model.fc.weight.cpu().detach().numpy()
        self.linear_bias = self.model.fc.bias.cpu().detach().numpy()
        '''
        self.model = torch_model.densenet121(pretrained=True)
        self.linear_weights = self.model.classifier.weight.cpu().detach().numpy()
        self.linear_bias = self.model.classifier.bias.cpu().detach().numpy()


class AdversarialAttack:
    def __init__(self, attack, model):
        self.attack_type = attack
        self.model = model
        self._load_attack()

    def run(self, images, labels):
        adv_images = self.attack(images, labels)

        return adv_images

    def _load_attack(self):
        if self.attack_type == "fgsm":
            self.attack = torchattacks.FGSM(self.model, eps=0.007)
        if self.attack_type == "deepfool":
            self.attack = torchattacks.DeepFool(self.model, steps=10, overshoot=0.02)
        if self.attack_type == "pgd":
            self.attack = torchattacks.PGD(self.model, eps=8/255, alpha=1/255, steps=40, random_start=True)
