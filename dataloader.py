import numpy as np
import random
import os
import tensorflow as tf

MobileNet_path = './MobileNet/'
ResNet_path = './ResNet/'
CLIP_path = './Clip/'
ViT_path = './ViT/'
swin_ViT_path = './swin-ViT/'


def dataLoader(setIndex, featureType='ResNet'):
    folder_name = ['STL-10', 'CIFAR-10', 'MIT-Places-Small', 'MNIST', 'Fashion-MNIST' \
                   , 'CatsVsDogs', 'Fake-STL10', 'CIFAR-100', 'Caltech101', 'Cub_200_2011', 'Paris']
    dataset_name = folder_name[setIndex]
    
    if setIndex == 0:
        if  featureType == 'ResNet':
            print('feature type: ', 'ResNet')
            path = os.path.join(ResNet_path,'STL-10/')
            all_feats = np.load(path+'resNet50.npy')
            all_labels = np.load(path+'gt.npy')
        elif featureType == 'ViT':
            print('feature type: ', 'ViT')
            path = os.path.join(ViT_path,'stl10_vit/')
            all_feats = np.load(path+'vit_feats.npy')
            all_labels = np.load(path+'vit_gt.npy')
        elif featureType == 'swin-ViT':
            print('feature type: ', 'Swin-ViT')
            path = os.path.join(swin_ViT_path)
            all_feats = np.load(path+'swin_large_224_STL-10.npy')
            all_labels = np.load(path+'swin_gt_STL-10.npy')
            
        elif featureType == 'Clip':
            print('feature type: ', 'clip')
            path = os.path.join(CLIP_path,'STL-10/')
            all_feats = np.load(path+'STL_clip_feat.npy')
            all_labels = np.load(path+'STL_clip_gt.npy')
            
    elif setIndex == 1:
        if  featureType == 'ResNet':
            print('feature type: ', 'ResNet')
            path = os.path.join(ResNet_path, 'cifar10/')
            train_data = np.load(path+'cifar10_train/resNet50.npy')
            train_gt = np.load(path+'cifar10_train/gt.npy')
            test_data = np.load(path+'cifar10_test/resNet50.npy')
            test_gt = np.load(path+'cifar10_test/gt.npy')

            all_feats = np.concatenate((train_data, test_data), axis = 0)
            all_labels = np.concatenate((train_gt, test_gt), axis = 0)
            
        elif featureType == 'ViT':
            path = os.path.join(ViT_path, 'cifar-10-vit/')
            train_data = np.load(path+'Training_feats.npy')
            train_gt = np.load(path+'Training_gt.npy')
            test_data = np.load(path+'Test_feats.npy')
            test_gt = np.load(path+'Test_gt.npy')

            all_feats = np.concatenate((train_data, test_data), axis = 0)
            all_labels = np.concatenate((train_gt, test_gt), axis = 0)
    
        elif featureType == 'Clip':
            path = os.path.join(CLIP_path, 'CIFAR/')
            train_data = np.load(path+'cifar_train_clip_feat.npy')
            train_gt = np.load(path+'cifar_train_clip_gt.npy')
            test_data = np.load(path+'cifar_test_clip_feat.npy')
            test_gt = np.load(path+'cifar_test_clip_gt.npy')

            all_feats = np.concatenate((train_data, test_data), axis = 0)
            all_labels = np.concatenate((train_gt, test_gt), axis = 0)
    
    elif setIndex == 2:
        if  featureType == 'ResNet':
            print('feature type: ', 'ResNet')
            path = os.path.join(ResNet_path,'MIT-Places-Small/')
            all_feats = np.load(path+'resNet50.npy')
            all_labels = np.load(path+'gt.npy')
        elif featureType == 'ViT':
            path = os.path.join(ViT_path,'MIT-Places-Small/')
            all_feats = np.load(path+'vit_feats.npy')
            all_labels = np.load(path+'vit_gt.npy')
        elif featureType == 'Clip':
            path = os.path.join(CLIP_path,'MIT-Places-Small/')
            all_feats = np.load(path+'MIT_clip_feat.npy')
            all_labels = np.load(path+'MIT_clip_gt.npy')
            
    elif setIndex == 3:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        trainFeat = x_train.reshape(x_train.shape[0], 28*28)
        testFeat = x_test.reshape(x_test.shape[0], 28*28)
        testGt = y_test
        trainGt = y_train

        trainFeat= trainFeat-255/2
        testFeat= testFeat-255/2

        all_feats = np.concatenate((trainFeat, testFeat), axis = 0)
        all_labels = np.concatenate((trainGt, testGt), axis = 0)

    elif setIndex == 4:
        mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        trainFeat = x_train.reshape(x_train.shape[0], 28*28)
        testFeat = x_test.reshape(x_test.shape[0], 28*28)
        testGt = y_test
        trainGt = y_train

        trainFeat= trainFeat-255/2
        testFeat= testFeat-255/2

        all_feats = np.concatenate((trainFeat, testFeat), axis = 0)
        all_labels = np.concatenate((trainGt, testGt), axis = 0)

    elif setIndex == 5:
        if  featureType == 'ResNet':
            path = os.path.join(ResNet_path,'cats_vs_dogs224feats/')
            trainFeat = np.load(path+'feats_train.npy')
            testFeat = np.load(path+'feats_test.npy')
            trainGt = np.load(path+'y_train.npy')
            testGt = np.load(path+'y_test.npy')

            all_feats = np.concatenate((trainFeat, testFeat), axis = 0)
            all_labels = np.concatenate((trainGt, testGt), axis = 0)
        elif featureType == 'Clip':
            path = os.path.join(CLIP_path,'CatsVSDogs/')
            all_feats = np.load(path+'catsvsdogs_clip_feat.npy')
            all_labels = np.load(path+'catsvsdogs_clip_gt.npy')
                  
    elif setIndex == 6:
        if  featureType == 'ResNet':
            print('feature type: ', 'ResNet')
            path = os.path.join(ResNet_path,'fake-stl10/')
            all_feats = np.load(path+'resNet50.npy')
            all_labels = np.load(path+'gt.npy')
        
        elif featureType == 'Clip':
            print('feature type: ', 'clip')
            path = os.path.join(CLIP_path,'Fake_stl10/')
            all_feats = np.load(path+'fake_STL_clip_feat.npy')
            all_labels = np.load(path+'fake_STL_clip_gt.npy')
            
        elif featureType == 'MobileNet':
            print('feature type: ', 'clip')
            path = os.path.join(MobileNet_path,'fake_stl10/')
            all_feats = np.load(path+'fake-stl10_mobilenetv2_feat.npy')
            all_labels = np.load(path+'fake-stl10_mobilenetv2_gt.npy')

    elif setIndex == 7:
        if  featureType == 'ResNet':
            print('feature type: ', 'ResNet')
            path =os.path.join(ResNet_path,'cifar100/')
            all_feats = np.load(path+'cifar100Feats.npy')
            all_labels = np.load(path+'cifar100Gt.npy')
            
        elif featureType == 'Clip':
            print('feature type: ', 'clip')
            path = os.path.join(CLIP_path,'Cifar100/')
            all_feats = np.load(path+'cifar100_train_feat.npy')
            all_labels = np.load(path+'cifar100_train_gt.npy')
    
    elif setIndex == 8:
        if  featureType == 'ResNet':
            print('feature type: ', 'ResNet')
            path = os.path.join(ResNet_path,'fake-stl10/')
            all_feats = np.load(path+'resNet50.npy')
            all_labels = np.load(path+'gt.npy')
        
        elif featureType == 'Clip':
            print('feature type: ', 'clip')
            path = os.path.join(CLIP_path,'Caltech101/')
            all_feats = np.load(path+'caltech101_clip_feat.npy')
            all_labels = np.load(path+'caltech101_clip_gt.npy')
            
        elif featureType == 'MobileNet':
            print('feature type: ', 'MobileNet')
            path = os.path.join(MobileNet_path,'Caltech101/')
            all_feats = np.load(path+'caltech101_mobilenetv2_feat.npy')
            all_labels = np.load(path+'caltech101_mobilenetv2_gt.npy')
            
            
    elif setIndex == 9:
        if  featureType == 'ResNet':
            print('feature type: ', 'ResNet')
            path = os.path.join(ResNet_path,'fake-stl10/')
            all_feats = np.load(path+'resNet50.npy')
            all_labels = np.load(path+'gt.npy')
        
        elif featureType == 'Clip':
            print('feature type: ', 'clip')
            path = os.path.join(CLIP_path,'Cub_200_2011/')
            all_feats = np.load(path+'cub_200_2011_clip_feat.npy')
            all_labels = np.load(path+'cub_200_2011_clip_gt.npy')
            

    elif setIndex == 10:
        if  featureType == 'ResNet':
            print('feature type: ', 'ResNet')
            path = os.path.join(ResNet_path,'fake-stl10/')
            all_feats = np.load(path+'resNet50.npy')
            all_labels = np.load(path+'gt.npy')
        
        elif featureType == 'Clip':
            print('feature type: ', 'clip')
            path = os.path.join(CLIP_path,'Paris/')
            all_feats = np.load(path+'paris_clip_feat.npy')
            all_labels = np.load(path+'paris_clip_gt.npy')
            
    return all_feats, all_labels, dataset_name


def _load_data_with_outliers(all_feats, all_labels, ind, p):
    np.random.seed(42)
    normal = all_feats[all_labels == ind]
    
    # simple test
    if normal.shape[0]>1000:
        normal = normal[:2000]
    
    abnormal = all_feats[all_labels != ind]
    # num_abnormal = int(normal.shape[0]*p/(1-p))
    num_abnormal = int(normal.shape[0]*p/(1-p))
    if  num_abnormal<1:
        num_abnormal = 1
    selected = np.random.choice(abnormal.shape[0], num_abnormal, replace=False)
    data = np.concatenate((normal, abnormal[selected]), axis=0)
    labels = np.ones((data.shape[0], ), dtype=np.int32)
    labels[:len(normal)] = 0
    return data, labels