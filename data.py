import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import yaml

N_CLASS = None
RESIZE_SHAPE = None
TRAIN_BATCH = None
VAL_BATCH = None
NUM_WORKER = None
PATH_ALL_IMAGES = None
PATH_INDICES_OF_TRAIN_DATA = None
PATH_LABELS_OF_TRAIN_DATA = None
PATH_INDICES_OF_VAL_DATA = None
PATH_LABELS_OF_VAL_DATA = None
PATH_TEST_IMAGES = None
PATH_LABELS_OF_TEST_DATA = None
transformWithAffine = None
transformWithoutAffine = None

class MyDataset(Dataset):
    def __init__(self, data, transform=None, **kwargs):
        self.data = data
        self.labels = kwargs["labels"] if "labels" in kwargs else None
        self.transform = transform
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform is not None: x = self.transform(x)
        if self.labels is not None: return x, self.labels[index]
        else: return x
    def __len__(self):
        return len(self.data)


class MyDataset_SA(Dataset):
    def __init__(self, data, transform=None, **kwargs):
        self.data = data
        self.labels = kwargs["labels"] if "labels" in kwargs else None
        self.transform = transform
    def __getitem__(self, index):
        x = self.data[index]
        x_list = [x]
        current_index = [index]
        while( len(current_index) < 3 ):
            idx = np.random.randint(len(self.data))
            if idx in current_index or self.labels[idx] != self.labels[index]:
                continue
            
            x_list.append(self.data[idx])
            current_index.append(idx)

        if self.transform != None:
            x_list = [self.transform(x) for x in x_list]

        if self.labels is not None: return x_list, self.labels[index]
        else: return x_list
            

        
    def __len__(self):
        return len(self.data)

class MyDataset_mixup(Dataset):
    def __init__(self, data, transform=None, **kwargs):
        self.data = data
        self.labels = kwargs["labels"] if "labels" in kwargs else None
        self.transform = transform
    def __getitem__(self, index_1):
        while 1:
            index_2 = np.random.randint(len(self.data))
            if index_2 != index_1:
                break
        x1 = self.data[index_1]
        x2 = self.data[index_2]
        if self.transform is not None: 
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        
        return (x1, self.labels[index_1]), (x2, self.labels[index_2])
        
    def __len__(self):
        return len(self.data)

def init():
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    dataset = config['hp']['dataset']
    # print(config)
    # Constants
    global N_CLASS
    global RESIZE_SHAPE
    global TRAIN_BATCH
    global VAL_BATCH
    global NUM_WORKER
    global PATH_ALL_IMAGES
    global PATH_INDICES_OF_TRAIN_DATA
    global PATH_LABELS_OF_TRAIN_DATA
    global PATH_INDICES_OF_VAL_DATA
    global PATH_LABELS_OF_VAL_DATA
    global PATH_TEST_IMAGES
    global PATH_LABELS_OF_TEST_DATA
    if dataset == 'mitbih':
        N_CLASS = 5
        RESIZE_SHAPE = (128, 128)
        TRAIN_BATCH = config['hp']['train_batch']
        VAL_BATCH = config['hp']['val_batch']
        NUM_WORKER = 2
        PATH_ALL_IMAGES            = 'MITBIH/train_data_2D_2000.npy'
        PATH_INDICES_OF_TRAIN_DATA  = "MITBIH/train_2000_10p_label_indices.npy"
        PATH_LABELS_OF_TRAIN_DATA  = "MITBIH/train_2000_10p_label_values.npy"
        PATH_INDICES_OF_VAL_DATA    = "MITBIH/val_2000_10p_label_indices.npy"
        PATH_LABELS_OF_VAL_DATA     = "MITBIH/val_2000_10p_label_values.npy"
        PATH_TEST_IMAGES            = 'MITBIH/test_data_2D_500.npy'
        PATH_LABELS_OF_TEST_DATA    = 'MITBIH/test_2D_500_label_values.npy'
    else:
        N_CLASS = 7
        RESIZE_SHAPE = (32, 32)
        TRAIN_BATCH = config['hp']['train_batch']
        VAL_BATCH = config['hp']['val_batch']
        NUM_WORKER = 2
        PATH_ALL_IMAGES            = "./WM811K/train_3150_data.npy"
        PATH_INDICES_OF_TRAIN_DATA  = "./WM811K/train_indices"
        PATH_LABELS_OF_TRAIN_DATA   = "./WM811K/train_labels"
        PATH_INDICES_OF_VAL_DATA    = "./WM811K/val_indices"
        PATH_LABELS_OF_VAL_DATA     = "./WM811K/val_labels"
        PATH_TEST_IMAGES            = "./WM811K/test_700_data.npy"
        PATH_LABELS_OF_TEST_DATA    = "./WM811K/test_700_label_values.npy"

    # All data
    global allImages; allImages = np.load(PATH_ALL_IMAGES, allow_pickle=True)
    global numOfAllData; numOfAllData = len(allImages)

    # --------------------------- FOR SUPERVISED MODEL --------------------------- #
    
    # Train data
    global indicesOfTrainData; indicesOfTrainData = np.load(PATH_INDICES_OF_TRAIN_DATA, allow_pickle = True)
    global labelsOfTrainData; labelsOfTrainData   = np.load(PATH_LABELS_OF_TRAIN_DATA, allow_pickle = True)
    global numOfTrainData; numOfTrainData         = len(indicesOfTrainData)

    # Val data
    global indicesOfValData; indicesOfValData = np.load(PATH_INDICES_OF_VAL_DATA, allow_pickle = True)
    global labelsOfValData; labelsOfValData   = np.load(PATH_LABELS_OF_VAL_DATA,  allow_pickle = True)
    global numOfValData; numOfValData         = len(indicesOfValData)
 
    # Labeled data = train + val
    global indicesOfLabeledData; indicesOfLabeledData = np.concatenate([indicesOfTrainData, indicesOfValData]) # 315
    global numOfLabeledData; numOfLabeledData         = len(indicesOfLabeledData);
    global labelsOfLabeledData; labelsOfLabeledData   = np.concatenate([labelsOfTrainData, labelsOfValData])

    # Unlabeled data = all - labeled
    mask = np.ones(numOfAllData, dtype=bool)
    mask[indicesOfLabeledData] = False
    global indicesOfUnabeledData; indicesOfUnabeledData = np.arange(numOfAllData)[mask]
    global numOfUnlabeledData; numOfUnlabeledData       = len(indicesOfUnabeledData)

    # Test data
    global testImages; testImages             = np.load(PATH_TEST_IMAGES,         allow_pickle=True)
    global labelsOfTestData; labelsOfTestData = np.load(PATH_LABELS_OF_TEST_DATA, allow_pickle=True)
    global numOfTestData; numOfTestData       = len(testImages)

    global transformWithAffine; transformWithAffine = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_SHAPE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.Normalize((0.0/255), (2.0/255))
    ])

    global transformWithoutAffine; transformWithoutAffine = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_SHAPE),
    transforms.Normalize((0.0/255), (2.0/255))
    ])

    # Dataset
    # global trainDatasetForSupervisedModel; trainDatasetForSupervisedModel     = MyDataset(allImages,  transform=transformWithAffine,    labels=labelsOfTrainData, indicesOfData=indicesOfTrainData)
    # global valDatasetForSupervisedModel;   valDatasetForSupervisedModel       = MyDataset(allImages,  transform=transformWithoutAffine, labels=labelsOfValData,   indicesOfData=indicesOfValData)
    global trainDatasetForSupervisedModel; trainDatasetForSupervisedModel             = MyDataset(allImages[indicesOfTrainData],  transform=transformWithAffine,    labels=labelsOfTrainData)
    global trainDatasetForSupervisedModel_SA; trainDatasetForSupervisedModel_SA       = MyDataset_SA(allImages[indicesOfTrainData],  transform=transformWithAffine,    labels=labelsOfTrainData)
    global trainDatasetForSupervisedModel_mixup; trainDatasetForSupervisedModel_mixup = MyDataset_mixup(allImages[indicesOfTrainData],  transform=transformWithAffine,    labels=labelsOfTrainData)
    global valDatasetForSupervisedModel;   valDatasetForSupervisedModel               = MyDataset(allImages[indicesOfValData],  transform=transformWithoutAffine, labels=labelsOfValData)
    global testDatasetForSupervisedModel;  testDatasetForSupervisedModel              = MyDataset(testImages, transform=transformWithoutAffine, labels=labelsOfTestData)

    # Dataloader
    global trainDataLoaderForSupervisedModel; trainDataLoaderForSupervisedModel                 = DataLoader(trainDatasetForSupervisedModel, batch_size=TRAIN_BATCH, shuffle=True,  num_workers=NUM_WORKER)
    global trainDataLoaderForSupervisedModel_SA; trainDataLoaderForSupervisedModel_SA           = DataLoader(trainDatasetForSupervisedModel_SA, batch_size=TRAIN_BATCH, shuffle=True,  num_workers=NUM_WORKER)
    global trainDataLoaderForSupervisedModel_mixup; trainDataLoaderForSupervisedModel_mixup     = DataLoader(trainDatasetForSupervisedModel_mixup, batch_size=TRAIN_BATCH, shuffle=True,  num_workers=NUM_WORKER)
    global valDataLoaderForSupervisedModel;   valDataLoaderForSupervisedModel                   = DataLoader(valDatasetForSupervisedModel,   batch_size=VAL_BATCH,   shuffle=False, num_workers=NUM_WORKER)
    global testDataLoaderForSupervisedModel;  testDataLoaderForSupervisedModel                  = DataLoader(testDatasetForSupervisedModel,  batch_size=VAL_BATCH,   shuffle=False, num_workers=NUM_WORKER)

    # ---------------------------- FOR MAIN CLASSIFIER --------------------------- #

    # Dataset and dataloader
    global representationVectorsForTrain   ; representationVectorsForTrain    = None
    global trainDatasetForMainClassifier   ; trainDatasetForMainClassifier    = None
    global trainDataLoaderForMainClassifier; trainDataLoaderForMainClassifier = None
    global valDatasetForMainClassifier     ; valDatasetForMainClassifier      = None
    global valDataLoaderForMainClassifier  ; valDataLoaderForMainClassifier   = None
    global representationVectorsForTest    ; representationVectorsForTest     = None
    global testDatasetForMainClassifier    ; testDatasetForMainClassifier     = None
    global testDataLoaderForMainClassifier ; testDataLoaderForMainClassifier  = None
    global unlabeledData      ; unlabeledData       = None
    global unlabeledDataset   ; unlabeledDataset    = None
    global unlabeledDataLoader; unlabeledDataLoader = None

init()

def summary():
    print(f'''[Data Hierarchy]
- All data:           {numOfAllData}
    - Labeled data:   {numOfLabeledData}
        - Train data: {numOfTrainData}
        - Val data:   {numOfValData}
    - Unlabeled data: {numOfUnlabeledData}
- Test data:          {numOfTestData}

[Dataset]
- len(trainDatasetForSupervisedModel) = {len(trainDatasetForSupervisedModel)}
- len(valDatasetForSupervisedModel)   = {len(valDatasetForSupervisedModel)}
- len(testDatasetForSupervisedModel)  = {len(testDatasetForSupervisedModel)}
- len(trainDatasetForMainClassifier)  = {len(trainDatasetForMainClassifier)}
- len(valDatasetForMainClassifier)    = {len(valDatasetForMainClassifier)}
- len(testDatasetForMainClassifier)   = {len(testDatasetForMainClassifier)}
- len(unlabeledDataset)               = {len(unlabeledDataset)}

[Dataloader]
- len(trainDataLoaderForSupervisedModel)   = {len(trainDataLoaderForSupervisedModel)}
- len(trainDataLoaderForSupervisedModel_SA)   = {len(trainDataLoaderForSupervisedModel_SA)}
- len(valDataLoaderForSupervisedModel)  = {len(valDataLoaderForSupervisedModel)}
- len(testDataLoaderForSupervisedModel) = {len(testDataLoaderForSupervisedModel)}
- len(trainDataLoaderForMainClassifier) = {len(trainDataLoaderForMainClassifier)}
- len(valDataLoaderForMainClassifier)   = {len(valDataLoaderForMainClassifier)}
- len(testDataLoaderForMainClassifier)  = {len(testDataLoaderForMainClassifier)}
- len(unlabeledDataLoader)              = {len(unlabeledDataLoader)}
''')

    print("[Consistenc check]")
    print('All data', end=":\t")
    if numOfAllData==len(allImages): print('✅')
    else: print('❌')
    print('Labeled data', end=":\t")
    if numOfLabeledData==len(indicesOfLabeledData) and numOfLabeledData==len(labelsOfLabeledData): print('✅')
    else: print('❌')
    print('Train data', end=":\t")
    if numOfTrainData==len(indicesOfTrainData) \
        and numOfTrainData==len(labelsOfTrainData)\
        and numOfTrainData==len(trainDatasetForSupervisedModel): print('✅')
    else: print('❌')
    print('Val data', end=":\t")
    if numOfValData==len(indicesOfValData) and numOfValData==len(labelsOfValData): print('✅')
    else: print('❌')
    print('Unlabeled data', end=":\t")
    if numOfUnlabeledData==len(indicesOfUnabeledData): print('✅')
    else: print('❌')
    print('Test data', end=":\t")
    if numOfTestData == len(labelsOfTestData): print('✅')
    else: print('❌')

if __name__ == "__main__":
    summary()
