import numpy as np
from termcolor import cprint
from tqdm import tqdm
import torch # GPU
from datetime import datetime
import pytz
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from shutil import copyfile
from models import Wide_ResNet, VAE, SimpleNN3 # Models
from resnet_vae import VAE_mitbih
from torch.utils.data import DataLoader
import statistics
from pathlib import Path
from ipdb import set_trace
from ranger import Ranger
import yaml
import data as limitedData # Data
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def oneTrainSupervisedModel():
    global supervisedModel
    global criterionForSupervisedModel
    global optimizerForSupervisedModel
    global statistics
    statistics.reset(mode="train")
    supervisedModel.train()
    for batch_idx, (x, y) in enumerate(limitedData.trainDataLoaderForSupervisedModel):
        if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
        # Forward pass
        y_hat, _ = supervisedModel(x)
        loss = criterionForSupervisedModel(y_hat, y.long())
        if accumulate_gradient:
            loss_accum = criterionForSupervisedModel(y_hat, y.long()) / accumulate_iter
            loss_accum.backward()
            if (batch_idx + 1) % accumulate_iter == 0 or batch_idx + 1 == len(limitedData.trainDataLoaderForSupervisedModel):
                optimizerForSupervisedModel.step()
                optimizerForSupervisedModel.zero_grad()
        else:
            loss = criterionForSupervisedModel(y_hat, y.long())
            # Backward pass
            loss.backward()
            optimizerForSupervisedModel.step()
            optimizerForSupervisedModel.zero_grad()
        # Statistics
        _, onehot = y_hat.max(1)
        statistics.numTotal += len(y)
        statistics.numCorrect += onehot.eq(y).sum().item()
        statistics.trainLoss += loss.item()
    statistics.trainAcc = statistics.numCorrect / statistics.numTotal
    statistics.trainLoss /= statistics.numTotal

def oneValSupervisedModel():
    global supervisedModel
    global criterionForSupervisedModel
    global statistics
    statistics.reset(mode="val")
    supervisedModel.eval()
    with torch.no_grad():
        for x, y in limitedData.valDataLoaderForSupervisedModel:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat, _ = supervisedModel(x)
            loss = criterionForSupervisedModel(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.valLoss += loss.item()
        statistics.valAcc = statistics.numCorrect / statistics.numTotal
        statistics.valLoss /= statistics.numTotal

def initialoneValSupervisedModel():
    global supervisedModel
    global criterionForSupervisedModel
    global statistics
    # statistics.reset(mode="val")
    supervisedModel.eval()
    with torch.no_grad():
        for x, y in limitedData.valDataLoaderForSupervisedModel:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat, _ = supervisedModel(x)
            loss = criterionForSupervisedModel(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.valLoss += loss.item()
        statistics.valAcc = statistics.numCorrect / statistics.numTotal
        statistics.valLoss /= statistics.numTotal
    # print('statistics.valAcc =', statistics.valAcc)
    # statistics.localBestValAcc = statistics.valAcc
    # statistics.globalBestValAcc = statistics.valAcc

def oneTestSupervisedModel(): 
    global supervisedModel
    global criterionForSupervisedModel
    global statistics
    statistics.reset(mode="test")
    if not statistics.improved: 
        saveModel("tempSupervisedModel")
        loadModel("supervisedModel")
    supervisedModel.eval()
    with torch.no_grad():
        for x, y in limitedData.testDataLoaderForSupervisedModel:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat, _ = supervisedModel(x)
            loss = criterionForSupervisedModel(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.testLoss += loss.item()
        statistics.testAcc = statistics.numCorrect / statistics.numTotal
        statistics.testLoss /= statistics.numTotal
    if not statistics.improved: loadModel("tempSupervisedModel")

def initialTestsupervisedModel():
    global supervisedModel
    global criterionForSupervisedModel
    global statistics
    statistics.reset(mode="test")
    # loadModel("globalSupervisedModel")
    supervisedModel.eval()
    with torch.no_grad():
        for x, y in limitedData.testDataLoaderForSupervisedModel:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat, _ = supervisedModel(x)
            loss = criterionForSupervisedModel(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.valLoss += loss.item()
        statistics.testAcc = statistics.numCorrect / statistics.numTotal
        statistics.testLoss /= statistics.numTotal
    
    print('\n---------------------------------- Summary of pretrained model ---------------------------------')
    print(f'BestVal  [{statistics.valAcc:.3%}]')
    print(f'TestAcc  [{statistics.testAcc:.3%}]')
    print(f'TestLoss [{statistics.testLoss:.6f}]')

def trainSupervisedModel():
    global statistics
    statistics.initRound()
    for epoch in range(NUM_EPOCH):
        oneTrainSupervisedModel()
        oneValSupervisedModel()
        if improved(model="supervisedModel", mode='local'): 
            saveModel("supervisedModel")
        oneTestSupervisedModel()
        if shouldEarlyStop("supervisedModel"): break
        summaryModel(epoch+1, "supervisedModel")
        if LOG: log("supervisedModel")
        if OPTIM == 'SGD':
            scheduler_s.step()
            print(f'Scheduler step')

def saveModel(model):
    if model == 'supervisedModel' or model == "tempSupervisedModel" or model == "globalSupervisedModel":
        global supervisedModel
        global optimizerForSupervisedModel
        checkpoint = {
            'state_dict' : supervisedModel.state_dict(), 
            'optimizer' : optimizerForSupervisedModel.state_dict(),
            }
        torch.save(checkpoint, f'{MODEL_SAVE_PATH}/{model}.pth')
        return
    elif model == 'mainClassifier' or model == "tempMainClassifier" or model == "globalMainClassifier":
        global mainClassifier
        global optimizerForMainClassifier
        checkpoint = {
            'state_dict' : mainClassifier.state_dict(), 
            'optimizer' : optimizerForMainClassifier.state_dict(),
            }
        torch.save(checkpoint, f'{MODEL_SAVE_PATH}/{model}.pth')
        return
    else:
        print("Warning: Undefined model")

def loadModel(model):
    if model == "supervisedModel" or model == "tempSupervisedModel" or model == "globalSupervisedModel":
        global supervisedModel
        global optimizerForSupervisedModel
        checkpoint = torch.load(f'{MODEL_SAVE_PATH}/{model}.pth')
        supervisedModel.load_state_dict(checkpoint['state_dict'])
        optimizerForSupervisedModel.load_state_dict(checkpoint['optimizer'])
    elif model == "mainClassifier" or model == "tempMainClassifier" or model == "globalMainClassifier":
        global mainClassifier
        global optimizerForMainClassifier
        checkpoint = torch.load(f'{MODEL_SAVE_PATH}/{model}.pth')
        mainClassifier.load_state_dict(checkpoint['state_dict'])
        optimizerForMainClassifier.load_state_dict(checkpoint['optimizer'])
    else:
        print("Warning: Undefined model")

def summaryModel(epoch, model):
    cprint(f'Epoch [{epoch}/{NUM_EPOCH}]', end=' ')
    cprint(f'Train [{statistics.trainAcc:.3%}]', 'yellow', end=' ')
    cprint(f'Val [{statistics.valAcc:.3%}]', 'magenta', end=' ')
    cprint(f'BestVal [{statistics.localBestValAcc:.3%}]', 'green', end=' ')
    cprint(f'Test [{statistics.testAcc:.3%}]', 'cyan', end=' ')
    if model == "supervisedModel":
        cprint(f'ESC [{statistics.earlyStopCountForSupervisedModel}/{MAX_ESC}]', end=' ')
    elif model == "mainClassifier":
        cprint(f'ESC [{statistics.earlyStopCountForMainClassifier}/{MAX_ESC}]', end=' ')
    else:
        print("Warning: Undefined model")
    cprint('+', 'green') if statistics.improved else cprint('-', 'red')

def improved(model, mode):
    global statistics
    if mode == 'local':
        if statistics.valAcc >= statistics.localBestValAcc:
            statistics.improved = True
            statistics.localBestValAcc = statistics.valAcc
            if model == 'supervisedModel':
                statistics.earlyStopCountForSupervisedModel = 0
            elif model == 'mainClassifier':
                statistics.earlyStopCountForMainClassifier = 0
            else:
                print("Warning: Undefined model")
            return True
        else:
            statistics.improved = False
            if model == 'supervisedModel':
                statistics.earlyStopCountForSupervisedModel += 1
            elif model == 'mainClassifier':
                statistics.earlyStopCountForMainClassifier += 1
            else:
                print("Warning: Undefined model")
            return False
    elif mode == 'global':
        if statistics.valAcc >= statistics.globalBestValAcc:
            statistics.globalBestValAcc = statistics.valAcc
            return True
        else:
            return False
    elif mode == 'round':
        if statistics.localBestValAcc >= statistics.globalBestValAcc  :
            print('ESC_ROUND return to 0')
            statistics.round_improved = True
            statistics.earlyStopCountForTrainingRound = 0
            return True
        else:
            print('ESC_ROUND +1 ')
            statistics.round_improved = False
            statistics.earlyStopCountForTrainingRound += 1
            return False
    else:
        print("Warning: Undefined mode")
        
def shouldEarlyStop(model):
    global statistics
    if model == 'supervisedModel':
        if statistics.earlyStopCountForSupervisedModel > MAX_ESC: return True
    elif model == 'mainClassifier':
        if statistics.earlyStopCountForMainClassifier > MAX_ESC: return True
    elif model =='round':
        if statistics.earlyStopCountForTrainingRound >= ESC_ROUND: return True
    else:
        print("Warning: Undefined model")
        return True
    return False

def initExperiment(config):
    # Initialize limitedData_loader
    limitedData.init(config)
    
    global EXPERIMENT_NAME
    global ACC_LOSS_SAVE_PATH
    global MODEL_SAVE_PATH
    global supervisedModel
    global criterionForSupervisedModel
    global optimizerForSupervisedModel
    global unsupervisedModel
    global mainClassifier
    global criterionForMainClassifier
    global optimizerForMainClassifier
    global LOG                
    global LR
    global WEIGHT_DECAY
    global MOMENTUM
    global scheduler_s
    global scheduler_m
    global OPTIM 
    global PL_RATE                 
    global NUM_PL             
    global NUM_EPOCH 
    global MAX_ESC     
    global NUM_ROUND 
    global ESC_ROUND         
    global accumulate_gradient
    global train_batch
    global train_batch_after_accumulate
    global accumulate_iter      
    global device
    global pretrained
    global TASK
       
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOG = config['hp']['log']
    LR = config['hp']['lr']
    PL_RATE  = config['hp']['pl_rate']
    NUM_PL = int(limitedData.numOfLabeledData * PL_RATE)
    NUM_EPOCH = config['hp']['num_epoch']
    # print(NUM_EPOCH)
    MAX_ESC = config['hp']['max_esc']
    NUM_ROUND = config['hp']['num_round']
    ESC_ROUND = config['hp']['esc_round']
    accumulate_gradient = config['hp']['accumulate_gradient']
    train_batch = config['hp']['train_batch']
    train_batch_after_accumulate = config['hp']['train_batch_after_accumulate']
    accumulate_iter = int(train_batch_after_accumulate / train_batch)
    OPTIM = config['hp']['optimizer']
    TASK = config['hp']['task']

    tpe = pytz.timezone('Asia/Taipei')
    EXPERIMENT_NAME = datetime.now(tpe).strftime("%Y-%m-%d %H:%M:%S")
    pretrained = config['hp']['pretrained']

    print(limitedData.N_CLASS)
    print(limitedData.RESIZE_SHAPE)
    print(device)
    supervisedModel = Wide_ResNet(28, 10, 0.2, limitedData.N_CLASS, data_shape=limitedData.RESIZE_SHAPE)
    supervisedModel = torch.nn.DataParallel(supervisedModel)
    if pretrained:
        print('load models')
        checkpoint = torch.load(f'./pretrained_models/{config["hp"]["dataset"]}/S/{OPTIM}/globalSupervisedModel.pth')
        #print(checkpoint['state_dict'])
        try: 
            supervisedModel.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(e)
    supervisedModel.to(device)

    mainClassifier = SimpleNN3(n_class=limitedData.N_CLASS, data_shape=limitedData.RESIZE_SHAPE)
    mainClassifier.to(device)

    torch.backends.cudnn.benchmark = True
    criterionForSupervisedModel = torch.nn.CrossEntropyLoss()
    criterionForMainClassifier = torch.nn.CrossEntropyLoss()
    if OPTIM == 'Ranger':
        optimizerForSupervisedModel = Ranger(supervisedModel.parameters(), lr=LR, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5, weight_decay=0, use_gc=True, gc_conv_only=False)
        optimizerForMainClassifier = Ranger(mainClassifier.parameters(), lr=LR, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5, weight_decay=0, use_gc=True, gc_conv_only=False)
    elif OPTIM == 'Adam':
        optimizerForSupervisedModel = torch.optim.Adam(supervisedModel.parameters(), lr=LR)
        optimizerForMainClassifier = torch.optim.Adam(mainClassifier.parameters(), lr=LR)
    elif OPTIM == 'SGD':
        optimizerForSupervisedModel = torch.optim.SGD(supervisedModel.parameters(), lr=LR,momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        optimizerForMainClassifier = torch.optim.SGD(mainClassifier.parameters(), lr=LR,momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerForSupervisedModel, T_max=NUM_EPOCH)
        scheduler_m = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerForMainClassifier, T_max=NUM_EPOCH)

    if config['hp']['dataset'] == 'mitbih':
        print(f'VAE_mitbih:{VAE_mitbih}')
        unsupervisedModel = VAE_mitbih(640)
        unsupervisedModel.to(device)
        unsupervisedModel.load_state_dict(torch.load('./unsupervised_model_mitbih.pt'))
    else:
        # print('yolo')
        unsupervisedModel = VAE()
        unsupervisedModel.to(device)
        unsupervisedModel.load_state_dict(torch.load('./best_u_model_VAE_CNN_AUG_640.pt'))
    ACC_LOSS_SAVE_PATH = f'./records/{config["hp"]["dataset"]}/{TASK}/{OPTIM}/{EXPERIMENT_NAME}'
    MODEL_SAVE_PATH = f'./checkpoints/{config["hp"]["dataset"]}/{TASK}/{OPTIM}/{EXPERIMENT_NAME}'
    Path(ACC_LOSS_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    # copyfile(src='./config.yaml', dst=f'{ACC_LOSS_SAVE_PATH}/config.yaml')
    # with open(f'{ACC_LOSS_SAVE_PATH}/config.yaml', 'w') as outfile:
    #     yaml.dump(config, outfile)
    

def log(model):
    global statistics
    global ACC_LOSS_SAVE_PATH

    if model == "supervisedModel":
        with open(f'{ACC_LOSS_SAVE_PATH}/sAcc', 'a+') as f:
            f.write(str(statistics.trainAcc) + ',')
            f.write(str(statistics.valAcc  ) + ',')
            f.write(str(statistics.testAcc ) + '\n')
        with open(f'{ACC_LOSS_SAVE_PATH}/sLoss', 'a+') as f:
            f.write(str(statistics.trainLoss) + ',')
            f.write(str(statistics.valLoss  ) + ',')
            f.write(str(statistics.testLoss ) + '\n')
    elif model == "mainClassifier":
        with open(f'{ACC_LOSS_SAVE_PATH}/mAcc', 'a+') as f:
            f.write(str(statistics.trainAcc) + ',')
            f.write(str(statistics.valAcc  ) + ',')
            f.write(str(statistics.testAcc ) + '\n')
        with open(f'{ACC_LOSS_SAVE_PATH}/mLoss', 'a+') as f:
            f.write(str(statistics.trainLoss) + ',')
            f.write(str(statistics.valLoss  ) + ',')
            f.write(str(statistics.testLoss ) + '\n')
    else:
        print("Warning: Undefined model")

def oneTrainMainClassifier():
    global mainClassifier
    global criterionForMainClassifier
    global optimizerForMainClassifier
    global statistics
    statistics.reset(mode="train")
    mainClassifier.train()
    #set_trace()
    for batch_idx, (x, y) in enumerate(limitedData.trainDataLoaderForMainClassifier):
      
        if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
     
        # Forward pass
        # set_trace()
        # print("x.shape", x.shape)
        # print("y.shape", y.shape)
        y_hat = mainClassifier(x)
        if accumulate_gradient:
            loss = criterionForMainClassifier(y_hat, y.long()) / accumulate_iter
            loss.backward()
            if (batch_idx + 1) % accumulate_iter == 0 or batch_idx + 1 == len(limitedData.trainDataLoaderForMainClassifier):
                optimizerForMainClassifier.step()
                optimizerForMainClassifier.zero_grad()
        else:
            loss = criterionForMainClassifier(y_hat, y.long())
            loss.backward()
            optimizerForMainClassifier.step()
            optimizerForMainClassifier.zero_grad()
        
        # Statistics
        _, onehot = y_hat.max(1)
        statistics.numTotal += len(y)
        statistics.numCorrect += onehot.eq(y).sum().item()
        statistics.trainLoss += loss.item()
    #set_trace()
    statistics.trainAcc = statistics.numCorrect / statistics.numTotal
    statistics.trainLoss /= statistics.numTotal
    # statistics.summary()

def oneValMainClassifier():
    global mainClassifier
    global criterionForMainClassifier
    global statistics
    statistics.reset(mode="val")
    mainClassifier.eval()
    with torch.no_grad():
        for x, y in limitedData.valDataLoaderForMainClassifier:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat = mainClassifier(x)
            loss = criterionForMainClassifier(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.valLoss += loss.item()
        statistics.valAcc = statistics.numCorrect / statistics.numTotal
        statistics.valLoss /= statistics.numTotal

def oneTestMainClassifier():
    global mainClassifier
    global criterionForMainClassifier
    global statistics
    statistics.reset(mode="test")
    if not statistics.improved: 
        saveModel("tempMainClassifier")
        loadModel("mainClassifier")
    mainClassifier.eval()
    with torch.no_grad():
        for x, y in limitedData.testDataLoaderForMainClassifier:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat = mainClassifier(x)
            loss = criterionForMainClassifier(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.testLoss += loss.item()
        statistics.testAcc = statistics.numCorrect / statistics.numTotal
        statistics.testLoss /= statistics.numTotal
    if not statistics.improved: loadModel("tempMainClassifier")

def trainMainClassifier():
    global statistics
    statistics.initRound()
    for epoch in range(NUM_EPOCH):
        oneTrainMainClassifier()
        oneValMainClassifier()
        if improved(model="mainClassifier", mode='local'): 
            saveModel('mainClassifier')
        if improved(model="mainClassifier", mode='global'): 
            saveModel("globalSupervisedModel")
            saveModel('globalMainClassifier')
        oneTestMainClassifier()
        if shouldEarlyStop("mainClassifier"): break
        summaryModel(epoch+1, "mainClassifier")
        if LOG: log("mainClassifier")
        if OPTIM == 'SGD':
            scheduler_m.step()
            print(f'Scheduler step')


def finalTestMainClassifier():
    global supervisedModel
    global mainClassifier
    global criterionForMainClassifier
    global statistics
    statistics.reset(mode="test")
    #loadModel("globalSupervisedModel")
    loadModel("globalMainClassifier")
    supervisedModel.eval()
    mainClassifier.eval()
    configDataForMainClassifier_final()
    with torch.no_grad():
        for x, y in limitedData.testDataLoaderForMainClassifier:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat = mainClassifier(x)
            loss = criterionForMainClassifier(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.testLoss += loss.item()
        statistics.testAcc = statistics.numCorrect / statistics.numTotal
        statistics.testLoss /= statistics.numTotal
    
    print('\n---------------------------------- Summary ---------------------------------')
    print(f'TestAcc  [{statistics.testAcc:.3%}]')
    print(f'TestLoss [{statistics.testLoss:.6f}]')

def buildRepresentationVectors(mode):
    global supervisedModel
    global unsupervisedModel
    loadModel("supervisedModel")
    if mode=='train':
        imagesDataset = limitedData.MyDataset(data=limitedData.allImages, transform=limitedData.transformWithoutAffine) 
    else:
        imagesDataset = limitedData.MyDataset(data=limitedData.testImages, transform=limitedData.transformWithoutAffine)
    representationVectors = []
    for x in tqdm(imagesDataset):
        x = x.to(device)
        x = torch.unsqueeze(x, 0)
        s_vec = supervisedModel(x)[1].detach().flatten().cpu().numpy()
        u_vec = unsupervisedModel(x)[1].detach().flatten().cpu().numpy()
        # print(f'u_vec:{u_vec.shape}')
        vec = np.concatenate([s_vec, u_vec])
        representationVectors.append(vec)
    return np.array(representationVectors)

def buildRepresentationVectors_final(mode):
    global supervisedModel
    global unsupervisedModel
    #loadModel("supervisedModel")
    loadModel("globalSupervisedModel")

    if mode=='train':
        imagesDataset = limitedData.MyDataset(data=limitedData.allImages, transform=limitedData.transformWithoutAffine) 
    else:
        imagesDataset = limitedData.MyDataset(data=limitedData.testImages, transform=limitedData.transformWithoutAffine)
    representationVectors = []
    for x in tqdm(imagesDataset):
        x = x.to(device)
        x = torch.unsqueeze(x, 0)
        s_vec = supervisedModel(x)[1].detach().flatten().cpu().numpy()
        u_vec = unsupervisedModel(x)[1].detach().flatten().cpu().numpy()
        vec = np.concatenate([s_vec, u_vec])
        representationVectors.append(vec)
    return np.array(representationVectors)




def configDataForMainClassifier():
    limitedData.representationVectorsForTrain    = buildRepresentationVectors('train')
    limitedData.trainDatasetForMainClassifier    = limitedData.MyDataset(limitedData.representationVectorsForTrain[limitedData.indicesOfTrainData], labels=limitedData.labelsOfTrainData)
    limitedData.trainDataLoaderForMainClassifier = DataLoader(limitedData.trainDatasetForMainClassifier, batch_size=limitedData.TRAIN_BATCH_MAIN_CLASSIFIER, shuffle=True, num_workers=2)

    limitedData.valDatasetForMainClassifier      = limitedData.MyDataset(limitedData.representationVectorsForTrain[limitedData.indicesOfValData], labels=limitedData.labelsOfValData)
    limitedData.valDataLoaderForMainClassifier   = DataLoader(limitedData.valDatasetForMainClassifier, batch_size=limitedData.TRAIN_BATCH_MAIN_CLASSIFIER, shuffle=False, num_workers=2)

    limitedData.representationVectorsForTest     = buildRepresentationVectors('test')
    limitedData.testDatasetForMainClassifier     = limitedData.MyDataset(limitedData.representationVectorsForTest, labels=limitedData.labelsOfTestData)
    limitedData.testDataLoaderForMainClassifier  = DataLoader(limitedData.testDatasetForMainClassifier, batch_size=limitedData.TRAIN_BATCH_MAIN_CLASSIFIER, shuffle=False, num_workers=2)

def configDataForMainClassifier_final():
    limitedData.representationVectorsForTrain    = buildRepresentationVectors('train')
    limitedData.trainDatasetForMainClassifier    = limitedData.MyDataset(limitedData.representationVectorsForTrain[limitedData.indicesOfTrainData], labels=limitedData.labelsOfTrainData)
    limitedData.trainDataLoaderForMainClassifier = DataLoader(limitedData.trainDatasetForMainClassifier, batch_size=limitedData.TRAIN_BATCH_MAIN_CLASSIFIER, shuffle=True, num_workers=2)

    limitedData.valDatasetForMainClassifier      = limitedData.MyDataset(limitedData.representationVectorsForTrain[limitedData.indicesOfValData], labels=limitedData.labelsOfValData)
    limitedData.valDataLoaderForMainClassifier   = DataLoader(limitedData.valDatasetForMainClassifier, batch_size=limitedData.TRAIN_BATCH_MAIN_CLASSIFIER, shuffle=False, num_workers=2)

    limitedData.representationVectorsForTest     = buildRepresentationVectors_final('test')
    limitedData.testDatasetForMainClassifier     = limitedData.MyDataset(limitedData.representationVectorsForTest, labels=limitedData.labelsOfTestData)
    limitedData.testDataLoaderForMainClassifier  = DataLoader(limitedData.testDatasetForMainClassifier, batch_size=limitedData.TRAIN_BATCH_MAIN_CLASSIFIER, shuffle=False, num_workers=2)



def pseudoLabel():
    global mainClassifier
    global NUM_PL

    limitedData.unlabeledDataset    = limitedData.MyDataset(limitedData.representationVectorsForTrain[limitedData.indicesOfUnabeledData])
    limitedData.unlabeledDataLoader = DataLoader(limitedData.unlabeledDataset, batch_size=limitedData.VAL_BATCH, shuffle=False, num_workers=2)

    mainClassifier.eval()
    confidenceList = np.array([])
    predictedLabelList = np.array([])
    with torch.no_grad():
        for x in limitedData.unlabeledDataLoader:
            if torch.cuda.is_available(): x = x.cuda()
            y_hat = mainClassifier(x)
            confidence, predictedLabels = y_hat.max(1)
            confidence = confidence.detach().cpu().numpy()
            predictedLabels = predictedLabels.detach().cpu().numpy()
            confidenceList = np.append(arr=confidenceList, values=confidence)
            predictedLabelList = np.append(arr=predictedLabelList, values=predictedLabels)

    NUM_PL = min(len(confidenceList), NUM_PL)
    indicesOfTopK = np.argpartition(confidenceList, -NUM_PL)[-NUM_PL:]
    
    indicesOfPseudolabeledData = limitedData.indicesOfUnabeledData[indicesOfTopK]
    labelsOfPseudolabeledData  = predictedLabelList[indicesOfTopK]
    # print("indicesOfPseudolabeledData",indicesOfPseudolabeledData)
    # print("labelsOfPseudolabeledData",labelsOfPseudolabeledData)
    
    # Update train
    limitedData.indicesOfTrainData   = np.append(arr=limitedData.indicesOfTrainData, values=indicesOfPseudolabeledData)
    limitedData.labelsOfTrainData    = np.append(arr=limitedData.labelsOfTrainData, values=labelsOfPseudolabeledData).astype(np.int64)
    limitedData.numOfTrainData += NUM_PL
    
    # Update labeled
    limitedData.indicesOfLabeledData = np.append(arr=limitedData.indicesOfLabeledData, values=indicesOfPseudolabeledData)
    limitedData.labelsOfLabeledData  = np.append(arr=limitedData.labelsOfLabeledData, values=labelsOfPseudolabeledData).astype(np.int64)
    limitedData.numOfLabeledData += NUM_PL
    
    # Update unlabeled
    mask = np.ones(limitedData.numOfAllData, dtype=bool)
    mask[limitedData.indicesOfLabeledData] = False
    limitedData.indicesOfUnabeledData = np.arange(limitedData.numOfAllData)[mask]
    limitedData.numOfUnlabeledData -= NUM_PL
    
    # Update trainDataLoader
    limitedData.trainDatasetForSupervisedModel = limitedData.MyDataset(limitedData.allImages[limitedData.indicesOfTrainData], transform=limitedData.transformWithAffine, labels=limitedData.labelsOfTrainData)
    limitedData.trainDataLoaderForSupervisedModel = DataLoader(limitedData.trainDatasetForSupervisedModel, batch_size=limitedData.TRAIN_BATCH, shuffle=True,  num_workers=limitedData.NUM_WORKER)

def summaryRound(roundID):
    print(f'Round [{roundID}/{NUM_ROUND}]', end=' ')
    print(f'numLabeled [{limitedData.numOfLabeledData}]', end=' ')
    print(f'numUnlabeled [{limitedData.numOfUnlabeledData}]', end=' ')
    print(f'numTrain [{limitedData.numOfTrainData}]', end=' ')
    print(f'+{NUM_PL}', end=' ')
    print(f'ESC [{statistics.earlyStopCountForTrainingRound}/{ESC_ROUND}]\n')

def main_exp(config):
    val_round =[]
    test_round = []
    numRound = config['hp']['num_round']
    for roundID in range(numRound+1):
        if pretrained and roundID ==0:
            print('12345')
            initialoneValSupervisedModel()
            initialTestsupervisedModel()
            saveModel("supervisedModel")
            saveModel("globalSupervisedModel")
        else:
            trainSupervisedModel()
        configDataForMainClassifier() #build L+UL vector
        trainMainClassifier()
        val_round.append(round(statistics.localBestValAcc*100,3)) #statistics.testAcc
        test_round.append(round(statistics.testAcc*100,3)) #statistics.testAcc
        if roundID != numRound: 
            pseudoLabel()
            summaryRound(roundID+1)
    finalTestMainClassifier()
    print(f'BestVal each round  [{val_round}]')
    print(f'TestAcc each round  [{test_round}]')
    config['BestVal_each_round'] = val_round
    config['TestAcc_each_round'] = test_round
    config['roundID'] = roundID
    if LOG: plot()

def plot():
    delta = 0.1
    fname = f'{ACC_LOSS_SAVE_PATH}'
    fig = plt.figure(figsize=(12, 9))
    
    title = f'BestVal[{statistics.globalBestValAcc:.3%}] Acc[{statistics.testAcc:.3%}] OPT[{OPTIM}] LR[{LR}] Batch[{limitedData.TRAIN_BATCH}] EPOCH[{NUM_EPOCH}] PL[{NUM_PL}/{NUM_ROUND}/{ESC_ROUND}]\n{EXPERIMENT_NAME}'    
    fig.suptitle(title)
    
    mapper = {0:'train', 1:'val', 2:'test'}

    sAcc = np.loadtxt(f'{fname}/sAcc', delimiter=',')
    ax = fig.add_subplot(2, 2, 1)
    ax.set_ylim(0-delta, 1+delta)
    for i in range(3):
        ax.plot(sAcc[:,i], label=mapper[i])
        ax.set_title('Supervised Model Acc')
        ax.legend()
    
    mAcc = np.loadtxt(f'{fname}/mAcc', delimiter=',')
    ax = fig.add_subplot(2, 2, 2)
    ax.set_ylim(0-delta, 1+delta)
    for i in range(3):
        ax.plot(mAcc[:,i])
        ax.set_title('Main Classifier Acc')
    
    sLoss = np.loadtxt(f'{fname}/sLoss', delimiter=',')
    ax = fig.add_subplot(2, 2, 3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for i in range(3):
        ax.plot(sLoss[:,i])
        ax.set_title('Supervised Model Loss')
    
    mLoss = np.loadtxt(f'{fname}/mLoss', delimiter=',')
    ax = fig.add_subplot(2, 2, 4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for i in range(3):
        ax.plot(mLoss[:,i])
        ax.set_title('Main Classifier Loss')
    plt.savefig(f'{ACC_LOSS_SAVE_PATH}/{int(statistics.testAcc*1e4)}')
    plt.close()

def main(config=None):
    seed = np.random.randint(1000)
    # seed = 819
    setup_seed(seed)
    config['seed'] = seed
    statistics.init()
    initExperiment(config)
    print("BATCH_SIZE = ", limitedData.TRAIN_BATCH)
    print("NUM_EPOCH = ", NUM_EPOCH)
    print("MAX_ESC = ", MAX_ESC)
    print("NUM_ROUND = ", NUM_ROUND)
    print("ESC_ROUND = ", ESC_ROUND)
    print("OPTIMIZER = ", OPTIM)
    print("Learning_rate = ", LR)
    print("Pseudo_rate = ", PL_RATE)

    main_exp(config)
    with open(f'{ACC_LOSS_SAVE_PATH}/config.yaml', 'w') as outfile:
        yaml.dump(config, outfile)
    # plot()
    print("BATCH_SIZE = ", limitedData.TRAIN_BATCH)
    print("NUM_EPOCH = ", NUM_EPOCH)
    print("MAX_ESC = ", MAX_ESC)
    print("NUM_ROUND = ", NUM_ROUND)
    print("ESC_ROUND = ", ESC_ROUND)
    print("OPTIMIZER = ", OPTIM)
    print("Learning_rate = ", LR)
    print("Pseudo_rate = ", PL_RATE)


if __name__ == '__main__':
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    main(config)