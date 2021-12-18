import numpy as np
from termcolor import cprint
from tqdm import tqdm
import torch # GPU
from datetime import datetime
import pytz
from shutil import copyfile
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import yaml
from models import Wide_ResNet, SimpleNN3, VAE, Wide_ResNet_preMixup_final, Wide_ResNet_postMixup_final# Models
from resnet_vae import VAE_mitbih
from torch.utils.data import DataLoader
import statistics
from pathlib import Path
from ipdb import set_trace
from ranger import Ranger
import data as limitedData # Data
import random


class NetworkA1(torch.nn.Module):
    def __init__(self, channels):
        super(NetworkA1, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(channels, out_channels=16, kernel_size=(3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=(5, 5), padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=(7, 7), padding=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, channels // 2, kernel_size=(1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def oneTrainSupervisedModel():
    global net_a
    global criterion_a  
    global supervisedModel_preMixup
    global supervisedModel_postMixup
    global criterionForSupervisedModel
    global optimizerForSupervisedModel
    global statistics
    statistics.reset(mode="train")
    supervisedModel_preMixup.train()
    supervisedModel_postMixup.train()
    # TODO
    for batch_idx, (sa_data, mixup_data) in enumerate(zip(limitedData.trainDataLoaderForSupervisedModel_SA, limitedData.trainDataLoaderForSupervisedModel_mixup)):
        # set_trace()
        mixup_data_1, mixup_data_2 = mixup_data
        x_mixup_1, y_mixup_1 = mixup_data_1
        x_mixup_2, y_mixup_2 = mixup_data_2
        
        x, y = sa_data
        x1, x2, x3 = x
        if torch.cuda.is_available(): 
            x1, x2, x3, y = x1.cuda(), x2.cuda(), x3.cuda(), y.cuda()
            x_mixup_1, y_mixup_1, x_mixup_2, y_mixup_2 = x_mixup_1.cuda(), y_mixup_1.cuda(), x_mixup_2.cuda(), y_mixup_2.cuda()

        # smart augmentation
        inp = torch.cat([x2, x3], dim=1)
        new_img = net_a(inp)
        loss_a = criterion_a(new_img, x1) 
        inp_batch = torch.cat([new_img, x1], dim=0)
        
        y_sa, _ = supervisedModel_postMixup(supervisedModel_preMixup(inp_batch))

        y_sa_gt = torch.cat([y, y], dim=0)

        loss_b = criterionForSupervisedModel(y_sa, y_sa_gt.long())
 
        # mixup
        rep_1 = supervisedModel_preMixup(x_mixup_1)
        rep_2 = supervisedModel_preMixup(x_mixup_2)
        ratio = np.random.uniform(0, 1)
        rep_mixup = rep_1 * ratio + rep_2 * (1-ratio)
        
        y_rep, _ = supervisedModel_postMixup(rep_mixup)

        loss_mixup = criterionForSupervisedModel(y_rep, y_mixup_1.long()) * ratio + criterionForSupervisedModel(y_rep, y_mixup_2.long()) * (1-ratio)
        loss = (my_alpha*loss_a + my_beta*loss_b + loss_mixup)
        if accumulate_gradient:
            loss_accum = loss / accumulate_iter  # parameter for mixup loss
            loss_accum.backward()
            if (batch_idx + 1) % accumulate_iter == 0 or batch_idx + 1 == len(limitedData.trainDataLoaderForSupervisedModel_SA):
                optimizerForSupervisedModel.step()
                optimizerForSupervisedModel.zero_grad()
        else:
            loss = (my_alpha*loss_a + my_beta*loss_b + loss_mixup)
            # Backward pass
            loss.backward()
            optimizerForSupervisedModel.step()
            optimizerForSupervisedModel.zero_grad()
        # Statistics
        _, onehot = y_sa.max(1)
        statistics.numTotal += len(y_sa_gt)
        statistics.numCorrect += onehot.eq(y_sa_gt).sum().item()
        statistics.trainLoss += loss.item()
        statistics.trainLoss_a += loss_a.item()
        statistics.trainLoss_b += loss_b.item()
        statistics.trainLoss_mixup += loss_mixup.item()
    statistics.trainAcc = statistics.numCorrect / statistics.numTotal
    statistics.trainLoss /= statistics.numTotal
    statistics.trainLoss_a /= statistics.numTotal
    statistics.trainLoss_b /= statistics.numTotal
    statistics.trainLoss_mixup /= statistics.numTotal

def oneValSupervisedModel():
    global supervisedModel_preMixup
    global supervisedModel_postMixup
    global criterionForSupervisedModel
    global statistics
    statistics.reset(mode="val")
    supervisedModel_preMixup.eval()
    supervisedModel_postMixup.eval()
    with torch.no_grad():
        for x, y in limitedData.valDataLoaderForSupervisedModel:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat, _ = supervisedModel_postMixup(supervisedModel_preMixup(x))
            loss = criterionForSupervisedModel(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.valLoss += loss.item()
        statistics.valAcc = statistics.numCorrect / statistics.numTotal
        statistics.valLoss /= statistics.numTotal

def oneTestSupervisedModel(): 
    global supervisedModel_preMixup
    global supervisedModel_postMixup
    global criterionForSupervisedModel
    global statistics
    statistics.reset(mode="test")
    if not statistics.improved: 
        saveModel("tempSupervisedModel")
        # TODO
        loadModel("supervisedModel")
    supervisedModel_preMixup.eval()
    supervisedModel_postMixup.eval()
    with torch.no_grad():
        for x, y in limitedData.testDataLoaderForSupervisedModel:
            if torch.cuda.is_available(): x, y = x.cuda(), y.cuda()
            # Forward pass
            y_hat, _ = supervisedModel_postMixup(supervisedModel_preMixup(x))
            loss = criterionForSupervisedModel(y_hat, y.long())
            # Prediction
            _, onehot = y_hat.max(1)
            statistics.numTotal += len(y)
            statistics.numCorrect += onehot.eq(y).sum().item()
            statistics.testLoss += loss.item()
        statistics.testAcc = statistics.numCorrect / statistics.numTotal
        statistics.testLoss /= statistics.numTotal
    if not statistics.improved: loadModel("tempSupervisedModel")

def trainSupervisedModel():
    global statistics
    statistics.initRound()
    for epoch in range(NUM_EPOCH):
        oneTrainSupervisedModel()
        oneValSupervisedModel()
        
        if improved(model="supervisedModel", mode='local'): saveModel("supervisedModel")
        oneTestSupervisedModel()
        if shouldEarlyStop("supervisedModel"): break
        summaryModel(epoch+1, "supervisedModel")
        if LOG: log("supervisedModel")

def saveModel(model):
    if model == 'supervisedModel' or model == "tempSupervisedModel" or model == "globalSupervisedModel":
        global supervisedModel_preMixup
        global supervisedModel_postMixup

        global optimizerForSupervisedModel
        checkpoint = {
            'state_dict_preMixup' : supervisedModel_preMixup.state_dict(), 
            'state_dict_postMixup' : supervisedModel_postMixup.state_dict(), 
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
        global supervisedModel_preMixup
        global supervisedModel_postMixup
        global optimizerForSupervisedModel
        checkpoint = torch.load(f'{MODEL_SAVE_PATH}/{model}.pth')
        supervisedModel_preMixup.load_state_dict(checkpoint['state_dict_preMixup'])
        supervisedModel_postMixup.load_state_dict(checkpoint['state_dict_postMixup'])
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
    else:
        print("Warning: Undefined mode")
        
def shouldEarlyStop(model):
    global statistics
    if model == 'supervisedModel':
        if statistics.earlyStopCountForSupervisedModel > MAX_ESC: return True
    elif model == 'mainClassifier':
        if statistics.earlyStopCountForMainClassifier > MAX_ESC: return True
    else:
        print("Warning: Undefined model")
        return True
    return False

def initExperiment(config):
    limitedData.init(config)

    global EXPERIMENT_NAME
    global ACC_LOSS_SAVE_PATH
    global MODEL_SAVE_PATH
    global net_a
    global supervisedModel_preMixup
    global supervisedModel_postMixup
    global criterion_a
    global criterionForSupervisedModel
    global optimizerForSupervisedModel
    global unsupervisedModel
    global mainClassifier
    global criterionForMainClassifier
    global optimizerForMainClassifier
    global LOG                
    global LR 
    global PL_RATE                
    global NUM_PL             
    global NUM_EPOCH          
    global NUM_ROUND          
    global MAX_ESC
    global OPTIM
    global my_alpha
    global my_beta
    global accumulate_gradient
    global train_batch
    global train_batch_after_accumulate
    global accumulate_iter 
    global device

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOG = config['hp']['log']
    LR = config['hp']['lr']
    PL_RATE  = config['hp']['pl_rate']
    NUM_PL = int(limitedData.numOfLabeledData * PL_RATE)
    NUM_EPOCH = config['hp']['num_epoch']
    print(NUM_EPOCH)
    NUM_ROUND = config['hp']['num_round']
    MAX_ESC = config['hp']['max_esc']
    my_alpha = config['hp']['alpha']
    my_beta = config['hp']['beta']
    accumulate_gradient = config['hp']['accumulate_gradient']
    train_batch = config['hp']['train_batch']
    train_batch_after_accumulate = config['hp']['train_batch_after_accumulate']
    accumulate_iter = int(train_batch_after_accumulate / train_batch)
    OPTIM = config['hp']['optimizer']


    tpe = pytz.timezone('Asia/Taipei')
    EXPERIMENT_NAME = datetime.now(tpe).strftime("%Y-%m-%d %H:%M:%S")

    print(limitedData.N_CLASS)
    print(limitedData.RESIZE_SHAPE)
    print(device)

    channels = 1
    net_a = NetworkA1(channels=2*channels)
    net_a.to(device)
    criterion_a = torch.nn.MSELoss(size_average=True)
    supervisedModel_preMixup = Wide_ResNet_preMixup_final(28, 10, 0.2, limitedData.N_CLASS, data_shape=limitedData.RESIZE_SHAPE)
    supervisedModel_postMixup = Wide_ResNet_postMixup_final(28, 10, 0.2, limitedData.N_CLASS, data_shape=limitedData.RESIZE_SHAPE)
    supervisedModel_preMixup = torch.nn.DataParallel(supervisedModel_preMixup)
    supervisedModel_postMixup = torch.nn.DataParallel(supervisedModel_postMixup)
    mainClassifier = SimpleNN3(n_class=limitedData.N_CLASS, data_shape=limitedData.RESIZE_SHAPE)
    if config['hp']['pretrained']:
        checkpoint = torch.load(f'./pretrained_models/{config["hp"]["dataset"]}/Mixup/{OPTIM}/globalSupervisedModel.pth')
        supervisedModel_preMixup.load_state_dict(checkpoint['state_dict_preMixup'])
        supervisedModel_postMixup.load_state_dict(checkpoint['state_dict_postMixup'])
        checkpoint_main = torch.load(f'./pretrained_models/{config["hp"]["dataset"]}/Mixup/{OPTIM}/globalMainClassifier.pth')
        mainClassifier.load_state_dict(checkpoint_main['state_dict'])
    supervisedModel_preMixup.to(device)
    supervisedModel_postMixup.to(device)
    mainClassifier.to(device)

    torch.backends.cudnn.benchmark = True
    criterionForSupervisedModel = torch.nn.CrossEntropyLoss()
    if config['hp']['optimizer'] == 'Ranger':
        optimizerForSupervisedModel = Ranger(list(supervisedModel_preMixup.parameters())+list(supervisedModel_postMixup.parameters()), lr=LR, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5, weight_decay=0, use_gc=True, gc_conv_only=False)
        optimizerForMainClassifier = Ranger(mainClassifier.parameters(), lr=LR, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95, 0.999), eps=1e-5, weight_decay=0, use_gc=True, gc_conv_only=False)
    else:
        optimizerForSupervisedModel = torch.optim.Adam(list(supervisedModel_preMixup.parameters())+list(supervisedModel_postMixup.parameters()), lr=LR)
        optimizerForMainClassifier = torch.optim.Adam(mainClassifier.parameters(), lr=LR)

    if config['hp']['dataset'] == 'mitbih':
        unsupervisedModel = VAE_mitbih(640)
        unsupervisedModel.to(device)
        unsupervisedModel.load_state_dict(torch.load('./unsupervised_model_mitbih.pt'))
    else:
        unsupervisedModel = VAE()
        unsupervisedModel.to(device)
        unsupervisedModel.load_state_dict(torch.load('./best_u_model_VAE_CNN_AUG_640.pt'))
    ACC_LOSS_SAVE_PATH = f'./records/{config["hp"]["dataset"]}/R_SA_Mixup/{config["hp"]["optimizer"]}/{EXPERIMENT_NAME}'
    MODEL_SAVE_PATH = f'./checkpoints/{config["hp"]["dataset"]}/R_SA_Mixup/{config["hp"]["optimizer"]}/{EXPERIMENT_NAME}'
    Path(ACC_LOSS_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    Path(MODEL_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    # copyfile(src='./config.yaml', dst=f'{ACC_LOSS_SAVE_PATH}/config.yaml')
    with open(f'{ACC_LOSS_SAVE_PATH}/config.yaml', 'w') as outfile:
        yaml.dump(config, outfile)
    criterionForMainClassifier = torch.nn.CrossEntropyLoss()

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
        with open(f'{ACC_LOSS_SAVE_PATH}/sLoss_ab', 'a+') as f:
            f.write(str(statistics.trainLoss_a) + ',')
            f.write(str(statistics.trainLoss_b) + ',')
            f.write(str(statistics.trainLoss_mixup) + '\n')      
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
        y_hat = mainClassifier(x)
        loss = criterionForMainClassifier(y_hat, y.long())
        if accumulate_gradient:
            loss_accum = loss / accumulate_iter
            loss_accum.backward()
            if (batch_idx + 1) % accumulate_iter == 0 or batch_idx + 1 == len(limitedData.trainDataLoaderForMainClassifier):
                optimizerForMainClassifier.step()
                optimizerForMainClassifier.zero_grad()
        else:
            loss.backward()
            # Backward pass
            optimizerForMainClassifier.step()
            optimizerForMainClassifier.zero_grad()
        # Statistics
        _, onehot = y_hat.max(1)
        statistics.numTotal += len(y)
        statistics.numCorrect += onehot.eq(y).sum().item()
        statistics.trainLoss += loss.item()
    
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
        if improved(model="mainClassifier", mode='local'): saveModel('mainClassifier')
        if improved(model="mainClassifier", mode='global'): 
            saveModel("globalSupervisedModel") # TODO: salima
            saveModel('globalMainClassifier')
        oneTestMainClassifier()
        if shouldEarlyStop("mainClassifier"): break
        summaryModel(epoch+1, "mainClassifier")
        if LOG: log("mainClassifier")

def finalTestMainClassifier():
    global supervisedModel_preMixup
    global supervisedModel_postMixup
    global mainClassifier
    global criterionForMainClassifier
    global statistics
    statistics.reset(mode="test")
    # loadModel("globalSupervisedModel") 
    loadModel("globalMainClassifier")
    supervisedModel_preMixup.eval() 
    supervisedModel_postMixup.eval() 
    mainClassifier.eval()
    # TODO
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
    global supervisedModel_preMixup
    global supervisedModel_postMixup
    global unsupervisedModel
    loadModel("supervisedModel") # TODO: salima fixed
    if mode=='train':
        imagesDataset = limitedData.MyDataset(data=limitedData.allImages, transform=limitedData.transformWithoutAffine) 
    else:
        imagesDataset = limitedData.MyDataset(data=limitedData.testImages, transform=limitedData.transformWithoutAffine)
    representationVectors = []
    for x in tqdm(imagesDataset):
        x = x.to(device)
        x = torch.unsqueeze(x, 0)
        s_vec = supervisedModel_postMixup(supervisedModel_preMixup(x))[1].detach().flatten().cpu().numpy()
        u_vec = unsupervisedModel(x)[1].detach().flatten().cpu().numpy()
        vec = np.concatenate([s_vec, u_vec])
        representationVectors.append(vec)
    return np.array(representationVectors)

def buildRepresentationVectors_final(mode):
    global supervisedModel_preMixup
    global supervisedModel_postMixup
    global unsupervisedModel
    loadModel("globalSupervisedModel") # TODO: salima fixed
    if mode=='train':
        imagesDataset = limitedData.MyDataset(data=limitedData.allImages, transform=limitedData.transformWithoutAffine) 
    else:
        imagesDataset = limitedData.MyDataset(data=limitedData.testImages, transform=limitedData.transformWithoutAffine)
    representationVectors = []
    for x in tqdm(imagesDataset):
        x = x.to(device)
        x = torch.unsqueeze(x, 0)
        s_vec = supervisedModel_postMixup(supervisedModel_preMixup(x))[1].detach().flatten().cpu().numpy()
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
    # limitedData.representationVectorsForTrain    = buildRepresentationVectors('train')
    # limitedData.trainDatasetForMainClassifier    = limitedData.MyDataset(limitedData.representationVectorsForTrain[limitedData.indicesOfTrainData], labels=limitedData.labelsOfTrainData)
    # limitedData.trainDataLoaderForMainClassifier = DataLoader(limitedData.trainDatasetForMainClassifier, batch_size=limitedData.TRAIN_BATCH, shuffle=True, num_workers=2)

    # limitedData.valDatasetForMainClassifier      = limitedData.MyDataset(limitedData.representationVectorsForTrain[limitedData.indicesOfValData], labels=limitedData.labelsOfValData)
    # limitedData.valDataLoaderForMainClassifier   = DataLoader(limitedData.valDatasetForMainClassifier, batch_size=limitedData.TRAIN_BATCH, shuffle=False, num_workers=2)

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
    limitedData.trainDatasetForSupervisedModel_SA = limitedData.MyDataset_SA(limitedData.allImages[limitedData.indicesOfTrainData], transform=limitedData.transformWithAffine, labels=limitedData.labelsOfTrainData)
    limitedData.trainDataLoaderForSupervisedModel_SA = DataLoader(limitedData.trainDatasetForSupervisedModel_SA, batch_size=limitedData.TRAIN_BATCH, shuffle=True,  num_workers=limitedData.NUM_WORKER)


    # Update trainDataLoader
    limitedData.trainDatasetForSupervisedModel_mixup = limitedData.MyDataset_mixup(limitedData.allImages[limitedData.indicesOfTrainData], transform=limitedData.transformWithAffine, labels=limitedData.labelsOfTrainData)
    limitedData.trainDataLoaderForSupervisedModel_mixup = DataLoader(limitedData.trainDatasetForSupervisedModel_mixup, batch_size=limitedData.TRAIN_BATCH, shuffle=True,  num_workers=limitedData.NUM_WORKER)


def summaryRound(roundID):
    print(f'Round [{roundID}/{NUM_ROUND}]', end=' ')
    print(f'numLabeled [{limitedData.numOfLabeledData}]', end=' ')
    print(f'numUnlabeled [{limitedData.numOfUnlabeledData}]', end=' ')
    print(f'numTrain [{limitedData.numOfTrainData}]', end=' ')
    print(f'+{NUM_PL}\n')

def main_exp(config):
    numRound = config['hp']['num_round']
    for roundID in range(numRound+1):
        trainSupervisedModel()
        configDataForMainClassifier()
        trainMainClassifier()
        if roundID != numRound: 
            pseudoLabel()
            summaryRound(roundID+1)
    finalTestMainClassifier()
    if LOG: plot()

def plot():
    delta = 0.1
    fname = f'{ACC_LOSS_SAVE_PATH}'
    fig = plt.figure(figsize=(12, 9))
    
    title = f'Acc[{statistics.testAcc:.2%}] OPT[Adam] LR[{LR}] Batch[{limitedData.TRAIN_BATCH}] EPOCH[{NUM_EPOCH}] PL[{NUM_PL}/{NUM_ROUND}]\n{EXPERIMENT_NAME}'
    fig.suptitle(title)
    
    mapper = {0:'train', 1:'val', 2:'test'}

    sAcc = np.loadtxt(f'{fname}/sAcc', delimiter=',')
    ax = fig.add_subplot(3, 3, 1)
    ax.set_ylim(0-delta, 1+delta)
    for i in range(3):
        ax.plot(sAcc[:,i], label=mapper[i])
        ax.set_title('Supervised Model Acc')
        ax.legend()
    
    mAcc = np.loadtxt(f'{fname}/mAcc', delimiter=',')
    ax = fig.add_subplot(3, 3, 2)
    ax.set_ylim(0-delta, 1+delta)
    for i in range(3):
        ax.plot(mAcc[:,i])
        ax.set_title('Main Classifier Acc')
    
    sLoss = np.loadtxt(f'{fname}/sLoss', delimiter=',')
    ax = fig.add_subplot(3, 3, 3)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for i in range(3):
        ax.plot(sLoss[:,i])
        ax.set_title('Supervised Model Loss')

    sLoss_ab = np.loadtxt(f'{fname}/sLoss_ab', delimiter=',')
    ax = fig.add_subplot(3, 3, 4)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.plot(sLoss_ab[:,0])
    ax.set_title('Supervised Model Loss_a')
    
    ax = fig.add_subplot(3, 3, 5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.plot(sLoss_ab[:,1])
    ax.set_title('Supervised Model Loss_b')

    ax = fig.add_subplot(3, 3, 6)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.plot(sLoss_ab[:,2])
    ax.set_title('Supervised Model Loss_mixup')


    mLoss = np.loadtxt(f'{fname}/mLoss', delimiter=',')
    ax = fig.add_subplot(3, 3, 7)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for i in range(3):
        ax.plot(mLoss[:,i])
        ax.set_title('Main Classifier Loss')
    plt.savefig(f'{ACC_LOSS_SAVE_PATH}/{int(statistics.testAcc*1e4)}')
    plt.close()

def main(config):
    seed = np.random.randint(1000)
    setup_seed(seed)
    config['seed'] = seed
    statistics.init()
    initExperiment(config)
    print("BATCH_SIZE = ", limitedData.TRAIN_BATCH)
    print("NUM_EPOCH = ", NUM_EPOCH)
    print("NUM_ROUND = ", NUM_ROUND)
    print("MAX_ESC = ", MAX_ESC)
    print("Learning_rate = ", LR)
    print("my_alpha = ", my_alpha)
    print("my_beta = ", 1-my_alpha)
    main_exp(config)
    # plot()
    print("BATCH_SIZE", limitedData.TRAIN_BATCH)
    print("NUM_EPOCH = ", NUM_EPOCH)
    print("NUM_ROUND = ", NUM_ROUND)
    print("MAX_ESC = ", MAX_ESC)
    print("Learning_rate = ", LR)
    print("my_alpha = ", my_alpha)
    print("my_beta = ", 1-my_alpha)

if __name__ == '__main__':
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    main(config)