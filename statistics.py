def init():
    global trainAcc  ; trainAcc                                               = 0
    global valAcc    ; valAcc                                                 = 0
    global testAcc   ; testAcc                                                = 0
    global trainLoss ; trainLoss                                              = 0
    global trainLoss_a ; trainLoss_a                                          = 0
    global trainLoss_b ; trainLoss_b                                          = 0
    global trainLoss_mixup ; trainLoss_mixup                                  = 0
    global valLoss   ; valLoss                                                = 0
    global testLoss  ; testLoss                                               = 0
    global numCorrect; numCorrect                                             = 0
    global numTotal  ; numTotal                                               = 0
    global localBestValAcc; localBestValAcc                                   = 0
    global globalBestValAcc; globalBestValAcc                                 = 0
    global earlyStopCountForSupervisedModel; earlyStopCountForSupervisedModel = 0
    global earlyStopCountForMainClassifier; earlyStopCountForMainClassifier   = 0
    global improved; improved                                                 = False
    

init()

def initRound():
    global trainAcc  ; trainAcc                                               = 0
    global valAcc    ; valAcc                                                 = 0
    global testAcc   ; testAcc                                                = 0
    global trainLoss ; trainLoss                                              = 0
    global trainLoss_a ; trainLoss_a                                          = 0
    global trainLoss_b ; trainLoss_b                                          = 0
    global trainLoss_mixup ; trainLoss_mixup                                  = 0
    global valLoss   ; valLoss                                                = 0
    global testLoss  ; testLoss                                               = 0
    global numCorrect; numCorrect                                             = 0
    global numTotal  ; numTotal                                               = 0
    global localBestValAcc; localBestValAcc                                   = 0
    global earlyStopCountForSupervisedModel; earlyStopCountForSupervisedModel = 0
    global earlyStopCountForMainClassifier; earlyStopCountForMainClassifier   = 0
    global improved; improved                                                 = False

def summary():
    print('trainAcc   =', trainAcc     )
    print('valAcc     =', valAcc       )
    print('testAcc    =', testAcc      )
    print('trainLoss  =', trainLoss    )
    print('trainLoss_a  =', trainLoss_a    )
    print('trainLoss_b  =', trainLoss_b    )
    print('trainLoss_mixup  =', trainLoss_mixup )
    print('valLoss    =', valLoss      )
    print('testLoss   =', testLoss     )
    print('numCorrect =', numCorrect     )
    print('numTotal   =', numTotal       )
    print('localBestValAcc =', localBestValAcc  )
    print('globalBestValAcc =', globalBestValAcc  )

def reset(mode):
    global trainAcc   ;
    global valAcc     ;
    global testAcc    ;
    global trainLoss  ;
    global trainLoss_a  ;
    global trainLoss_b  ;
    global trainLoss_mixup  ;
    global valLoss    ;
    global testLoss   ;
    global numCorrect   ;
    global numTotal     ;
    global improved;
    if mode=='train':
        improved = False
        trainAcc = 0
        trainLoss = 0
        trainLoss_a = 0
        trainLoss_b = 0
        trainLoss_mixup = 0

    if mode=='val':
        valAcc = 0
        valLoss = 0
    if mode=='test':
        testAcc = 0
        testLoss = 0
    numCorrect = 0
    numTotal = 0
    