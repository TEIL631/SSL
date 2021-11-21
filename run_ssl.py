import yaml
import S
import P
import R
import S_Mixup
import R_Mixup
import R_SA_Mixup

config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)

task = config['hp']['task']
repeat_times = config['hp']['repeat_times']

for _ in range(repeat_times):
    if task == 'S':
        S.main(config)
    elif task == 'R':
        R.main(config)
    elif task == 'P':
        P.main(config)
    elif task == 'S_Mixup':
        S_Mixup.main(config)
    elif task == 'R_Mixup':
        R_Mixup.main(config)
    elif task == 'R_SA_Mixup':
        R_SA_Mixup.main(config)