import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('rtdetr-l.yaml')
    # model.load('') # loading pretrain weights
    model.train(data=r'替换大家自己数据集的地址',
                cache=False,
                imgsz=640,
                epochs=72,
                batch=4,
                workers=0,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                # amp=True
                )