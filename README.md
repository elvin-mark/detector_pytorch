# detector_pytorch
Trainer for object detection models

## How to use?
```
python detector.py \
  --model {fasterrcnn} \  
  --root ROOT \           
  --batch-size BATCH_SIZE \
  --epochs EPOCHS \       
  --lr LR \ 
  --optim {SGD,Adam} \   
  --gpu \                
  --no-pretrained \
  --num-classes NUM_CLASSES \
  --save-model
```

Your images for training should be structures as follows:
```
root/
  train/
    img_1.jpg
    ...
  test/
    img_2.jpg
    ...
  annotations/
    img_1.xml
    ...
  labels.txt
```