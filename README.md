# facial-recognition
## Hướng dẫn sử dụng code để train 
1. Tải dataset RAF-DB DATASET từ link [https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset]
2. Train
```
python train.py --model <tên model:IR_50,IR_50ViT,vgg19,resnet50> --image_train <đường dẫn tới ảnh train> --csv_train <đường dẫn tới csv train> --image_test <đường dẫn tới ảnh test> --csv_test <đường dẫn tới csv test --checkpoint_path <đường dẫn tới model đã được train trước> 
```

