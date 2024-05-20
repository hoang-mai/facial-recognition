# facial-recognition
## Hướng dẫn sử dụng code để train 
1. Tải dataset RAF-DB DATASET từ link https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset
2. Train
```
python train.py --model <tên model:IR_50,IR_50ViT,vgg19,resnet50> --image_train <đường dẫn tới ảnh train> --csv_train <đường dẫn tới csv train> --image_test <đường dẫn tới ảnh test> --csv_test <đường dẫn tới csv test --checkpoint_path <đường dẫn tới model đã được train trước> 
```
3.Test

```
python cảm_xúc_có_xác_xuất_cao_nhất.py --model <tên model:IR_50,IR_50ViT,vgg19,resnet50> --checkpoint_path <đường dẫn tới model đã được train> 
```

```
python hiện_xác_xuất_cảm_xúc.py --model <tên model:IR_50,IR_50ViT,vgg19,resnet50> --checkpoint_path <đường dẫn tới model đã được train> 
```

```
python cảm_xúc_khi_cho_ảnh.py --model <tên model:IR_50,IR_50ViT,vgg19,resnet50> --checkpoint_path <đường dẫn tới model đã được train> --image_path <đường dẫn cho ảnh>
```

## Model đã được train trước 
Model đã được train trước này dùng để nhận diện khuôn mặt đã được train với 5,822,653 ảnh từ link https://drive.google.com/file/d/1EsGSnY7KlzDTPA2EDWxQ1ey06kivRr0l/view?usp=drive_link

Sử dụng model này để train 2 model là IR_50 và IR_50ViT
