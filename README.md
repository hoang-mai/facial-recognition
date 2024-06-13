# facial-recognition
## Hướng dẫn sử dụng code để train 
1. Tải dataset RAF-DB DATASET từ link [here](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)
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
Model đã được train trước này dùng để nhận diện khuôn mặt đã được train với tập dữ liệu MS-Celeb-1M_Align_112x112 gồm 5,822,653 ảnh từ link [here](https://drive.google.com/file/d/1EsGSnY7KlzDTPA2EDWxQ1ey06kivRr0l/view?usp=drive_link)

Sử dụng model này để train 2 model là IR_50 và IR_50ViT
Còn đối với resnet50 và vgg19 thì dùng model đã được train trước với tập dữ liệu là ImageNet
## Đánh giá model
Tất cả các model đều được train với ảnh 112*112 của tập dataset RAF_DB DATASET , mean=[0.5, 0.5, 0.5] , std=[0.5, 0.5, 0.5] 


| Model    | Pretrain | Nopretrain   |Kích thước|
|--------|------|-------------|----------------|
| Vgg19    | 86,01%   |80.46%      |558.41 MB|
| ResNet50    | 84,97%   | 67,69% |94.41 MB|
| IR_50   | 88,65%   | 80,93%    |123.81 MB|
| IR_50ViT |  88,88%  |  81.12%   |348.75 MB|

Lý do model IR_50ViT tốt hơn so với mô hình IR_50 là do model sâu hơn và nhiều trọng số hơn

Link tới model IR_50ViT [here](https://drive.google.com/file/d/1_R-DWByrVQu8Hdvr8vkgAb2mLeLd99dZ/view?usp=sharing)
Link tới model IR_50 [here](https://drive.google.com/file/d/1KwixXWrJBlemIhj70K56vvmEyzNqaw-A/view?usp=sharing)
  
