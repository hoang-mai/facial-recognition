import argparse

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def classify_image(image, args):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    if args.model == 'IR_50':
        from backbone.IR_50 import model
    elif args.model == 'IR_50ViT':
        from backbone.IR_50ViT import model
    elif args.model == 'vgg19':
        from backbone.vgg19 import model
    else:
        from backbone.resnet50 import model
    class_labels = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
    device = torch.device("cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    # Chuyển đổi ảnh thành tensor và chuẩn hóa

    # Áp dụng biến đổi và chuyển ảnh sang tensor
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Phân loại ảnh và trả về nhãn dự đoán
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

    return class_labels[predicted_label]


def main(args):
    # Đường dẫn tới file haarcascade
    alg = "haarcascade_frontalface_default.xml"

    # Tạo đối tượng classifier
    haar_cascade = cv2.CascadeClassifier(alg)

    # Mở video feed từ camera
    cam = cv2.VideoCapture(0)

    while True:
        _, img = cam.read()

        # Chuyển ảnh sang dạng grayscale
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt sử dụng Haar Cascade
        faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

        # Vẽ hình chữ nhật xung quanh khuôn mặt và phân loại cảm xúc
        for (x, y, w, h) in faces:
            # Cắt ra ảnh chứa khuôn mặt
            face_img = img[y:y + h, x:x + w]

            # Chuyển ảnh từ BGR sang RGB
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # Chuyển ảnh sang định dạng Image của PIL
            pil_image = Image.fromarray(face_img_rgb)

            # Phân loại cảm xúc và in kết quả
            emotion = classify_image(pil_image, args)
            print("Emotion:", emotion)

            # Vẽ hình chữ nhật xung quanh khuôn mặt
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Hiển thị cảm xúc trên ảnh
            img = cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị ảnh
        cv2.imshow("Face Detection", img)
        key = cv2.waitKey(10)

        if key == 27:
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--model', default='IR_50ViT', type=str, help='tên model cần test')
    parser.add_argument('--checkpoint_path',
                        default='C:\\Users\\Laptop\\Downloads\\best_model.pth',
                        help='đường dẫn tới thư mục chứa file pretrain')

    args = parser.parse_args()
    main(args)
