import argparse

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image





def classify_image(image,args):
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

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        probabilities = probabilities.squeeze(0)  # Loại bỏ kích thước batch
        probabilities = probabilities.tolist()  # Chuyển tensor thành list
        emotions_prob = {class_labels[i]: probabilities[i] for i in range(len(class_labels))}
    return emotions_prob

def detect_faces(args):
    # Load ảnh từ đường dẫn
    image = cv2.imread(args.image_path)
    alg = "haarcascade_frontalface_default.xml"

    # Tạo đối tượng classifier
    haar_cascade = cv2.CascadeClassifier(alg)
    # Chuyển đổi ảnh sang đen trắng (grayscale)
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x, y, w, h) in faces:
        # Cắt ra ảnh chứa khuôn mặt
        face_img = image[y:y + h, x:x + w]

        # Chuyển ảnh từ BGR sang RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Chuyển ảnh sang định dạng Image của PIL
        pil_image = Image.fromarray(face_img_rgb)

        # Phân loại cảm xúc và in kết quả
        emotion = classify_image(pil_image,args)
        print("Emotion:", emotion)

        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(image, (x, y), (x + w, y + h), (203, 192, 255), 2)



        for i, (emotion1, prob) in enumerate(emotion.items()):
            text = f"{emotion1}: {prob:.2f}"
            y_offset = i * 30  # Điều chỉnh vị trí văn bản
            image = cv2.putText(image, text, (0, 10+y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    scaled_image = cv2.resize(image, (800, 700))

    # Hiển thị ảnh
    cv2.imshow("Face Detection", scaled_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--model', default='IR_50ViT', type=str, help='tên model cần test')
    parser.add_argument('--checkpoint_path',
                        default='C:\\Users\\Laptop\\Downloads\\best_model.pth',
                        help='đường dẫn tới thư mục chứa file pretrain')
    parser.add_argument('--image_path',default='C:\\Users\\Laptop\\Downloads\\buonba.jpg',help='đường dẫn chứa ảnh test')
    args = parser.parse_args()
    detect_faces(args)

# Example usage:
#image_path = "C:\\Users\Laptop\\Downloads\\vuive.jpg"
#image_path = "C:\\Users\Laptop\\Downloads\\tucgian.jpg"
#image_path = "C:\\Users\Laptop\\Downloads\\kinhtom.jpg"
#image_path ="C:\\Users\\Laptop\\Downloads\\batngo.webp"
#image_path ="C:\\Users\\Laptop\\Downloads\\buonba.jpg"
#image_path ="C:\\Users\\Laptop\\Downloads\\binhthuong.jpg"
#image_path ="C:\\Users\\Laptop\\Downloads\\sohai.jpg"
