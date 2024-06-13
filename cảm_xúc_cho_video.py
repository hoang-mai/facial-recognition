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

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        probabilities = probabilities.squeeze(0)
        probabilities = probabilities.tolist()
        emotions_prob = {class_labels[i]: probabilities[i] for i in range(len(class_labels))}
    return emotions_prob


def detect_faces_from_video(args):
    alg = "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(alg)

    cap = cv2.VideoCapture(args.video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_img_rgb)

            emotion = classify_image(pil_image, args)
            print("Emotion:", emotion)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (203, 192, 255), 2)

            for i, (emotion1, prob) in enumerate(emotion.items()):
                text = f"{emotion1}: {prob:.2f}"
                y_offset = i * 30
                frame = cv2.putText(frame, text, (0, 10 + y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        scaled_frame = cv2.resize(frame, (800, 700))
        cv2.imshow("Face Detection", scaled_frame)


        key = cv2.waitKey(10)

        if key == 27 or cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--model', default='IR_50ViT', type=str, help='tên model cần test')
    parser.add_argument('--checkpoint_path', default='C:\\Users\\Laptop\\Downloads\\best_model.pth',
                        help='đường dẫn tới thư mục chứa file pretrain')
    parser.add_argument('--video_path', default='C:\\Users\\Laptop\\Downloads\\video.mp4',
                        help='đường dẫn chứa video test')
    args = parser.parse_args()
    detect_faces_from_video(args)
