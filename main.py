
import cv2
from ultralytics import YOLO

# 1. Load model YOLOv8n
print("Dang tai mo hinh YOLOv8n...")
model = YOLO("yolov8n.pt")


# 2. Demo nhieu anh
def detect_image():

    image_paths = ["test.jpg", "test1.jpg"]

    for image_path in image_paths:

        print(f"\nDang xu ly anh: {image_path}")

        image = cv2.imread(image_path)

        if image is None:
            print(f"Khong tim thay anh: {image_path}")
            continue

        # Nhan dien doi tuong
        results = model.predict(
            source=image,
            conf=0.5,
            verbose=False
        )

        result = results[0]

        # In ket qua nhan dien
        print("\nCAC DOI TUONG PHAT HIEN:")

        if len(result.boxes) == 0:
            print("Khong phat hien doi tuong nao!")

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[class_id]

            print(f"{name}: {confidence:.2f}")

        # Ve Bounding Box
        output = result.plot()

        # Hien thi anh
        cv2.imshow(
            f"YOLOv8 - {image_path}",
            output
        )

        print("Nhan phim bat ky tren cua so anh de xem anh tiep theo.")

        # Cho nguoi dung nhan phim
        cv2.waitKey(0)

        # Dong cua so anh hien tai
        cv2.destroyAllWindows()

    print("\nDa hoan thanh nhan dien tat ca anh!")


# 3. Demo voi webcam
def detect_webcam():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Khong mo duoc webcam!")
        return

    print("Dang chay webcam...")
    print("Nhan phim Q de thoat.")

    while True:

        success, frame = camera.read()

        if not success:
            print("Khong doc duoc khung hinh!")
            break

        # Nhan dien doi tuong
        results = model.predict(
            source=frame,
            conf=0.5,
            verbose=False
        )

        # Ve Bounding Box
        output = results[0].plot()

        cv2.imshow(
            "YOLOv8 - Webcam Detection",
            output
        )

        # Nhan Q de thoat
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


# 4. Menu lua chon
print("\n===== OBJECT DETECTION WITH YOLOv8 =====")
print("1. Detect Objects in Image")
print("2. Detect Objects using Webcam")

choice = input("Chon che do (1/2): ")

if choice == "1":
    detect_image()

elif choice == "2":
    detect_webcam()

else:
    print("Lua chon khong hop le!")
