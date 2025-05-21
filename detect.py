import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO('/home/irem/Documents/PythonProjects/ObjectDetection/runs/detect/yolov8_model10/weights/best.pt')  # 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt' gibi farklı modeller kullanılabilir

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamerayı temsil eder

if not cap.isOpened():
    print("Kamera açilamadi!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı, çıkılıyor...")
        break

    # YOLOv8 ile tahmin yap
    results = model(frame)

    # Tahmin edilen sonuçları çizin
    annotated_frame = results[0].plot()

    # Çerçeveyi göster
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows() 