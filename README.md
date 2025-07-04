
# 🚨 Helmet and Face Recognition-Based Safety System

A real-time safety enforcement system using **YOLOv11** for helmet detection and the **face_recognition** library for identifying individuals. If a person is detected without a helmet, the system automatically identifies them and sends a **challan (fine notice)** to their registered email. Ideal for monitoring safety compliance in colleges, industries, and construction sites.

---

## ⚙️ Features

✅ Real-time helmet detection using **YOLOv11**  
✅ Face recognition using the **face_recognition** library  
✅ Automated challan with fine details sent via email  
✅ Student data managed through a `.csv` file  
✅ Easy to set up and extend for SMS/WhatsApp alerts (future)  

---

## 🛠️ Technologies Used

- Python  
- OpenCV  
- YOLOv11 (Helmet Detection)  
- face_recognition (Face Identification)  
- Pandas  
- SMTP (Email Alerts)  

---

## 🚀 How It Works

1. Live video feed is captured.  
2. **YOLOv11** checks for helmet presence.  
3. If helmet is absent:  
   - **face_recognition** identifies the person.  
   - A challan (fine notice) is emailed with their details.  

---
