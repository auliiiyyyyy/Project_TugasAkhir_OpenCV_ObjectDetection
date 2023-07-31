import cv2 as cv
import numpy as np
from tracker import *
from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    "Video Streaming"
    return render_template('index.html')

def gen():
    "Video Streaming"

    #Memasukkan file video
    cap = cv.VideoCapture(r"D:\POLBAN\TUGAS\Bismillah TA lulus 2023\Program\Program_TA_Bismillah\Video1TA.mp4")

    #membuat tracker object
    tracker = Tracker()

    #Perintah untuk mengetahui resolusi video 
    w = cap.get(3) #lebar
    h = cap.get(4) #tinggi
    totalArea = w * h
    print(w, h, totalArea)

    #Perintah untuk mendapatkan informasi durasi video 
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    duration_seconds = frame_count / fps

    #Perintah membuat background subtractor 
    MOG2 = cv.createBackgroundSubtractorMOG2(history = 100, varThreshold = 90)

    #Mentukan batas kernel
    kernel = np.ones((5,5), np.uint8)

    #Area deteksi jalur masuk parkir
    point_area = [(209,200),(181,260),(432,227),(420,170)] #video1
    #point_area = [(146,193),(126,261),(406,216),(403,150)] #video2
    #point_area = [(134,98),(79,152),(348,143),(353,95)] #video3
    point_area_1 = set()

    #Area deteksi jalur keluar parkir
    point_area2 = [(492,320),(491,403),(832,410),(782,330)] #video1
    #point_area2 = [(492,320),(491,403),(832,414),(782,340)] #video2
    #point_area2 = [(408,160),(416,210),(801,205),(732,160)] #video3

    point_area_2 = set()


    #Melakukan pembacaan pada semua frame
    while True: 

        succes, frame = cap.read()
        if not succes:
            break

        else:
            #Mehitung durasi video
            current_pos = int(cap.get(cv.CAP_PROP_POS_MSEC) / 1000)
            duration_min = int(current_pos/60)
            duration_sec = int(current_pos % 60)

            #Mengaplikasikan background subtractor
            fgmask = MOG2.apply(frame)
            
            #Mengubah tampilan gambar menjadi hitam putih 
            ret,thresh = cv.threshold(fgmask, 175, 255, cv.THRESH_BINARY)

            #mengurangi noise atau derau 
            dilation = cv.dilate(thresh, kernel, iterations=4)
            mask = cv.erode(dilation, kernel, iterations=2)

            #mendeteksi objek dengan mencari kontur 
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            detections = []
                
            #Mendeteksi objek berdasarkan kontur 
            for cnt in contours:
                area = cv.contourArea(cnt)
                if area >9400:
                    (x,y,w,h) = cv.boundingRect(cnt)
                    cv.rectangle(frame,(x,y),(x+w,y+h),(48,255,173),3)
                    cv.rectangle(fgmask,(x,y),(x+w,y+h),(255,255,255),-1)    
                    detections.append([x,y,w,h])

            #Mendeteksi objek pada jalur masuk parkir
            boxes_id = tracker.update(detections)
            for up in boxes_id:
                x,y,w,h,id1 = up
                x1 = int(w/2)
                y1 = int(h/2)
                cx = x + x1
                cy = y + y1
                cv.circle(frame,(cx,cy),4,(0,0,255), -1)
                result = cv.pointPolygonTest(np.array(point_area,np.int32),
                        (cx,cy),False)
                if result > 0:
                    point_area_1.add(id1)

            #Mendeteksi objek pada jalur keluar parkir
            for down in boxes_id:
                x,y,w,h,id2 = down
                x2 = int(w/2)
                y2 = int(h/2)
                cx = x + x2
                cy = y + y2
                cv.circle(frame,(cx,cy),4,(0,0,255), -1)
                result2 = cv.pointPolygonTest(np.array(point_area2,np.int32),
                            (cx,cy),False)
                if result2 > 0:
                    point_area_2.add(id2)

            #Menghitung jumlah kendaraan 
            up = len(point_area_1)
            dwn = len(point_area_2)    
            jumlah = int(up-dwn)

            #Konndisi kepadatan kendaraan berdasarkan jumlah kendaraan 
            if 0 == jumlah < 90:
                Status = ('Area Parkir Masih Kosong')
            elif 91 == jumlah < 185:
                Status = ('Area Parkir Ramai')
            elif 186 == jumlah < 280:
                Status= ('Area Parkir Hampir Penuh')
            elif 281 == jumlah < 386:
                Status = ('Area Parkir Penuh')
                
            #Menampilkan area deteksi
            cv.polylines(frame,[np.array(point_area,np.int32)],True,(128,255,63),3)
            cv.polylines(frame,[np.array(point_area2,np.int32)],True,(71,99,250),3)
            
            #Menampilkan informasi pada frame
            cv.putText(frame, f"Durasi Video: {duration_min}menit {duration_sec}detik", 
                        (10, 430), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(frame, ('Motor Masuk ' + str(up)), 
                        (10,470), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA )
            cv.putText(frame, ('Motor Keluar ' + str(dwn)), 
                        (200,470), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA )
            cv.putText(frame, ('Total Motor ' + str(jumlah)), 
                        (400,470), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA )
            cv.putText(frame, ('Status: ' + str(Status)), 
                        (550,470), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA )

            #cv.imshow("Fgmask",fgmask)
            #cv.imshow("Frame",frame)

            buffer = cv.imencode('.jpg', frame)[1]
            buffer2 = cv.imencode('.jpg', fgmask)[1]

            frame = buffer.tobytes()
            frame2 = buffer2.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
            yield (b'--frame2\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
            
            key = cv.waitKey(30)
            if key == 27:
                break

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame2')
    
if __name__ == '__main__':
    app.run(debug=True)
