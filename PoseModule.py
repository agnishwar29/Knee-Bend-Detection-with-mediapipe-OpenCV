import cv2
import mediapipe as mp
import time

class PoseDetector():

    def __init__(self, mode=False, complexity=1, smooth_land=True,
                 segmentation=False, smooth_segmentation=True
                 , detection_conf=0.5, track_conf=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth_land = smooth_land
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_land,
                                     self.segmentation, self.smooth_segmentation,
                                     self.detection_conf,self.track_conf)

    def findPose(self,img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):

        lm_list = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list



def main():
    cap = cv2.VideoCapture('videos/KneeBendVideo.mp4')
    bends =0
    count = 0
    start =0
    elapsed = 0

    detector= PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)


        if lmList:
            if abs(lmList[25][2] - lmList[23][2]) >= 20:
                elapsed = time.time() - start
                # print(int(elapsed))
                if elapsed:
                    cv2.putText(img, str(f"Timer- {int(elapsed)}"), (650, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)

                count += 1

            elif abs(lmList[25][2] - lmList[23][2]) <= 20:
                cv2.putText(img, str("Keep your knee bent"), (480, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
                start = time.time()
                count = 0

            if count == 210:
                bends += 1




            cv2.circle(img, (lmList[29][1], lmList[29][2]), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (lmList[23][1], lmList[23][2]), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (lmList[25][1], lmList[25][2]), 10, (0, 0, 255), cv2.FILLED)

            cv2.putText(img, str(f"Bend Count- {bends}"), (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
