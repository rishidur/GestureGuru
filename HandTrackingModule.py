import cv2
import mediapipe as mp
import time




class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]


            for id,lm in enumerate(myHand.landmark):

                h ,w, c = img.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                print(id, cx,cy)

                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

        return self.lmList

    def numOfFingers(self, img):
        if len(self.lmList) >= 21:

            thumb_open = self.lmList[4][1] > self.lmList[2][1]
            index_finger_open = self.lmList[8][2] < self.lmList[6][2]
            middle_finger_open = self.lmList[12][2] < self.lmList[10][2]
            ring_finger_open = self.lmList[16][2] < self.lmList[14][2]
            pinky_finger_open = self.lmList[20][2] < self.lmList[18][2]


            open_fingers = [thumb_open, index_finger_open, middle_finger_open, ring_finger_open, pinky_finger_open]
            count_open = open_fingers.count(True)


            if count_open == 0:
                cv2.putText(img, "Zero", (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            elif count_open == 1:
                cv2.putText(img, "One", (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            elif count_open == 2:
                cv2.putText(img, "Two", (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            elif count_open == 3:
                cv2.putText(img, "Three", (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            elif count_open == 4:
                cv2.putText(img, "Four", (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            elif count_open == 5:
                cv2.putText(img, "Five", (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Unknown Gesture", (200, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        else:
            print("Insufficient landmarks detected")


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()


    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        fingers = detector.numOfFingers(img)

        # if len(lmList) != 0:
        #     print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
