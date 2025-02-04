import cv2 as cv
import mediapipe as mp 
import csv
import os
mphand = mp.solutions.hands
hands = mphand.Hands()
mpdraw= mp.solutions.drawing_utils

def opendata():
    ogPath = "C:/Users/amend/Desktop/Hand_Images/archive/hagrid-classification-512p"
    #print(ogPath)

    # Loop through the directories in the path
    for path in os.listdir(ogPath):
        if path in ["call", "stop", "dislike", "like", "one", "ok"]:
            print(path)

            paths = os.path.join(ogPath, path)
            
            if path == "one":
                # Process "one" gesture for right and left orientation
                process_orientation(paths, "right", cv.ROTATE_90_CLOCKWISE)
                process_orientation(paths, "left", cv.ROTATE_90_COUNTERCLOCKWISE)
            else:
            # Process other gestures as-is
                li=os.listdir(paths)
                for i in range (600,1200):
                #for images in os.listdir(paths):
                    images=li[i]
                    imagepath = os.path.join(paths, images)
                    final = cv.imread(imagepath)
                    if final is not None:
                        collectdata(final, path)
                    else:
                        print(f"Could not read image: {imagepath}")
def process_orientation(paths, gesture, rotation):
    print(gesture)
    li=os.listdir(paths)
    for i in range (600,1200):
        images=li[i]
    #for images in os.listdir(paths):
        imagepath = os.path.join(paths, images)
        final = cv.imread(imagepath)
        if final is not None:
            rotated_final = cv.rotate(final, rotation)
            collectdata(rotated_final, gesture)
        else:
            print(f"Could not read image: {imagepath}")
    
def collectdata(image, gestures):
    # Initialize Mediapipe once
    
    
    # Open the CSV file in append mode
    with open("C:/Users/amend/Desktop/hand_Data_test_BIG2.csv", mode='a', newline='') as file:

        writer = csv.writer(file)

        frame_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        
        if result.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = result.multi_handedness[hand_index].classification[0].label
                mpdraw.draw_landmarks(image, hand_landmarks, mphand.HAND_CONNECTIONS)
                
                # Collect landmark data
                land = []
                for lm in hand_landmarks.landmark:
                    land.extend([lm.x, lm.y, lm.z])
                
                # Write gesture and landmarks to CSV
                writer.writerow([gestures, handedness] + land)

        # Display the image
        cv.imshow('feed', cv.flip(image, 1))
        if cv.waitKey(1) == ord("q"):
            cv.destroyAllWindows()

def main():
    opendata()

if __name__ == "__main__":
    main()
