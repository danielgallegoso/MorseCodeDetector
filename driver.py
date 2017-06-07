import cv2
import  blobDetection
import motionTracking
import translate

cap = cv2.VideoCapture('data/notdark.mov')

blobs = []
count = 0
while(True):
    # Capture frame-by-frame
    ret, im = cap.read()

    if not ret: break
    if count % 50 == 0:
        print "starting frame #{0}".format(count)
    count += 1
    keypoints = blobDetection.findBlob(im)
    blobs = motionTracking.track(blobs, keypoints)
    blobs = motionTracking.prune_noise(blobs)
    blobDetection.getKeypointIm(im, blobs)
    # blobDetection.debugKeypointIm(im, keypoints)
    # if len(blobs[0].signal)>10:
    #     print translate.decode(blobs[0].signal*2 - 1)
    # print len(blobs)

    # Display the resulting frame
    cv2.imshow('frame',im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

blobs = motionTracking.prune_noise(blobs, 0, True)
for blob in blobs:
    print translate.decode(blob.signal*2 - 1)
    print list(blob.signal*2 - 1)

cap.release()
cv2.destroyAllWindows()
