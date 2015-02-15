# Various plots from tutorial

def plotThresholding():
    f = plt.figure(figsize=(12,3))
    sub1 = plt.subplot(1,4,1)
    plt.imshow(im, cmap=cm.gray)
    sub1.set_title("Original Image")
    
    sub2 = plt.subplot(1,4,2)
    plt.imshow(imthr, cmap=cm.gray_r)
    sub2.set_title("Thresholded Image")
    
    sub3 = plt.subplot(1, 4, 3)
    plt.imshow(imdilated, cmap=cm.gray_r)
    sub3.set_title("Dilated Image")

    sub4 = plt.subplot(1, 4, 4)
    sub4.set_title("Labeled Image")
    #plt.imshow(labels)
    #plt.show()