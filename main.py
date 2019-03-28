import cv2
from WordSegmentation import wordSegmentation, prepareImg
from DataLoader import Batch
from SamplePreprocessor import preprocess
from Model import Model, DecoderType
import numpy as np


def infer(model, fnImg):
    " recognize text in image provided by file path "
    img = preprocess(fnImg, Model.imgSize)
    batch = Batch(None, [img] * Model.batchSize)
    recognized = model.inferBatch(batch)
    return recognized[0]


def doWords(img, model):
    img = prepareImg(img, 50)
    # img = cv2.fastNlMeansDenoising(img, None)
    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 11, 11)
    cv2.imshow("1", img)
    cv2.waitKey(0)
    res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
    words = ""
    print(len(res))
    for (j, w) in enumerate(res):
        (wordBox, wordImg) = w
        words = words + " " + infer(model, wordImg)
    return words


def main():
    """reads images from data/ and outputs the word-segmentation to out/"""
    fnCharList = 'model/charList.txt'
    decoderType = DecoderType.BestPath
    model = Model(open(fnCharList).read(), decoderType, mustRestore=True)
    img = cv2.imread('data/test2.jpg',  cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, int(0.7*(img.shape[1]))), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
    lines = ""
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = img[y:y + h, x:x + w]
        lines = lines +"\n" + doWords(roi, model)
    print(lines)


if __name__ == '__main__':
    main()
