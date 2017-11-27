# Mike Carter
# Nov 25 2017
# OpenCV scripting for trivia game

# starter code from https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python

import pyscreenshot as ImageGrab
from PIL import Image
import pytesseract
import argparse
import cv2
import os
from nltk.corpus import stopwords
import webbrowser

def ParseArgs():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False,
                    help="path to input image to be OCR'd")
    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
                    help="type of preprocessing to be done")
    args = vars(ap.parse_args())
    return args

def GrabScreen():
    input("Press Enter to screenshot and start...")
    image = ImageGrab.grab(bbox=(1250, 300, 1725, 775))  # X1,Y1,X2,Y2
    #image.show()
    filename = 'screencap.jpg'
    image.save(filename, 'JPEG')
    return filename

def LoadImageAndPreprocess(args):
    if args["image"]:
        # load the example image and convert it to gray-scale
        image = cv2.imread(args["image"])
    else:
        image = cv2.imread(GrabScreen())

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # check to see if we should apply thresholding to pre-process the
    # image
    if args["preprocess"] == "thresh":
        gray = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # make a check to see if median blurring should be done to remove
    # noise
    elif args["preprocess"] == "blur":
        gray = cv2.medianBlur(gray, 3)
    # write the gray-scale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    return filename, image, gray

def ReadText(filename):
    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    return text

def ExtractQuestion(question_raw, s):
    lines = question_raw.split("\n")
    lines = [line for line in lines if len(line)>=2]
    question = " ".join(lines).strip()
    keywords = list(filter(lambda w: not w in s,question.split()))
    return question, keywords

def ExtractAnswers(answers_raw):
    lines = answers_raw.split("\n")
    i = 0
    while len(lines) > 3:
        lines = [line for line in lines if len(line) > i]
        i += 1
    while len(lines) < 3:
        print("\nRead error: answer missed.\n")
        lines.append("")
    return lines

def ParseText(text, s):
    question_raw, answers_raw = text.split("?")
    question, keywords = ExtractQuestion(question_raw, s)
    answers = ExtractAnswers(answers_raw)
    print("\nQuestion:\n" + question + "\n")
    print("\nKeywords:\n" + str(keywords) + "\n")
    print("\nAnswers:\n" + str(answers) + "\n")
    return question, keywords, answers

def DebugOutput(text, image, gray):
    print(text)
    # show the output images
    cv2.imshow("Image", image)
    cv2.imshow("Output", gray)
    cv2.waitKey(0)

def SearchGoogle(question, keywords, answers):
    print("\n Searching google...\n")
    url = "https://www.google.com.tr/search?q={}".format(question)
    #url0 = "https://www.google.com.tr/search?q={}".format(str(answers[0] + " " + question))
    #url1 = "https://www.google.com.tr/search?q={}".format(str(answers[1] + " " + question))
    #url2 = "https://www.google.com.tr/search?q={}".format(str(answers[2] + " " + question))
    webbrowser.open(url, new=1)
    #webbrowser.open(url0, new=1)
    #webbrowser.open(url1, new=1)
    #webbrowser.open(url2, new=1)


# Initialize
s = set(stopwords.words('english'))
args = ParseArgs()
# Run
filename, image, gray = LoadImageAndPreprocess(args)
text = ReadText(filename)
question, keywords, answers = ParseText(text, s)
#DebugOutput(text, image, gray)
SearchGoogle(question, keywords, answers)