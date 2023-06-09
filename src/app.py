import cv2 as cv 
import imutils
import pandas as pd
from engine import pipeline, detect_choice, get_x_y_spacing, preprocessing
import math
import pandas as pd


def eval_result(answers, correct_answers):
    total_questions = correct_answers.shape[0]
    matched_count = 0

    for detected, correct in zip(answers, correct_answers["correct answers"]):
        if detected == correct:
            matched_count += 1
    return {"total_questions": total_questions, 
            "obtained marks" : matched_count}


def parse_answers(groups, binary_image, rect_w, rect_h):
    ebpg = 4
    total_questions = 75
    len_column = len(groups)
    num_column = math.ceil(total_questions/len_column)
    answers = []
    for column in range(0, num_column):
        for question_no in range(0, len(groups)):
            answers.append(detect_choice(groups[question_no][column], binary_image, rect_w, rect_h))
    return answers

def run():
    correct_answers = pd.read_csv("../notebook/correct_answers.csv")
    img = cv.imread("../data/answer_sheet_filled_5.jpg")
    img = imutils.resize(img, width=500)
    groups = pipeline(img.copy())
    binary_image = preprocessing(img.copy())
    rect_w, rect_h = get_x_y_spacing(groups[:2])
    answers = parse_answers(groups, binary_image, rect_w, rect_h)
    print(answers)
    result = eval_result(answers, correct_answers)
    print(result)
    pass

if __name__ == "__main__":
    run()