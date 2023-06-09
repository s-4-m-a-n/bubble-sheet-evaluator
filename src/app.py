import cv2 as cv 
import imutils
import pandas as pd
# from . import engine
from .engine import pipeline, detect_choice, get_x_y_spacing, preprocessing
import math
import pandas as pd
import click
import os

def eval_result(answers, correct_answers):
    total_questions = correct_answers.shape[0]
    matched_count = 0

    for detected, correct in zip(answers, correct_answers["correct answers"]):
        if detected == correct:
            matched_count += 1
    return {"total_questions": total_questions, 
            "obtained marks" : matched_count}


def parse_answers(groups, binary_image, rect_w, rect_h, total_questions):
    len_column = len(groups)
    num_column = math.ceil(total_questions/len_column)
    answers = []
    for column in range(0, num_column):
        for question_no in range(0, len(groups)):
            answers.append(detect_choice(groups[question_no][column], binary_image, rect_w, rect_h))
    return answers


def batch_run(sheets_dir, ebpg, total_questions, correct_answers):
    output = []

    for file in os.listdir(sheets_dir):
        if file.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(sheets_dir, file)
            result = evaluate_single_sheet(file_path, ebpg, total_questions, correct_answers)
            output.append((file.split(".")[0], result["obtained marks"]))
    return output

def evaluate_single_sheet(sheet_path, ebpg, total_questions, correct_answers):
    # img = cv.imread("../data/answer_sheet_filled_2.jpg")
    img = cv.imread(sheet_path)
    img = imutils.resize(img, width=500)
    groups = pipeline(img.copy(), ebpg)
    binary_image = preprocessing(img.copy())
    rect_w, rect_h = get_x_y_spacing(groups[:2])
    answers = parse_answers(groups, binary_image, rect_w, rect_h, total_questions)
    result = eval_result(answers, correct_answers)
    return result

@click.command()
@click.option('--gt', help='path to the correct answers (ground truth) csv')
@click.option('--ebpg',default=4, help="number of bubbles per question")
@click.option('--s', help="path to the answer sheet/s to be evaluated")
@click.option("--o", default="result",  help="output file name")
def run(gt, s, ebpg, o):
    # correct_answers = pd.read_csv("../notebook/correct_answers.csv")
    correct_answers = pd.read_csv(gt)
    total_questions = len(correct_answers)

    if os.path.isfile(s):
        print(evaluate_single_sheet(s, ebpg, total_questions, correct_answers))
    else:
        output = batch_run(s, ebpg, total_questions, correct_answers)
        df=pd.DataFrame(output, columns=["name", "obtained marks"])
        os.makedirs("output", exist_ok=True) 
        df.to_csv(f"output/{o}.csv")
        print("evaluation complete")



if __name__ == "__main__":
    run()
