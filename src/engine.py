import cv2 as cv
import numpy as np 
import imutils
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from statistics import median
from collections import defaultdict
import pandas as pd


def preprocessing(img):
    # colored image to gray scale image
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # blurring the image
    
    gauss_image = cv.GaussianBlur(gray_image, (5,5,), 0)
    
    # erosion
#     kernel = np.ones((1,1), np.uint8)      
#     close_image = cv.morphologyEx(gauss_image, cv.MORPH_CLOSE, kernel)
    
    kernel = np.ones((3,3), np.uint8)      
    close_image = cv.morphologyEx(gauss_image, cv.MORPH_OPEN, kernel)
    
    #opening -> erosion + dilation
    #closing -> dilation + erosion
    
    #close_image = cv.dilate(close_image, kernel, iterations=1)
    # noise filter
    median_blur= cv.medianBlur(close_image, 1)
    
    # binary image
    # blockSize = 11; Size of a pixel neighborhood that is used to calculate a threshold value for the pixel,
    # constant = 2 subtracted from mean or weighted mean
    binary_image =  cv.adaptiveThreshold(median_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2 )

    #otsu thresholding
#     ret3,binary_image = cv.threshold(median_blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return binary_image


def bubble_contour_extractor(contours):
    contours_features = []
    for cnt in contours:
        if len(cnt) > 3:
            area = cv.contourArea(cnt)
            length = len(cnt)
            parimeter =  cv.arcLength(cnt,False)
            contours_features.append([area, length, parimeter]) 
    contours_features = np.array(contours_features)
    return contours_features


def generate_histogram(list_, n_bins=4):
    max_value, min_value = max(list_), min(list_)
    bin_size = (max_value - min_value)/n_bins
    bins_ranges = [(min_value+i*bin_size, min_value+(i+1)*bin_size) for i in range(0, n_bins)]
    histogram = {x:0 for x in range(len(bins_ranges))}
    for i in list_:
        for index, bin_ in enumerate(bins_ranges):
            if i < bin_[1] and  i >= bin_[0]:
                histogram[index] += 1
    return histogram, bins_ranges
    

def get_max_bin_thresholds(histogram, bins_ranges):
    hist = dict(sorted(histogram.items(), key=lambda item: item[1], reverse=True))
    return bins_ranges[list(hist.keys())[0]]


def remove_outlier(df, columns=["area"], upper_bound=0.99, lower_bound=0.01, _type="drop"):
    if _type == "clip":
        for col in columns:
            df[col] = df[col].clip(
                        lower=df[col].quantile(lower_bound),
                        upper=df[col].quantile(upper_bound)
                        )
    elif _type == "drop":
        for col in columns:
            df_temp = df[df[col] < df[col].quantile(upper_bound)]
            df = df_temp[df_temp[col] > df_temp[col].quantile(lower_bound)]  
    else:
        raise Exception("_type takes either 'clip' or 'drop' ")
    return df


def remove_noise(centers, contours, parimeters, tolerance = 5):
    median_parimeter = median(parimeters)
    new_centers = []
    new_contours = []
    for i in range(len(parimeters)):
        if (parimeters[i] < (median_parimeter + tolerance)) and (parimeters[i] > (median_parimeter - tolerance)):
            new_centers.append(centers[i])
            new_contours.append(contours[i])
    return new_centers, new_contours


def groupby_axis_y(bubble_centers, tolerance = 3):
    '''
    parameters:
        * bubble_centers : list of bubble centers 
        * tolerance: an integer that specifies the range of variation for equivalent values
    
    Description:
        This function transforms 1D list of contours into 2D where contours are grouped into rows
        (note that each row may contain corresponding row of more that one columns).
        Grouping is based on Y_coordinate of the bubble center. 
    
    returns:
        * 2D list -> grouped bubbles center
        * min_x, min_y -> minimum x and y coordinate (center of top-left bubble)
        * max_x, max_y -> maximum x and y coordinate
    '''
    groups = defaultdict(list)
    mapper = {}
    # initializing mapper
    x,y = bubble_centers[0]
    mapper[y] = (y-tolerance, y+tolerance)
    groups[y].append((x,y))
    min_x, max_x, min_y, max_y = x, x, y, y
    for x,y in bubble_centers[1:]:
        # grouping bubbles
        flag_found = False
        new_y = None
        
        for k,v in mapper.items():
            new_y = k
            if v[0] < y and v[1] > y:
                groups[k].append((x,new_y))
                flag_found = True
                break
        # if mapper doesnot have y already
        if not flag_found:
            mapper[y] = (y-tolerance, y+tolerance)
            groups[y].append((x,y))
            new_y = y
        # finding min and max x,y
        if x < min_x:
            min_x = x
        if new_y < min_y:
            min_y = new_y
        if x > max_x:
            max_x = x
        if new_y > max_y:
            max_y = new_y
    return list(groups.values()) , [(min_x, min_y), (max_x, max_y)]


def get_avg_x_spacing(groups_y, ebpg):
    '''
    parameters:
        * groups_y: list of row wised grouped bubbles center
        * ebpg (expected bubble per group): integer value that specifies the number of bubbles for a single question
    
    Description:
       Ebpg helps us to determine the number of bubbles in each column of a row.
       This function will check for the perfect rows where ever bubble is detected and there is no noise. 
       If no perfect rows is availabe then halt.
       Else, find the mode of median of spacing of two consecutive bubbles i.e X2 - X1.
       
       It will also calculate x_coordinate of a single perfect row by computing the columnwise mode of all the available perfect rows.
       This will basically determines the expected x_coordinate of each bubbles of each row.
       
    Returns:
        * minimum average x-spacing
        * perfect_row_x: mode of x_coordinates of the bubbles of perfect_rows; axis=0 -> columnwise  
        
    '''
    perfect_rows_x = [] # stores x coordinates of bubbles of perfect rows
    average_x_spacing = []
    
    columns_x = defaultdict(list) # stores the x coordinate of first bubble of each column group
    
    for row in groups_y:
        row = sorted(row)
        spacing  = []
        pointer = 0
        if len(row) % ebpg != 0: # if the row is perfect then when we divide it by the number of bubbles per question the remainder will be zero
            continue
    
        perfect_rows_x.append([x for x,y in row])
        
        for i in range(1, len(row)):
            if (i-1) % ebpg == 0 and (i-1) != 0:
                columns_x[pointer].append(row[i-1][0])
                pointer = (pointer + 1) % ebpg
            
            spacing.append(abs(row[i][0] - row[i-1][0]))
            
        # recording average x spacing of this row
        average_x_spacing.append(statistics.median(spacing))
#         average_x_spacing.append(statistics.mode(spacing))
    
    if len(average_x_spacing) > 0 and len(columns_x.values()) > 0:
        # find mode of perfect_rows column wise
        perfect_row_x = pd.DataFrame(perfect_rows_x).mode(axis=0)
        columns_x_mode = [statistics.mode(X) for X in columns_x.values()]
        return min(average_x_spacing), columns_x_mode, perfect_row_x.values.tolist()[0]
    else:
        return None

    

def groupby_axis_x(groups_y, min_x, max_x, tolerance=3, ebpg=4): #ebpg(expected bubble per group)=expected number of options/bubbles per question
    ''' 
    parameters:
        * groups_y: list of row wised grouped bubbles center
        * min_x, max_x = minimum and maximum x_coordinate 
        * tolerance =  an integer that specifies the range of variation for equivalent values 
        * ebpg (expected bubble per group): integer value that specifies the number of bubbles for a single question
        
    Description:
        It takes list of grouped bubbles (grouped into rows) and segment each rows into multiple columns. 
        For instance; [[1,2,3,4,5,6],[7,8,9,10,11,12]] is grouped into columns as 
                
                [[[1,2,3], [4,5,6]],
                 [[7,8,9], [10,11,12]]]
    Returns:
        * groups: 3D list of grouped bubbles
        * perfect_rows_x: x-coordinate of the bubbles center of a perfect rows 
    '''
    # setting 2D empty bucket
    groups = [[] for i in range(len(groups_y))]
    avg_distance, columns_x, perfect_row_x = get_avg_x_spacing(groups_y, ebpg)
    column_pointer = 0 # initially pointing to the zero th column
    if avg_distance is None: # there is not row that matches the specified ebpg, mean doesnot exist perfect rows (without missing bubble and noises)
        return []
    for i, row in enumerate(groups_y):
        row = sorted(row)
        previous_bubble_center_x = 0
        current_bubble_center_x = 0
        # checking if the first bubble of the row is detected or not
        if row[0][0] > (min_x - tolerance) and row[0][0] < (min_x + tolerance):
            groups[i].append([(row[0])])
            current_bubble_center_x = row[0][0]
        else:
            groups[i].append([(min_x, None)])
            current_bubble_center_x = min_x
        
        # now checkig for the rest of the bubble of the row
        bubble_index = 1
        while current_bubble_center_x < (max_x-tolerance):
                
            if len(row) > bubble_index and\
                   (abs(row[bubble_index][0] - current_bubble_center_x) < (avg_distance + tolerance) and\
                   abs(row[bubble_index][0] - current_bubble_center_x) > (avg_distance - tolerance)) :

                    # if one options set is covered then add list for new col group
                    groups[i][len(groups[i])-1].append(row[bubble_index])
                    previous_bubble_center_x = current_bubble_center_x
                    current_bubble_center_x = row[bubble_index][0]
                    bubble_index += 1
                    
                    
            # there is a missing bubble (unable to detect bubble)
            else:
                if len(groups[i][len(groups[i]) - 1]) % ebpg == 0: 

                    if len(row) > bubble_index and (row[bubble_index][0] < (columns_x[column_pointer] + tolerance)) and\
                    (row[bubble_index][0] > (columns_x[column_pointer] - tolerance)):
                        
                        groups[i].append([(columns_x[column_pointer], row[bubble_index][1])])
                        bubble_index += 1
                    
                    else:
                        groups[i].append([(columns_x[column_pointer], None)])
                        
                    previous_bubble_center_x = current_bubble_center_x
                    current_bubble_center_x = columns_x[column_pointer]
                    column_pointer = (column_pointer + 1) % len(columns_x)
                        
                else:
                    groups[i][len(groups[i]) - 1].append((None, None))
                    # last bubble of the group + average distance
                    previous_bubble_center_x = current_bubble_center_x
                    current_bubble_center_x = current_bubble_center_x + avg_distance
    return groups, perfect_row_x
    
    
def estimate_missing_bubbles(groups, perfect_row_x, ebpg=4): #consecutive bubble distance, row distance, coln distance
    '''
    Parameters: 
        * groups : 3D list of grouped bubbles obtained from groupby_axis_x function
        * perfect_row_x: x-coordinate of the bubbles of a perfect rows obtained from groupby_axis_x_function
    
    Description:
        it will replace the missing values in the groups by estimation the possible coordinate value for the missing bubbles
    
    Returns:
        * groups: 3D list of grouped bubbles where missing values are replaced with the estimated possible values
    '''
    for row_index, row in enumerate(groups):
        flat_row = [y for col in row for x,y in col]
        for col_index, col in enumerate(row):
            # flattern row
            for bubble_index, bubble in enumerate(col):
                # if x coordinate is missing
                if bubble[0] == None:
                    # convert non liner indexing to linear 
                    linear_index = col_index * ebpg + bubble_index
                    
                    new_x = perfect_row_x[linear_index]
                    old_y = groups[row_index][col_index][bubble_index][1]
                    groups[row_index][col_index][bubble_index] = (new_x, old_y)
                    
                # if y coordinate is missing
                if bubble[1] == None:
                    # find not none y or bubble[1]
                    unique_y = list(set(flat_row))

                    new_y = unique_y[0] if unique_y[0] else unique_y[len(unique_y) - 1]
                    old_x = groups[row_index][col_index][bubble_index][0]

                    groups[row_index][col_index][bubble_index] = (old_x, new_y)
    
    return groups

def get_x_y_spacing(groups, margin=5): # margin is the gap betwen two consecutive bubbles
    '''
    Parameters:
        * groups: groups obtained from estimate_missing_bubbles function
    
    Description:
        It will return the half of the distance between two consecutive bubbles (both X axis and Y axis)
    
    Returns:
        * h_spacing : half of the distance between two consecutive bubbles of a row 1
        * v_spacing : half of the distance between the first bubbles of two consecutive rows 
    '''
    x_spacing = abs(groups[0][0][1][0] - groups[0][0][0][0]) 
    y_spacing = abs(groups[1][0][1][1] - groups[0][0][0][1])
    return int((x_spacing - margin)//2), int((y_spacing - margin)//2)


def extract_bubbles_center(contours, lower_bound, upper_bound):
    '''
    returns the centerX and centerY of the bubble contours 
    from the given list of contours and the lower and upper
    threshold values of the corresponding bin of the contours
    '''
    bubbles_contour = []
    bubbles_center = []
    bubbles_parimeter = []
    for ctr in contours:
        area = cv.contourArea(ctr)
        area_condition = area < upper_bound and area > lower_bound
        if area_condition:
            # computing center of contour
            M = cv.moments(ctr)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                bubbles_contour.append(ctr)
                bubbles_parimeter.append(cv.arcLength(ctr,True))

                bubbles_center.append((cX, cY))
                
    return bubbles_contour, bubbles_center, bubbles_parimeter


def pipeline(image, ebpg=4):
    binary_image = preprocessing(image.copy())
    contours = cv.findContours(binary_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    ctrs = imutils.grab_contours(contours)
    
    contours_features = bubble_contour_extractor(ctrs)
    df = pd.DataFrame(contours_features, columns=["area", "length", "parimeter"])
    df = df[(df["area"] < 500) & (df["area"] > 100) & (df["length"] > 5) & (df["length"] < 20)]
    histogram, bins_ranges = generate_histogram(df["area"], n_bins=3)
    
    lower_bound, upper_bound = get_max_bin_thresholds(histogram, bins_ranges)
    
    bubbles_contour, bubbles_center, bubbles_parimeter  = extract_bubbles_center(ctrs,
                                                                                 lower_bound,
                                                                                 upper_bound)
    
    bubbles_center, bubbles_contour = remove_noise(bubbles_center,
                                                   bubbles_contour,
                                                   bubbles_parimeter)
    
#     contoured_image = draw_contour(image, bubbles_contour, lower_bound, upper_bound) 
#     plt.figure(figsize=(10,10))
#     plt.imshow(contoured_image, cmap="gray")
#     plt.show()
#     return
    
    groups_y , min_max_bubble_center = groupby_axis_y(bubbles_center, tolerance=6)
    #-----------------------------------------------
    min_x = min_max_bubble_center[0][0]
    max_x = min_max_bubble_center[1][0]
    bubble_grouped, perfect_row_x_coords = groupby_axis_x(groups_y,
                                                          min_x,
                                                          max_x,
                                                          ebpg=ebpg,
                                                          tolerance=7)
    
    groups = estimate_missing_bubbles(bubble_grouped,
                                      perfect_row_x_coords,
                                      ebpg=ebpg)
    # we need to reverse groups 
    groups = list(reversed(groups))
    return groups


def compute_proportion_of_black(image): #image is an binary image
    '''
    Parameter:
        *image : takes an binarized image
        
    Description:
        It will calculate the percentage of black (0) pixels in that image.
    
    Returns:
        * proportion of black pixels
    '''
    # convert pixel value 255 into 1
    pixel_ones_count = (np.array(image)%254).sum()
    total_pixel = image.shape[0] * image.shape[1]

    # calculating the perportion of black in an image
    proportion_of_zeros = (total_pixel - pixel_ones_count)/total_pixel 
    return proportion_of_zeros

def crop_area(image):
    '''
    parameter: 
        *image: cropped image around the rectangular bounding box around a bubble
        
    Description:
        It will further cropped the image just to include the bubble, so that we will get more accurate result.
    
    Returns:
        *cropped image in a numpy array formate.
    '''
    new_rows = []
    # crop row
    for row in np.array(image)%254:
        sum_ = row.sum()
        if sum_ != len(row):
            new_rows.append(list(row))
    # crop columns
    df = pd.DataFrame(new_rows)
    for column in df.columns:
        if sum(df[column]) == len(df[column]):
            df.drop(column, axis=1, inplace=True)
    return np.array(df)

def detect_choice(question_bubble, image, rect_w, rect_h, black_percentage=0.65):
    '''
    Parameters:
        * question_bubble: list of center of bubbles of a particular question
        * image: binarized image of the bubble answer sheet
        * rect_w: width of a rectangular bounding box
        * rect_h: height of a rectangular bounding box
        * black_percentage: minimum percentage of black pixels for a chosen option
    
    Details:
        It will check each bubbles of a question and evaluate the proportion of black pixels in it, 
        based on that value it will determine which option is chosen and also accounts the errors. 
    
    Returns:
        * index of chosen option or -1 if no option is chosen or multiple options are chosen.
        
    '''
    choice = []
    for index, bubble in enumerate(question_bubble):

        bubble_area = image[round(bubble[1]- 12): round(bubble[1]+ 12),
                            round(bubble[0] - 12): round(bubble[0] + 12)]
        bubble_area = crop_area(bubble_area)
#         plt.imshow(bubble_area, cmap='gray')
#         plt.show()
        proportion_of_zeros = compute_proportion_of_black(bubble_area)
        if proportion_of_zeros >= black_percentage:
            choice.append(index+1)
    if len(choice) == 1:
        return choice[0]
    else:
        return -1