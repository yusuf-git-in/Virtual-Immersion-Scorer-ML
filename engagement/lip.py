"""_summary_

    Standalone Module, Graphs Values, Calculates Engagement Level, and Average Engagement Level %

"""

# Threshold Model

# Ranges: 
# [-90, 90]
# [0, 1]
# [-90, 90]
# [0, 100]

head_pose = 0 
eye_asp_ratio = 0
eye_dir = 0
lip_dist = 0

engagement_level = 0

# Emotion Points

arr = {
"Angry": -3,
"Fear": -2,
"Disgust": -1,
"Neutral": 0,
"Sad": 1,
"Happy": 2,
"Surprise": 3
}

def calc_delta_new(head_pose, eye_asp_ratio, eye_dir, lip_dist, emotion):
    """_summary_

    Args:
        head_pose (int): in degrees
        eye_asp_ratio (float): _description_
        eye_dir (int): in degrees
        lip_dist (float): _description_
        emotion (string): _description_
    Returns:
        float: engagement level %
    """
    
    val = 0
    
    if head_pose == 'forward':
        h = 1
    else:
        h = 0

    if eye_dir == "forward":
        ed = 1
    else:
        ed = 1 - 0.5

    if eye_asp_ratio == 'active':
        e = 100
    else:
        e = 25
    
    if lip_dist == 'talking':
        ld = 35
    elif lip_dist == 'yawning':
        ld = 25
    else:
        ld = 0

    val = h * ed * (e - ld + (arr[emotion] * 2.5))
    
    return val

def calc_delta(head_pose, eye_asp_ratio, eye_dir, lip_dist, emotion):
    """_summary_

    Args:
        head_pose (int): in degrees
        eye_asp_ratio (float): _description_
        eye_dir (int): in degrees
        lip_dist (float): _description_

    Returns:
        float: engagement level %
    """
    
    val = 0
    
    # if -20<= head_pose <= 20:
    #     h = 1
    # else:
    #     h = 0

    if head_pose == 'forward':
        h = 1
    else:
        h = 0

    if eye_dir == "center":
        ed = 1
    else:
        ed = 1 - 0.5

    if eye_asp_ratio == 'active':
        e = 100
    else:
        e = 25

    # if eye_asp_ratio > 0.25:
    #     e = 100
    # else:
    #     e = eye_asp_ratio/0.25 * 100
    
    # if -20<= eye_dir <= 20:
    #     ed = 1
    # else:
    #     ed = 1 - abs(eye_dir/360)

    val = h * ed * (e - lip_dist + (emotion * 2.5))
    
    return val

import string
import matplotlib.pyplot as plt

import random

STANDALONE = True

if STANDALONE:
    N = 12
    x = range(0, N)
    y = []

    # naming the x axis
    plt.xlabel('Time in minutes')
    # naming the y axis
    plt.ylabel('Concentration')


    # Test Values
    TEST_head_pose = [10, 10, 10, 20, 10, 40, 40, 40, 40, 0, -10, -40]
    TEST_eye_asp_ratio = [0, 0, 1, 0, 0, 0, 0.1, 0.25, 1, 1.5, 0.1, 0.25]
    TEST_eye_dir = [10, 20, 90, 20, 10, 40, 50, 40, 40, 0, -10, -40]
    TEST_lip_dist = [0, 1, 0, 1, 0, 1, 1.2, 0, 0.25, 0.9, 1, 2]

    TEST_emotion = ["Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Neutral", "Happy", "Happy", "Happy", "Happy", "Angry", "Angry", "Neutral"]

    TEST_VAL = True

    for i in range(0, N):
        
        inf = random.choice(list(arr.keys()))

        # Ranges: 
        # [-90, 90]
        # [0, 1]
        # [-90, 90]
        # [0, 100]

        # Random

        if not TEST_VAL:
            
            head_pose = random.randint(-90, 90)
            eye_asp_ratio = random.random()
            eye_dir = random.randint(-90, 90)
            lip_dist = random.random() * 100

            emotion = arr[TEST_emotion[i]]
        else:

            # Test Vals
        
            head_pose = TEST_head_pose[i]
            eye_asp_ratio = TEST_eye_asp_ratio[i]
            eye_dir = TEST_eye_dir[i]
            lip_dist = TEST_lip_dist[i]

            emotion = arr[TEST_emotion[i]]

        s = calc_delta(head_pose, eye_asp_ratio, eye_dir, lip_dist, emotion)

        y.append(s)

    import statistics

    avg = statistics.mean(y)
    msg = "AVG Engagement level is: " + str(avg) + "%"

    plt.text(0, 90, msg, fontsize=12)
    plt.plot(x, y)

    plt.savefig("mygraph.png")

    plt.show()