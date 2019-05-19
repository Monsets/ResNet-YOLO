def iou(rect1, rect2):
    '''calculates iou over two rectangels
    params:
      rect1 - tuple containing 4 coords of first rectangle: x, y, width, height
      rect2 - tuple containing 4 coords of second rectangle: x, y, width, height
    return:
      % of Intersection which lies between 0 and 1.
    '''

    assert len(rect1) == 4 and len(rect2) == 4, 'Parameters should be two tuples: (x, y, width, height)'

    # left corner of intersection rectangle
    l_x = max(rect1[0], rect2[0])
    l_y = max(rect1[1], rect2[1])
    # right corner
    r_x = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    r_y = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

    if r_x < l_x or r_y < l_y:
        return 0

    # area of intersection
    s_i = (r_x - l_x) * (r_y - l_y)

    # area of rectangles
    s_rect1 = rect1[2] * rect1[3]
    s_rect2 = rect2[2] * rect2[3]

    iou = s_i / (s_rect1 + s_rect2 - s_i)

    return iou


def area_in_rect(rect1, rect2):
    '''calculates percentage of first rect's area which lies within second rect
    params:
      rect1 - tuple containing 4 coords of first rectangle: x, y, width, height
      rect2 - tuple containing 4 coords of second rectangle: x, y, width, height
    return:
      % of area (between 0 and 1).
    '''

    assert len(rect1) == 4 and len(rect2) == 4, 'Parameters should be two tuples: (x, y, width, height)'

    # left corner of intersection rectangle
    l_x = max(rect1[0], rect2[0])
    l_y = max(rect1[1], rect2[1])
    # right corner
    r_x = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    r_y = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

    if r_x < l_x or r_y < l_y:
        return 0

    # area of intersection
    s_i = (r_x - l_x) * (r_y - l_y)

    # area of rectangles
    s_rect1 = rect1[2] * rect1[3]

    perc_of_area = s_i / s_rect1

    return perc_of_area
