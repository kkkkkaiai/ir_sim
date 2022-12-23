import numpy as np
from math import sqrt, pi, cos, sin
from collections import namedtuple

# collision detection

## circle:  x, y, r
## segment: [point1, point2]
## point: x, y
## point_set: 2*n, matrix 
## rectangle: 
## polygon: [point1, point2,....]  convex

## collision state: the position of the left object shoud be when collision with the right object

circle_geometry = namedtuple('circle', ['x', 'y', 'r'])

# collision between circle and circle
def collision_cir_cir(circle1, circle2):
    collision_flag = False
    collision_state = None

    dis = sqrt( (circle2.x - circle1.x)**2 + (circle2.y - circle1.y)**2 )

    if (dis - (circle1.r + circle2.r)) < 0:
        collision_flag = True
        unit_diff = np.array([[circle1.x - circle2.x], [circle1.y - circle2.y]]) / dis

        collision_state = np.array([[circle2.x], [circle2.y]]) + unit_diff * (circle1.r + circle2.r)

    return collision_flag, collision_state

# collision between circle and polygon
def collision_cir_poly(circle, polygon):
    polygon.append(polygon[0])

    collision_position = None
    collision_flag = False

    for i in range(len(polygon)-1):
        segment = [ polygon[i], polygon[i+1] ]
        collision_flag, collision_position = collision_cir_seg(circle, segment) 

        if collision_flag:
            if i+1 < (len(polygon)-1):
                new_segment =  [ polygon[i+1], polygon[i+2] ]
                new_circle = circle_geometry(collision_position[0, 0], collision_position[1, 0], circle.r)  

                new_collision_flag, new_collision_position = collision_cir_seg(new_circle, new_segment) 

                if new_collision_flag:
                    return new_collision_flag, new_collision_position
                else:
                    return collision_flag, collision_position
            
    return collision_flag, collision_position

def collision_poly_cir(polygon, circle):
    polygon.append(polygon[0])

    for i in range(len(polygon)-1):
        segment = [ polygon[i], polygon[i+1] ]
        if collision_seg_cir(segment, circle): 
            return True
    
    return False

# collision between polygon and polygon
def collision_poly_poly(polygon1, polygon2):

    polygon1.append(polygon1[0])
    polygon2.append(polygon2[0])

    for i in range(len(polygon1)-1):
        segment1 = [ polygon1[i], polygon1[i+1] ]
        for j in range(len(polygon2)-1):
            segment2 = [ polygon2[j], polygon2[j+1] ]
            if collision_seg_seg(segment1, segment2): return True
    
    return False

# collision between circle and segment
def collision_cir_seg(circle, segment):

    collision_flag = False
    collision_position = None

    # reference: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    point = np.array([circle.x, circle.y])
    sp = np.array([segment[0].x, segment[0].y])
    ep = np.array([segment[1].x, segment[1].y])
    
    l2 = (ep - sp) @ (ep - sp)

    if (l2 == 0.0): 
        dis = sqrt( (circle.x - segment[0].x)**2 + (circle.y - segment[0].y)**2 ) 
        if dis < circle.r:
            collision_flag = True
            unit_diff = np.array([[circle.x - segment[0].x], [circle.y - segment[0].y]]) / dis
            collision_position = np.array([[segment[0].x], [segment[0].y]]) + unit_diff * circle.r

        return collision_flag, collision_position
    
    t = max(0, min(1, ((point-sp) @ (ep-sp)) / l2 ))

    projection = sp + t * (ep-sp)
    relative = projection - point

    dis = sqrt( relative[0]**2 + relative[1]**2 )

    if dis < circle.r:
        collision_flag = True
        unit_diff = np.array([[circle.x - projection[0]], [circle.y - projection[1]]]) / dis
        collision_position = np.array([[projection[0]], [projection[1]]]) + unit_diff * circle.r

    return collision_flag, collision_position

def collision_seg_cir(segment, circle):
    # reference: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    point = np.array([circle.x, circle.y])
    sp = np.array([segment[0].x, segment[0].y])
    ep = np.array([segment[1].x, segment[1].y])
    
    l2 = (ep - sp) @ (ep - sp)

    if (l2 == 0.0): return sqrt( (circle.x - segment[0].x)**2 + (circle.y - segment[0].y)**2 ) < circle.r
    
    t = max(0, min(1, ((point-sp) @ (ep-sp)) / l2 ))

    projection = sp + t * (ep-sp)
    relative = projection - point

    return sqrt( relative[0]**2 + relative[1]**2 ) < circle.r



# collision between segment and segment
def collision_seg_seg(segment1, segment2):
    # reference https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/; https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    p1, p2, q1, q2 = segment1[0], segment1[1], segment2[0], segment2[1]

    o1 = orientation(p1, q1, q2)
    o2 = orientation(p2, q1, q2)
    o3 = orientation(p1, p2, q1)
    o4 = orientation(p1, p2, q2)

    # general case
    if o1 != o2 and o3 != o4:
        return True

    # special case
    if o1 == 0 and onSegment(p1, q1, p2):
            return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0 and onSegment(p1, q2, p2)):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0 and onSegment(q1, p1, q2)):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if (o4 == 0 and onSegment(q1, p2, q2)):
        return True

    return False

def onSegment(p, q, r):

    if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
        return True
    
    return False

def orientation(p, q, r):

    # 0 collinear 
    # 1 counterclockwise 
    # 2 clockwise

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))

    if val > 0:
        return 1
    elif val < 0:
        return 2
    else:
        return 0
