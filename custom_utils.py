import math


def get_points_within_distance(new_points: list[int], prev_points: list[dict[str, tuple[int, int] | int]]):
    """
        only used in demo_2.py
    """

    radius = 20  # pixels
    res_points = []
    existing_ids = []

    for prev_point in prev_points:
        for new_point in new_points:
            # print(prev_point)
            new_point_x = int(new_point[0])
            new_point_y = int(new_point[1])
            prev_point_x = int(prev_point["coord"][0])
            prev_point_y = int(prev_point["coord"][1])
            if abs(math.hypot(new_point_x - prev_point_x, new_point_y - prev_point_y)) <= radius:
                if not prev_point["id"] in existing_ids:
                    existing_ids.append(prev_point["id"])
                    res_points.append({"id": prev_point["id"], "coord": new_point})

    return res_points
