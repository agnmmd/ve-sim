import traci
import math

class TraciAnnotation:
    def __init__(self):
        # Initialize the TraCI connection with SUMO
        self.shapes = []  # To store the shapes (rectangles and circles)

        if not traci.isLoaded():
            raise ValueError('TraCI is not loaded!')

    def add_rectangle(self, rect_id, bottom_left, top_right, color=(255, 0, 0, 255), layer=0, fill=False, lineWidth=1.0):
        """Add a rectangle using the bottom-left and top-right coordinates."""
        # Calculate the other two points (top-left and bottom-right)
        bottom_right = (top_right[0], bottom_left[1])
        top_left = (bottom_left[0], top_right[1])

        # Create the list of 4 corner points in clockwise order, ensuring to close the rectangle
        rectangle_points = [bottom_left, top_left, top_right, bottom_right, bottom_left]

        # Store the rectangle in the shapes list
        rectangle = {
            "type": "rectangle",
            "id": rect_id,
            "points": rectangle_points,
            "color": color,
            "layer": layer,
            "fill": fill,
            "lineWidth": lineWidth
        }
        self.shapes.append(rectangle)

    def add_circle(self, circle_id, center, radius, color=(0, 0, 255, 255), layer=0, fill=False, lineWidth=1.0):
        """Add a circle to the list of shapes."""
        # A circle is represented as a polygon with many points forming the shape
        num_segments = 20  # Number of points to approximate the circle
        points = [
            (center[0] + radius * math.cos(2 * math.pi * i / num_segments),
             center[1] + radius * math.sin(2 * math.pi * i / num_segments))
            for i in range(num_segments)
        ]
        # Close the circle by appending the first point to the end
        points.append(points[0])

        circle = {
            "type": "circle",
            "id": circle_id,
            "points": points,
            "color": color,
            "layer": layer,
            "fill": fill,
            "lineWidth": lineWidth
        }
        self.shapes.append(circle)

    def draw_shapes(self):
        """Draw all shapes (rectangles and circles) in SUMO."""
        for shape in self.shapes:
            traci.polygon.add(shape['id'], shape['points'], color=shape['color'], fill=shape['fill'],
                              layer=shape['layer'], lineWidth=shape['lineWidth'])