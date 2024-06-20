import numpy as np
from bisect import insort
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


def get_perspective_transform_coeffs(src, dst):
    in_matrix = []
    for (X, Y), (x, y) in zip(src, dst):
        in_matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    a = np.matrix(in_matrix, dtype=np.float64)
    b = np.array(src).reshape(8)
    af = np.dot(np.linalg.inv(a.T * a) * a.T, b)

    return np.array(af).reshape(8)


class Scene:

    def __init__(self):
        self.objects = []

    def add(self, obj):
        # Insert, objects sorted by ascending z-depth
        insort(self.objects, obj, key=lambda x: x.pos[2])

    def render(self, active_camera):
        # TODO: figure out a solution for z-hierarchy of clipping objects
        out = Image.new('RGBA', active_camera.resolution)

        # TODO: Remove this temp screen center marker
        draw = ImageDraw.Draw(out)
        rx, ry = active_camera.resolution
        draw.ellipse((rx/2-2, ry/2-2, rx/2+2, ry/2+2), fill='white')

        for obj in self.objects[::-1]:  # display according to descending z-order
            layer = obj.draw(active_camera)
            out = Image.alpha_composite(out, layer)

        return out


class VirtualCamera:
    #  Camera projection matrix explanations:
    #  https://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
    def __init__(self, resolution, focal_length, sensor_width):
        self.resolution = resolution
        self.ar = self.resolution[0] / self.resolution[1]  # Camera aspect ratio
        self.f = focal_length
        self.s = (sensor_width, sensor_width / self.ar)  # Sensor dimensions (SI units)
        self.fov = 2 * np.arctan(sensor_width / (2 * focal_length))  # Camera FOV

        sx, sy = self.s
        self.k_matrix = np.matrix([  # camera intrinsic matrix
            [focal_length, 0, -sx/2],   # f, 0, px
            [0, focal_length, -sy/2],   # 0, f, py
            [0, 0, 1]                   # 0, 0, 1
        ])

        self._pos = np.zeros(3)
        self._rot = np.identity(3)  # Rotation matrix

        self.p_matrix = self.get_projection_matrix()

    def get_projection_matrix(self):
        p = self.k_matrix * self._rot * np.c_[np.identity(3), - self._pos]

        return p

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, position):
        self._pos = np.array(position)
        self.p_matrix = self.get_projection_matrix()

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, angles):
        self._rot = Rotation.from_euler('xyz', angles).as_matrix()
        self.p_matrix = self.get_projection_matrix()

    def project(self, vertex):
        # Source: https://en.wikipedia.org/wiki/3D_projection
        h_coords = np.r_[vertex, 1].reshape((4, 1))  # Get world homogenous coordinates
        fx, fy, fw = np.asarray(self.p_matrix * h_coords).flatten()  # Get normalized image homogenous coords

        rx, ry = self.resolution  # Grab the resolution
        sx, sy = self.s  # Grab sensor dimensions
        # Transform homogenous coords to screen coords
        # Scale normalized coords to sensor size then to pixel resolution
        bx = -(fx/fw) * (rx/sx)
        by = -(fy/fw) * (ry/sy)

        return np.array((bx, by))  # Returns viewing plane coords


class Object:

    def __init__(self, vertices):
        self._pos = np.zeros(3)
        self._rot = Rotation.from_matrix(np.identity(3))  # Rotation matrix
        self._scale = np.ones(3)

        self.loc_verts = np.array(vertices)  # Make sure the vertices iterable is an array
        self.vertices = self.get_vertices()

    def get_vertices(self):
        """
        Calculate every vertex's world coordinates.
        Only called when position, rotation or scale are changed.
        :return: coords
        """
        # Scale in LOCAL COORDS
        verts = self.loc_verts * self._scale
        # Rotate these around LOCAL ORIGIN (haven't applied global translation yet)
        verts = self._rot.apply(verts)
        # Translate
        verts += self._pos

        return verts

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, position):
        self._pos = np.array(position)
        self.vertices = self.get_vertices()

    @property
    def rot(self):
        return self._rot

    @rot.setter
    def rot(self, angles):
        self._rot = Rotation.from_euler('xyz', angles)
        self.vertices = self.get_vertices()

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scales):
        self._scale = np.array(scales)
        self.vertices = self.get_vertices()


class Reticle:

    def __init__(self):
        self.pos = np.zeros(3)

    def draw(self, active_cam):
        layer = Image.new('RGBA', active_cam.resolution)
        drawer = ImageDraw.Draw(layer)

        cx, cy = active_cam.project(self.pos)
        drawer.ellipse((cx-10, cy-10, cx+10, cy+10), outline='white')

        return layer

    @property
    def depth(self):
        return self.pos[2]

    @depth.setter
    def depth(self, value):
        self.pos[2] = value


class Axes(Object):
    
    def __init__(self):
        super().__init__(np.identity(3))

    def draw(self, active_cam):
        """
        Draw this object as seen by the active camera
        :param active_cam:
        :return:
        """
        layer = Image.new('RGBA', active_cam.resolution)
        drawer = ImageDraw.Draw(layer)

        loc_org = tuple(active_cam.project(self._pos))  # Grab axes coords in screen space
        colors = ('red', 'green', 'blue')
        # Draw avery axis
        for i in range(len(self.vertices)):
            col = colors[i]
            point = tuple(active_cam.project(self.vertices[i]))  # Project to screen space
            drawer.line((loc_org, point), fill=col, width=1)

        return layer
    

class WireSquare(Object):

    def __init__(self):
        verts = np.array([
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, 0]
        ], dtype=np.float_)
        super().__init__(verts)

    def draw(self, active_cam):
        """
        Draw this object as seen by the active camera
        :param active_cam:
        :return:
        """
        layer = Image.new('RGBA', active_cam.resolution)
        drawer = ImageDraw.Draw(layer)

        points = []  # Screen space points
        for v in self.vertices:
            points.append(tuple(active_cam.project(v)))  # Project to screen space

        drawer.polygon(points, outline='white')

        return layer


# TODO: switch perspective transfrom to ImageOps.deform
class ImagePlane(Object):

    def __init__(self, image):
        w, h = image.size
        verts = np.array([
            [w, h, 0],
            [-w, h, 0],
            [-w, -h, 0],
            [w, -h, 0]
        ], dtype=np.float_)
        verts /= max(w, h)
        super().__init__(verts)
        self.image = image

    def draw(self, active_cam):
        w, h = self.image.size
        src_points = [(0, 0), (w, 0), (w, h), (0, h)]
        dst_points = []
        for v in self.vertices:
            dst_points.append(tuple(active_cam.project(v)))

        coeffs = get_perspective_transform_coeffs(src_points, dst_points)

        layer = self.image.transform(active_cam.resolution, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
        layer = layer.convert('RGBA')

        return layer
