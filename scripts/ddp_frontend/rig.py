"""Camera rig and input conversion of the DDP / CAA-Policy data (no carla import)."""
import numpy as np
import torch
from PIL import Image

CAMERAS = ('camera_front', 'camera_front_left', 'camera_front_right',
           'camera_back', 'camera_back_left', 'camera_back_right')
# x, y, z, yaw from config/sensor_specs.yaml in ai4ce/CAAPolicy.
MOUNTS = dict(camera_front=(2.36, 0., 1.5, 0.), camera_front_left=(2.36, -.792, 1.5, -55.),
              camera_front_right=(2.36, .792, 1.5, 55.), camera_back=(-2.36, 0., 1.55, -180.),
              camera_back_left=(-2.36, -.792, 1.55, -110.), camera_back_right=(-2.36, .792, 1.55, 110.))
WIDTH, HEIGHT, FOV, CROP = 400, 300, 100., 294
MEAN, STD = np.array([.485, .456, .406], np.float32), np.array([.229, .224, .225], np.float32)
CAM_FROM_VEHICLE_AXES = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], float)


def carla_matrix(x=0., y=0., z=0., yaw=0., pitch=0., roll=0.):
    """carla.Transform.get_matrix. Angles in degrees."""
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    return np.array([[cp*cy, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr, x],
                     [cp*sy, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr, y],
                     [sp, -cp*sr, cp*cr, z], [0, 0, 0, 1]])


def intrinsics():
    focal = WIDTH/(2*np.tan(np.radians(FOV)/2))
    matrix = np.array([[focal, 0, WIDTH/2-(WIDTH-CROP)/2], [0, focal, HEIGHT/2-(HEIGHT-CROP)/2], [0, 0, 1]])
    return torch.tensor(matrix, dtype=torch.float32).expand(len(CAMERAS), 3, 3).clone()


def extrinsics():
    """Vehicle to camera, in the axis order the depth frustum uses."""
    matrices = []
    for name in CAMERAS:
        x, y, z, yaw = MOUNTS[name]
        matrices.append(CAM_FROM_VEHICLE_AXES@np.linalg.inv(carla_matrix(x, y, z, yaw)))
    return torch.tensor(np.stack(matrices), dtype=torch.float32)


def prepare_image(image):
    """Centre crop to 294 and normalise. Returns [3, 294, 294]."""
    array = np.asarray(image.convert('RGB') if isinstance(image, Image.Image) else image)
    top, left = HEIGHT//2-CROP//2, WIDTH//2-CROP//2
    array = array[top:top+CROP, left:left+CROP].astype(np.float32)/255
    return torch.from_numpy(((array-MEAN)/STD).transpose(2, 0, 1).copy())


def target_in_ego(measurement, goal):
    """Parking goal as (x, y, yaw deg) in the current vehicle frame."""
    pose = carla_matrix(measurement['x'], measurement['y'], measurement['z'], measurement['yaw'],
                        measurement['pitch'], measurement['roll'])
    x, y = (np.linalg.inv(pose)@np.array([goal['x'], goal['y'], 1., 1.]))[:2]
    yaw = goal['yaw']-measurement['yaw']
    yaw = yaw-360 if yaw > 180 else yaw+360 if yaw < -180 else yaw
    return torch.tensor([x, y, yaw], dtype=torch.float32)


def segmentation_labels(path, target):
    """`topdown` image to BEV labels: 1 vehicle, 2 target slot, as in CAA-Policy."""
    image = Image.open(path).convert('L')
    width, height = int(image.width*2), int(image.height*2)
    array = np.asarray(image.resize((width, height), Image.NEAREST))
    top, left = height//2-100, width//2-100
    array = array[top:top+200, left:left+200].copy()
    centre = np.array([100-target[0]/.1, 100+target[1]/.1], int)
    slot = np.stack(np.meshgrid(np.arange(-27, 28), np.arange(-15, 16), indexing='ij'), -1).reshape(-1, 2).T
    points = (carla_matrix(yaw=float(-target[2]))[:2, :2]@slot).astype(int)+centre[:, None]
    inside = ((points >= 0) & (points < 200)).all(0)
    array[points[0, inside], points[1, inside]] = 255
    labels = np.zeros((200, 200), int)
    labels[array == 75], labels[array == 255] = 1, 2
    return labels[::-1].copy()


def iou(prediction, labels, value):
    union = ((prediction == value) | (labels == value)).sum()
    return ((prediction == value) & (labels == value)).sum()/union if union else np.nan
