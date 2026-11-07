"""Static Town04_Opt parking benchmark of the DDP evaluator, driven by plan_server.

Runs under Python 3.7 with the CARLA 0.9.11 egg. The protocol is that of
carla_parking_eva.py, data_generation/network_evaluator.py, world.py and
agent/parking_agent.py of the DDP release: a fresh map without the ParkedVehicles layer,
16 goal slots (index 16, 18, ... of parking_vehicle_locations_Town04), 6 starts each at
goal y-8..+8, seed 66 plus the slot number for the parked cars, 30 Hz, a 1 s hold, a 30 s
limit, out of bounds for 10 s, success after two stopped reverse frames in the goal, and
the three weathers. The frame order is the one of game_loop: the agent acts on the sensor
frame of the previous tick. The target is the one the DDP agent makes, with the position
dead reckoned at 10 Hz from velocity and IMU and a goal yaw of 180 for every slot.
"""
from __future__ import print_function
import argparse
import csv
import json
from multiprocessing.connection import Client
import math
import os
import queue
import random
import sys
import time
import numpy as np

EGG = os.environ.get('CARLA_EGG', 'carla-0.9.11-py3.7-win-amd64.egg')
if not os.path.isfile(EGG):
    # A missing egg would otherwise fall through to any carla package pip installed.
    sys.exit('CARLA_EGG must point to the CARLA 0.9.11 egg, no such file: {}'.format(EGG))
sys.path.insert(0, EGG)
import carla

CARLA_VERSION = '0.9.11'

CAMERAS = ('camera_front', 'camera_front_left', 'camera_front_right',
           'camera_back', 'camera_back_left', 'camera_back_right')
MOUNTS = dict(camera_front=(2.36, 0., 1.5, 0.), camera_front_left=(2.36, -.792, 1.5, -55.),
              camera_front_right=(2.36, .792, 1.5, 55.), camera_back=(-2.36, 0., 1.55, -180.),
              camera_back_left=(-2.36, -.792, 1.55, -110.), camera_back_right=(-2.36, .792, 1.55, 110.))
ROWS = (298.5, 290.9, 280.0, 272.5)
SLOT_Y = (-235.73, -232.73, -229.53, -226.43, -223.43, -220.23, -217.23, -214.03,
          -210.73, -207.30, -204.23, -201.03, -198.03, -194.90, -191.53, -188.20)
SPOTS = [(x, y) for x in ROWS for y in SLOT_Y]
GOAL_INDICES = list(range(16, 48, 2))
SLOT_NAMES = ['{}-{}'.format(row, index) for row in (2, 3) for index in range(1, 16, 2)]
BOUND = dict(x_min=264., x_max=304., y_min=-241., y_max=-178.)
EGO_X, EGO_Z = 285.600006, .32682
FPS = 30
HOLD_FRAMES, TOTAL_FRAMES, OUT_FRAMES, NEAR_FRAMES, IN_GOAL_FRAMES = 30, 30*FPS, 10*FPS, 2*FPS, 1
X_TOL, Y_TOL, YAW_TOL = 1., .6, 10.
SPOT_Z = .3
PROCESS_FRAMES, TRACK_DT = 3, .1    # the agent reads its sensors at 10 Hz
VELOCITY_POINT = .335       # metres ahead of the actor origin, fitted on 54 Same episodes
# cloudiness, sun azimuth, sun altitude, fog, wetness, precipitation, deposits
WEATHERS = dict(same=(15., 0., 75., 0., 0., 0., 0.), mild=(35., 20., 25., 2., 20., 30., 20.),
                large=(45., 30., 5., 3., 30., 45., 30.))
REAR_OFFSET = 1.5
STEP_SECONDS = .4       # one model step: 4 frames at 10 Hz
# Steering command against driven curvature, measured on e2e_dataset.
CURVATURE = (0., .072, .151, .24, .345, .47)
STEER = (0., .2, .4, .6, .8, 1.)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Tracker:
    """Pure pursuit on the rear axle, with gear changes only at rest."""

    def __init__(self, lookahead=1.2, stop_decel=.8, min_speed=.45):
        self.lookahead, self.stop_decel, self.min_speed = lookahead, stop_decel, min_speed
        self.path = None
        self.gear, self.integral = 1, 0.

    def set_path(self, rear, rows):
        """rows: rear-axle path in the frame of `rear` (x, y, yaw, speed, phase)."""
        rows = np.asarray(rows, float)
        c, s = math.cos(rear[2]), math.sin(rear[2])
        x = rear[0]+c*rows[:, 0]-s*rows[:, 1]
        y = rear[1]+s*rows[:, 0]+c*rows[:, 1]
        self.path = np.stack((x, y), 1)
        self.speed, self.phase = np.abs(rows[:, 3]), rows[:, 4].astype(int)
        self.gears = np.sign(rows[:, 3])
        for phase in set(self.phase.tolist()):
            part = self.phase == phase
            sign = np.sign(self.gears[part].sum()) or 1
            self.gears[part] = sign
        step = np.linalg.norm(np.diff(np.vstack((rear[:2], self.path)), axis=0), axis=1)
        self.arc = np.cumsum(step)
        self.index = 0

    def clear(self):
        self.path = None

    def remaining(self):
        return 0. if self.path is None else float(self.arc[-1]-self.arc[self.index])

    def control(self, rear, speed):
        """Returns a VehicleControl for the rear-axle pose and the unsigned speed in m/s."""
        hold = carla.VehicleControl(throttle=0., brake=1., steer=0., reverse=self.gear < 0)
        if self.path is None:
            return hold
        phase = self.phase[self.index]
        last = int(np.flatnonzero(self.phase == phase)[-1])
        window = np.arange(self.index, min(self.index+60, last+1))
        self.index = int(window[np.argmin(np.linalg.norm(self.path[window]-rear[:2], axis=1))])
        wanted = int(self.gears[self.index])
        left = float(self.arc[last]-self.arc[self.index])
        final_phase = last == len(self.path)-1
        rolling_end = final_phase and self.speed[-1] > .3
        if left < .08 and not rolling_end:
            if not final_phase and speed < .05:
                self.index = last+1
                self.integral = 0.
                return hold
            if final_phase:
                self.path = None
            return hold
        if wanted != self.gear:
            if speed > .05:
                return hold
            self.gear, self.integral = wanted, 0.
        reach = self.arc[self.index]+self.lookahead
        point = int(min(np.searchsorted(self.arc, reach), last))
        dx, dy = self.path[point]-rear[:2]
        c, s = math.cos(rear[2]), math.sin(rear[2])
        ahead, side = c*dx+s*dy, -s*dx+c*dy
        distance = max(ahead*ahead+side*side, .25)
        curvature = 2*side/distance
        steer = math.copysign(float(np.interp(abs(curvature), CURVATURE, STEER)), curvature)
        target = max(float(self.speed[self.index]), self.min_speed)
        if not rolling_end:
            target = min(target, math.sqrt(2*self.stop_decel*left)+.1)
        error = target-speed
        self.integral = float(np.clip(self.integral+error/FPS, -.5, .5))
        drive = .3+.5*error+.4*self.integral
        if error < -.25:
            return carla.VehicleControl(throttle=0., brake=float(min(1., -.6*error)), steer=steer, reverse=self.gear < 0)
        return carla.VehicleControl(throttle=float(np.clip(drive, 0., .75)), brake=0., steer=steer, reverse=self.gear < 0)


class Bench:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client('127.0.0.1', args.port)
        self.client.set_timeout(60.)
        versions = self.client.get_client_version(), self.client.get_server_version()
        if versions != (CARLA_VERSION, CARLA_VERSION):
            sys.exit('The benchmark needs CARLA {}, found client {} and server {}'.format(CARLA_VERSION, *versions))
        self.world = self.client.load_world(args.map)
        # The map bakes in parked cars, two of them inside goal slots 2-7 and 3-3.
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1./FPS
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        library = self.world.get_blueprint_library()
        self.library = library
        start = carla.Transform(carla.Location(x=EGO_X, y=-243.729996, z=EGO_Z), carla.Rotation(yaw=90.))
        self.ego = self.world.spawn_actor(library.find('vehicle.tesla.model3'), start)
        self.sensors, self.queues = [], {}
        self.collided, self.other = False, None
        collision = self.world.spawn_actor(library.find('sensor.other.collision'), carla.Transform(), attach_to=self.ego)
        collision.listen(self.on_collision)
        self.sensors.append(collision)
        for name in CAMERAS:
            blueprint = library.find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '400')
            blueprint.set_attribute('image_size_y', '300')
            blueprint.set_attribute('fov', '100')
            x, y, z, yaw = MOUNTS[name]
            self.attach(name, blueprint, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)))
        self.attach('imu', library.find('sensor.other.imu'), carla.Transform())
        self.npcs = []
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=283.825165, y=-210.039487, z=35.), carla.Rotation(pitch=-90.)))
        cloud, azimuth, altitude, fog, wet, rain, deposits = WEATHERS[args.weather]
        self.world.set_weather(carla.WeatherParameters(
            cloudiness=cloud, precipitation=rain, precipitation_deposits=deposits, sun_azimuth_angle=azimuth,
            sun_altitude_angle=altitude, fog_density=fog, wetness=wet))
        self.data = None
        self.link = Client(('127.0.0.1', args.server_port), authkey=b'structure-flow-benchmark')

    def attach(self, name, blueprint, mount):
        sensor = self.world.spawn_actor(blueprint, mount, attach_to=self.ego)
        self.queues[name] = queue.Queue()
        sensor.listen(self.queues[name].put)
        self.sensors.append(sensor)

    def on_collision(self, event):
        self.collided = True
        self.other = event.other_actor.type_id

    def next_slot(self, seed, goal_index, start):
        """start_next_slot of the DDP evaluator: the ego leaves the old goal before the new cars spawn."""
        for actor in self.npcs:
            actor.destroy()
        self.npcs = []
        self.ego.set_transform(start)
        free = self.spawn_npcs(seed, goal_index)
        self.ego.set_transform(start)
        self.ego.apply_control(carla.VehicleControl())
        self.ego.set_target_velocity(carla.Vector3D())
        return free

    def spawn_npcs(self, seed, goal_index):
        """init_static_npc of the DDP world, in the same random order."""
        random.seed(seed)
        spots = list(SPOTS)
        goal = spots[goal_index]
        count = random.randint(len(spots)//3, len(spots)-1)
        order = list(spots)
        random.shuffle(order)
        blueprints = [b for b in self.library.filter('vehicle') if int(b.get_attribute('number_of_wheels')) == 4]
        free = []
        for spot in order[:count]:
            if spot == goal:
                free.append(spot)
                continue
            transform = carla.Transform(carla.Location(x=spot[0], y=spot[1], z=SPOT_Z),
                                        carla.Rotation(yaw=random.choice((180., 0.))))
            actor = self.world.try_spawn_actor(random.choice(blueprints), transform)
            if actor is None:
                free.append(spot)
            else:
                actor.set_simulate_physics(False)
                self.npcs.append(actor)
        return free+order[count:]

    def read(self, frame):
        """The frame of the evaluator: vehicle state and the sensors of this very tick."""
        data = dict(transform=self.ego.get_transform(), velocity=self.ego.get_velocity(), control=self.ego.get_control(),
                    spin=self.ego.get_angular_velocity())
        for name, stream in self.queues.items():
            item = stream.get(timeout=20.)
            while item.frame < frame:
                item = stream.get(timeout=20.)
            data[name] = item
        velocity = data['velocity']
        data['speed'] = math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
        return data

    def track(self, goal, tracked):
        """Target in the vehicle frame as the DDP agent makes it: position dead reckoned at 10 Hz, yaw measured."""
        data = self.data
        location, velocity, accelerometer = data['transform'].location, data['velocity'], data['imu'].accelerometer
        heading = data['transform'].rotation.yaw
        if not tracked:
            tracked[:] = [location.x, location.y]*2
        yaw = math.radians(heading)
        c, s = math.cos(yaw), math.sin(yaw)
        tracked[0] += velocity.x*TRACK_DT+.5*(accelerometer.x*c+accelerometer.y*s)*TRACK_DT**2
        tracked[1] += velocity.y*TRACK_DT+.5*(-accelerometer.x*s+accelerometer.y*c)*TRACK_DT**2
        # get_velocity belongs to a point ahead of the actor origin, so a turn shifts the plain sum by about 0.3 m.
        turn = math.radians(data['spin'].z)*VELOCITY_POINT
        tracked[2] += (velocity.x+turn*s)*TRACK_DT
        tracked[3] += (velocity.y-turn*c)*TRACK_DT
        self.motion.append((location.x, location.y, heading, velocity.x, velocity.y, accelerometer.x, accelerometer.y,
                            data['spin'].z, tracked[0], tracked[1], tracked[2], tracked[3]))
        known = dict(gt=[location.x, location.y], corrected=tracked[2:], tracking=tracked)[self.args.target]
        dx, dy = goal['x']-known[0], goal['y']-known[1]
        difference = goal['yaw']-heading
        difference = difference-360 if difference > 180 else difference+360 if difference < -180 else difference
        return [dx*c+dy*s, -dx*s+dy*c, difference]

    def plan(self, target):
        data = self.data
        images = np.stack([np.frombuffer(data[name].raw_data, np.uint8).reshape(300, 400, 4)[:, :, 2::-1]
                           for name in CAMERAS])
        control = data['control']
        state = dict(target=target, speed=-data['speed'] if control.reverse else data['speed'],
                     steer=control.steer, reverse=bool(control.reverse))
        self.link.send_bytes(json.dumps(state).encode())
        self.link.send_bytes(np.ascontiguousarray(images).tobytes())
        return json.loads(self.link.recv_bytes().decode())

    def episode(self, start, goal, free, name):
        """One pass of the DDP loop per frame: world tick, agent on the previous frame, evaluator on this one."""
        args = self.args
        tracker, tracked = Tracker(), []
        counters = dict(in_goal=0, near_goal=0, near_other=0, outside=0)
        trace, plans, seconds = [], [], []
        next_plan, outcome, step, total = HOLD_FRAMES, None, -1, 0
        self.other, self.motion = None, []
        while outcome is None:
            frame = self.world.tick()
            step += 1
            if step < HOLD_FRAMES:
                self.ego.apply_control(carla.VehicleControl())
                self.ego.set_transform(start)
            else:
                if step % PROCESS_FRAMES == 0:
                    target = self.track(goal, tracked)
                    if step >= next_plan or (tracker.path is not None and tracker.remaining() < .1 and tracker.speed[-1] > .3):
                        reply = self.plan(target)
                        seconds.append(reply['seconds'])
                        rear = rear_pose(self.data['transform'])
                        plans.append(dict(frame=step, h=reply.get('h'), stop=reply['stop'], blocked=reply['blocked'],
                                          safe=reply['safe'], p=[round(p, 3) for p in reply['probabilities']], target=target,
                                          mask=reply.get('mask')))
                        if args.save_paths:
                            plans[-1].update(rear=rear.tolist(), path=reply.get('path'))
                        if reply['stop']:
                            tracker.clear()
                            next_plan = step+FPS
                        else:
                            tracker.set_path(rear, reply['path'])
                            next_plan = step+int(args.replan_steps*STEP_SECONDS*FPS)
                speed = self.ego.get_velocity()
                speed = math.sqrt(speed.x**2+speed.y**2+speed.z**2)
                if tracker.path is None and step+1 < next_plan and speed < .02:
                    next_plan = min(next_plan, step+FPS)
                self.ego.apply_control(tracker.control(rear_pose(self.ego.get_transform()), speed))
            total += 1
            self.data = self.read(frame)
            outcome, final = self.judge(self.data, goal, free, counters, total)
            location, control = self.data['transform'].location, self.data['control']
            trace.append((total, location.x, location.y, self.data['transform'].rotation.yaw,
                          -self.data['speed'] if control.reverse else self.data['speed'], control.steer, control.throttle, control.brake))
        np.save(os.path.join(args.out, name+'.npy'), np.array(trace, np.float32))
        if args.save_motion:
            np.save(os.path.join(args.out, name+'.motion.npy'), np.array(self.motion, np.float64))
        with open(os.path.join(args.out, name+'.json'), 'w') as stream:
            json.dump(dict(outcome=outcome, plans=plans, final=final, other=self.other), stream)
        return outcome, total, final, seconds

    def judge(self, data, goal, free, counters, total):
        """tick and eva_check_goal of the DDP evaluator. Distances are 3D to the spots at z = 0.3."""
        location, yaw, control = data['transform'].location, data['transform'].rotation.yaw, data['control']
        x, y = location.x, location.y
        distance_to = lambda p: math.sqrt((x-p[0])**2+(y-p[1])**2+(location.z-SPOT_Z)**2)
        closest = min(free, key=distance_to)
        yaw_error = min(abs(yaw), 180-abs(yaw))
        final = dict(x=x, y=y, z=location.z, yaw=yaw, distance=distance_to(closest), yaw_error=yaw_error,
                     goal_distance=math.hypot(x-goal['x'], y-goal['y']))
        if self.collided:
            self.collided = False
            return 'collision', final
        if total > TOTAL_FRAMES:
            return 'timeout', final
        outside = x < BOUND['x_min'] or x > BOUND['x_max'] or y < BOUND['y_min'] or y > BOUND['y_max']
        counters['outside'] = counters['outside']+1 if outside else 0
        if counters['outside'] > OUT_FRAMES:
            return 'outbound', final
        if not (control.throttle == 0. and data['speed']*3.6 < 1e-3 and control.reverse):
            counters.update(in_goal=0, near_goal=0, near_other=0)
            return None, final
        target = closest[0] == goal['x'] and closest[1] == goal['y']
        dx, dy = abs(x-closest[0]), abs(y-closest[1])
        if dx <= X_TOL and dy <= Y_TOL and yaw_error <= YAW_TOL:
            counters['in_goal'] += 1
        if counters['in_goal'] > IN_GOAL_FRAMES:
            return ('target_success' if target else 'non_target_success'), final
        if X_TOL < dx <= 2*X_TOL or Y_TOL < dy <= 2*Y_TOL or YAW_TOL < yaw_error <= 2*YAW_TOL:
            counters['near_goal' if target else 'near_other'] += 1
        if counters['near_goal'] > NEAR_FRAMES:
            return 'target_fail', final
        if counters['near_other'] > NEAR_FRAMES:
            return 'non_target_fail', final
        return None, final

    def close(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        for actor in self.sensors+self.npcs+[self.ego]:
            actor.destroy()
        self.link.close()


def rear_pose(transform):
    yaw = math.radians(transform.rotation.yaw)
    location = transform.location
    return np.array([location.x-REAR_OFFSET*math.cos(yaw), location.y-REAR_OFFSET*math.sin(yaw), yaw])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--slots', default='', help='comma separated names, all 16 by default')
    parser.add_argument('--starts', default='0,1,2,3,4,5')
    parser.add_argument('--weather', choices=sorted(WEATHERS), default='same')
    parser.add_argument('--replan-steps', type=float, default=4)
    parser.add_argument('--save-paths', action='store_true', help='keep every planned path in the episode json')
    parser.add_argument('--save-motion', action='store_true', help='keep the 10 Hz dead reckoning inputs of every episode')
    parser.add_argument('--target', choices=('tracking', 'gt', 'corrected'), default='tracking',
                        help='tracking and gt are target types of the DDP agent, corrected removes the velocity point offset')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--map', default='Town04_Opt')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--server-port', type=int, default=6100)
    args = parser.parse_args()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    wanted = [name for name in args.slots.split(',') if name] or SLOT_NAMES
    starts = [int(value) for value in args.starts.split(',')]
    bench = Bench(args)
    rows_path = os.path.join(args.out, 'episodes.csv')
    fields = ('epoch', 'slot', 'start', 'outcome', 'seconds', 'position_error', 'yaw_error', 'goal_distance', 'inference_ms')
    done = set()
    if os.path.exists(rows_path):
        with open(rows_path) as stream:
            done = {(int(r['epoch']), r['slot'], int(r['start'])) for r in csv.DictReader(stream)}
    try:
        with open(rows_path, 'a', newline='') as stream:
            writer = csv.DictWriter(stream, fields)
            if not done:
                writer.writeheader()
            for epoch in range(args.epochs):
                for number, (name, index) in enumerate(zip(SLOT_NAMES, GOAL_INDICES)):
                    left = [s for s in starts if (epoch, name, s) not in done]
                    if name not in wanted or not left:
                        continue
                    spot = SPOTS[index]
                    goal = dict(x=spot[0], y=spot[1], yaw=180. if index <= 30 else 0.)    # the eva_parking_goal property
                    free = None
                    for start_index in left:
                        y = spot[1]-8+3.2*start_index
                        start = carla.Transform(carla.Location(x=EGO_X, y=y, z=EGO_Z),
                                                carla.Rotation(yaw=90. if start_index < 3 else -90.))
                        if free is None:
                            free = bench.next_slot(args.seed+number, index, start)
                        else:
                            bench.ego.set_transform(start)
                            bench.ego.apply_control(carla.VehicleControl())
                        began = time.time()
                        label = 'e{}_{}_{}'.format(epoch, name, start_index)
                        outcome, frames, final, seconds = bench.episode(start, goal, free, label)
                        writer.writerow(dict(epoch=epoch, slot=name, start=start_index, outcome=outcome,
                                             seconds=round(frames/FPS, 2), position_error=round(final['distance'], 3),
                                             yaw_error=round(final['yaw_error'], 2),
                                             goal_distance=round(final['goal_distance'], 3),
                                             inference_ms=round(1000*float(np.mean(seconds)), 1) if seconds else ''))
                        stream.flush()
                        print(label, outcome, 'sim {:.1f}s wall {:.0f}s goal {:.2f}m yaw {:.1f}'.format(
                            frames/FPS, time.time()-began, final['goal_distance'], final['yaw_error']), flush=True)
    finally:
        bench.close()


if __name__ == '__main__':
    main()
