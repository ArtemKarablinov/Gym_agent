import sys
import gym
import time
import datetime

print(sys.argv)

assert len(sys.argv) in (2, 3), "pls, run that: python3 keyboard_agent.py env_name [frames_per_second]"
env_name = sys.argv[1]
frames_per_second = int(sys.argv[2]) if len(sys.argv) > 2 else 25

now = datetime.datetime.now()

env = gym.make(env_name)
env = gym.wrappers.Monitor(env, '{}_{}'.format(env_name, now))

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')

ACTIONS = env.action_space.n
SKIP_CONTROL = 0

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

mapping = {
    65361: 3,  # left
    65363: 2,  # right
    65362: 1,  # up
    65364: 4,  # down
}


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 0xff0d:
        human_wants_restart = True
    if key == 32:
        human_sets_pause = not human_sets_pause
    if key in mapping:
        a = mapping[key]
    else:
        a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS:
        return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    if key in mapping:
        a = mapping[key]
    else:
        a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS:
        return
    if human_agent_action == a:
        human_agent_action = 0


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if not window_still_open:
            return False
        if done:
            break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(1./frames_per_second)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


print("ACTIONS={}".format(ACTIONS))
print("Press arrows or keyboard buttons from 1 to {}".format(ACTIONS))
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if not window_still_open:
        break
env.env.close()