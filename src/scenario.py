from collections import deque
import numpy as np

class Stats:
  '''
  Simulate time statistics
  '''

  def __init__(self, n_choices):
    self.time_gap = lambda: np.random.poisson(10)
    self.expire_after = 180
    self.history = deque()

    self.vct = np.array([0] * n_choices)
    self.time = 0

    # init time for the first event
    self.time_pass()

  def time_pass(self):
    # random time pass
    self.time += self.time_gap()
    while self.history \
      and self.history[0]['time'] + self.expire_after <= self.time:
      recomm = self.history.popleft()
      self.vct[recomm['action']] -= 1

  def update(self, action):
    if action is None:
      return

    self.history.append({
      'action': action,
      'time': self.time
    })
    self.vct[action] += 1

  def reset(self):
    self.time = self.time_gap()
    self.vct.fill(0)
    self.history = deque()


class TaskProfile:
  def __init__(self, ctx_size, n_choices):
    self.weight = np.concatenate([
      np.random.randn(n_choices, ctx_size),
      np.random.normal(-4, .5, (n_choices, n_choices))
    ], axis=1)

    self.stats = Stats(n_choices)

    self.choice_history = []

class Scenario:
  '''
  Linear scenario
  '''

  def __init__(self, ctx_size, n_choices, n_tasks=1, noise_scale=0.1):
    self.ctx_size = ctx_size
    self.n_choices = n_choices
    self.n_tasks = n_tasks

    self.global_weight = np.concatenate([
      np.random.randn(n_choices, ctx_size),
      np.random.normal(-5, 1, (n_choices, n_choices))
    ], axis=1)

    self.task_profiles = [TaskProfile(ctx_size, n_choices) for i in range(n_tasks)]

    self.ctx = None

    self.noise_scale = noise_scale
    self.noise = lambda: np.random.normal(scale=noise_scale)

    self.active_n_tasks = n_tasks


  @property
  def payload(self):
    return {
      'ctx_size': self.ctx_size,
      'n_choices': self.n_choices,
      'noise_scale': self.noise_scale,
      'global_weight': self.global_weight.tolist(),
      'task_profiles': [u.weight for u in self.task_profiles]
    }

  @classmethod
  def load(cls, payload):
    scenario = cls(payload['ctx_size'], payload['n_choices'], len(payload['task_profiles']), payload['noise_scale'])
    scenario.global_weight = np.array(payload['global_weight'])
    scenario.task_profiles = [TaskProfile(payload['ctx_size'], payload['n_choices']) for _ in payload['task_profiles']]
    for u, weight in zip(scenario.task_profiles, payload['task_profiles']):
      u.weight = weight

    return scenario

  def next_event(self):
    ''' Sample the next event and return it '''

    # choose the task with the earliest timestamp
    task_idx = np.argmin([u.stats.time for u in self.task_profiles[:self.active_n_tasks]])
    task = self.task_profiles[task_idx]
    time = task.stats.time

    ctx = np.concatenate([
      np.random.normal(0, 1, self.ctx_size),
      task.stats.vct
    ])

    session = ScenarioSession(self, task_idx, ctx, time)

    return session

  def reward(self, choice):
    # return self.weight[choice] @ self.ctx + self.noise()
    pass

  def insight(self, task_idx, ctx, choice):
    ''' Return both the reward & regret '''

    task = self.task_profiles[task_idx]

    truth = [v + self.noise() for v in (task.weight + self.global_weight) @ ctx]

    # append zero for no feedback
    truth.append(0)
    opt_reward = max(truth)
    if choice is not None:
      reward = truth[choice]
    else:
      reward = 0

    return reward, opt_reward - reward

  def reset_stats(self):
    for u in self.task_profiles:
      u.stats.reset()


class ScenarioSession:
  '''
  Session of each scenario ctx event to be used in simulator
  '''
  def __init__(self, scenario, task_idx, ctx, time):
    self.scenario = scenario
    self.task_idx = task_idx
    self.ctx = ctx
    self.time = time

    self.choice, self.reward, self.regret = None, None, None

  @property
  def task(self):
    return self.scenario.task_profiles[self.task_idx]

  def recomm(self, choice):
    ''' Recommend the choice & Return both the reward & regret '''

    assert self.choice is None

    self.choice = choice

    self.reward, self.regret = self.scenario.insight(self.task_idx, self.ctx, choice)

    self.task.stats.update(choice)
    self.task.stats.time_pass()

    return self.reward, self.regret

