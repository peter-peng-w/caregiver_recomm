from collections import deque
from datetime import timedelta, datetime
import numpy as np


class Stats:
    '''
    Past action statistics
    '''

    def __init__(
        self,
        n_choices,
        expire_after=180
    ):
        # expiration time for the action
        self.expire_after = timedelta(seconds=expire_after)
        self.history = deque()                  # store the history of recommendations
        self.vct = np.array([0] * n_choices)    # counter of the active recommendations

    def refresh_vct(self, time=None):
        if not time:
            time = datetime.now()

        # For each recommendation in history, if it is already expired,
        # then remove it from the deque and also update the counter
        while self.history and self.history[0]['time'] + self.expire_after <= time:
            recomm = self.history.popleft()     # remove expired recommendaiton
            self.vct[recomm['action']] -= 1     # udapte the vector

    def update(self, action, time=None):
        ''' Update the statistics when new recommendation comes
        Input Param:
            action: new recommendation (action index), type: int
            time:   recommendation send time, type: datetime
        '''
        # If no recommendation action is triggered (could be rejected),
        # then skip the update procedure and directly return
        if action is None:
            return
        # If no time is provided, then set it to be the current time
        if not time:
            time = datetime.now()
        # Update the history deque, each element is a combination of action id and timestamp
        self.history.append({
            'action': action,
            'time': time
        })
        self.vct[action] += 1       # update the counter

    def reset(self):
        self.vct.fill(0)            # reset the counter
        self.history = deque()      # flush the cache of history
